// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/estimators/solvers/relpose_one_sided_focal.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/math/random_eigen.h"
#include "colmap/scene/camera.h"
#include "colmap/sensor/models.h"
#include "colmap/util/eigen_alignment.h"

#include <algorithm>
#include <limits>
#include <optional>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

using M_t = RelativePoseOneSidedFocalEstimator::M_t;

// The unknown focal of the first (uncalibrated) view. The second view is
// calibrated and enters through a real camera, whose unprojection Jacobian lets
// its share of the residual be measured in its own pixels.
constexpr double kFocal1 = 1000.0;
constexpr double kFocal2 = 800.0;
constexpr size_t kWidth2 = 1600;
constexpr size_t kHeight2 = 1200;

// The calibrated second camera. A distortion-free model keeps the projection
// round trip used to build exact correspondences exact.
Camera TestCamera2(
    const CameraModelId model_id = PinholeCameraModel::model_id) {
  return Camera::CreateFromModelId(
      /*camera_id=*/2, model_id, kFocal2, kWidth2, kHeight2);
}

// Rejection thresholds used to condition a minimal sample: minimum depth in
// front of the second camera and minimum parallax (sin^2 of the ray angle).
constexpr double kMinDepth = 0.5;
constexpr double kMinParallax = 1e-2;  // ~5.7 degrees.

// Random relative pose with a unit-norm baseline (away from the pure-rotation
// degeneracy) and a rotation bounded so the two view frustums overlap, which
// keeps the sampled points in front of both cameras.
Rigid3d TestCam2FromCam1(const double max_angle_deg = 60.0) {
  const Eigen::Vector3d axis = RandomEigenVectord<3>().normalized();
  return Rigid3d(
      Eigen::Quaterniond(Eigen::AngleAxisd(
          DegToRad(RandomUniformReal<double>(0.0, max_angle_deg)), axis)),
      RandomEigenVectord<3>().normalized());
}

// Generates correspondences from random 3D points: centered image points scaled
// by `focal1` for the uncalibrated view, and for the calibrated one the pixel
// it projects to together with the bearing and unprojection Jacobian read back
// from `camera2`. Clearing require_front_of_cam2 admits rays at any angle, as a
// fisheye or spherical second camera would observe.
//
// The second view is round-tripped through the camera rather than synthesized
// as a bare bearing, because the residual now needs d(ray2)/d(pixel2), which
// only the camera model can supply.
void RandomOneSidedFocalCorrespondences(
    const Rigid3d& cam2_from_cam1,
    const Camera& camera2,
    const double focal1,
    const size_t num_points,
    const bool reject_degenerate,
    std::vector<Eigen::Vector2d>& points1,
    std::vector<CamRayWithJac>& cam_rays2_with_jac,
    std::vector<Eigen::Vector2d>* img_points2 = nullptr,
    const bool require_front_of_cam2 = true) {
  for (size_t i = 0; i < num_points; ++i) {
    Eigen::Vector3d point_in_cam1;
    Eigen::Vector2d img_point2;
    std::optional<CamRayWithJac> cam_ray2_with_jac;
    bool degenerate;
    do {
      // Moderate field of view (|x/z|, |y/z| <= ~0.5): wide-angle points span a
      // magnitude range that degrades the conditioning of the minimal solve.
      Eigen::Vector3d ray1 = RandomEigenVectord<3>();
      ray1.z() = std::abs(ray1.z()) + 2.0;
      const double depth = RandomUniformReal<double>(1.0, 3.0);
      point_in_cam1 = depth * ray1.normalized();
      const Eigen::Vector3d point_in_cam2 = cam2_from_cam1 * point_in_cam1;
      const Eigen::Vector3d ray1_in_cam2 =
          cam2_from_cam1.rotation() * point_in_cam1.normalized();
      const double cos_parallax = ray1_in_cam2.dot(point_in_cam2.normalized());
      // Cheirality in cam1 holds by construction; requiring it in cam2 is only
      // meaningful for a pinhole second view.
      degenerate = 1.0 - cos_parallax * cos_parallax < kMinParallax ||
                   (require_front_of_cam2 && point_in_cam2.z() < kMinDepth);
      cam_ray2_with_jac.reset();
      if (const std::optional<Eigen::Vector2d> xy =
              camera2.ImgFromCam(point_in_cam2);
          xy.has_value()) {
        img_point2 = *xy;
        cam_ray2_with_jac = camera2.CamRayFromImgWithJac(img_point2);
      }
      // An unprojectable or rank-deficient pixel carries no usable Jacobian, so
      // it is always resampled, whatever the caller asked for.
      degenerate = degenerate || !cam_ray2_with_jac.has_value();
    } while (degenerate &&
             (reject_degenerate || !cam_ray2_with_jac.has_value()));
    points1.emplace_back(focal1 * point_in_cam1.x() / point_in_cam1.z(),
                         focal1 * point_in_cam1.y() / point_in_cam1.z());
    cam_rays2_with_jac.push_back(*cam_ray2_with_jac);
    if (img_points2 != nullptr) {
      img_points2->push_back(img_point2);
    }
  }
}

// Whether at least one model recovers the essential matrix (up to scale/sign)
// and the unknown focal length, with small residuals on the exact
// correspondences. Returns a bool rather than asserting, so that
// FullSphereCalibratedRays can tolerate the solver's intrinsic failure rate
// over many draws.
bool HasValidModel(const std::vector<Eigen::Vector2d>& points1,
                   const std::vector<CamRayWithJac>& cam_rays2_with_jac,
                   const Eigen::Matrix3d& expected_E,
                   const double expected_focal,
                   const std::vector<M_t>& models,
                   const double E_eps = 5e-3,
                   const double focal_rel_eps = 1e-2,
                   const double r_eps = 1e-2) {
  const Eigen::Matrix3d expected_E_n = expected_E.normalized();
  for (const M_t& model : models) {
    const Eigen::Matrix3d E = model.E.normalized();
    if (std::min((E - expected_E_n).norm(), (E + expected_E_n).norm()) >
        E_eps) {
      continue;
    }
    if (std::abs(model.focal - expected_focal) / expected_focal >
        focal_rel_eps) {
      continue;
    }
    std::vector<double> residuals;
    RelativePoseOneSidedFocalEstimator::Residuals(
        points1, cam_rays2_with_jac, model, &residuals);
    if (std::any_of(residuals.begin(),
                    residuals.end(),
                    [r_eps](const double r) { return r >= r_eps; })) {
      continue;
    }
    return true;
  }
  return false;
}

// The minimal 6-point solver recovers the pose and the unknown focal on clean
// samples.
TEST(RelativePoseOneSidedFocalEstimator, Nominal) {
  SetPRNGSeed(0);
  const Camera camera2 = TestCamera2();
  for (size_t k = 0; k < 100; ++k) {
    const Rigid3d cam2_from_cam1 = TestCam2FromCam1();
    const Eigen::Matrix3d expected_E = EssentialMatrixFromPose(cam2_from_cam1);
    std::vector<Eigen::Vector2d> points1;
    std::vector<CamRayWithJac> cam_rays2_with_jac;
    RandomOneSidedFocalCorrespondences(
        cam2_from_cam1,
        camera2,
        kFocal1,
        RelativePoseOneSidedFocalEstimator::kMinNumSamples,
        /*reject_degenerate=*/true,
        points1,
        cam_rays2_with_jac);

    std::vector<M_t> models;
    RelativePoseOneSidedFocalEstimator::Estimate(
        points1, cam_rays2_with_jac, &models);

    EXPECT_TRUE(HasValidModel(
        points1, cam_rays2_with_jac, expected_E, kFocal1, models));
  }
}

// Residuals are squared tangent Sampson errors in pixels: they vanish on exact
// correspondences and otherwise measure how far the correspondence sits from
// the epipolar variety, in the *joint* pixel space of both views. That is what
// makes the pixel-valued RANSAC threshold meaningful; in ray units they would
// be ~1/focal of this and any correspondence, however wrong, would pass.
// Degenerate inputs -- a non-positive focal, and a zero ray left by a failed
// unprojection -- are scored as outliers rather than as perfect fits.
//
// The gradients are rebuilt here as the implementation does, so the scaling
// checks pin the residual's units rather than the formula; the formula is
// pinned by the exact correspondences, which come from projected 3D points.
TEST(RelativePoseOneSidedFocalEstimator, Residuals) {
  SetPRNGSeed(0);
  const Camera camera2 = TestCamera2();
  const Rigid3d cam2_from_cam1 = TestCam2FromCam1();
  std::vector<Eigen::Vector2d> points1;
  std::vector<CamRayWithJac> cam_rays2_with_jac;
  std::vector<Eigen::Vector2d> img_points2;
  RandomOneSidedFocalCorrespondences(cam2_from_cam1,
                                     camera2,
                                     kFocal1,
                                     30,
                                     /*reject_degenerate=*/false,
                                     points1,
                                     cam_rays2_with_jac,
                                     &img_points2);

  M_t model;
  model.E = EssentialMatrixFromPose(cam2_from_cam1);
  model.focal = kFocal1;

  std::vector<double> residuals;
  RelativePoseOneSidedFocalEstimator::Residuals(
      points1, cam_rays2_with_jac, model, &residuals);
  for (const double residual : residuals) {
    EXPECT_LT(std::sqrt(residual), 1e-6);
  }

  const Eigen::Matrix3d M = model.E * Eigen::DiagonalMatrix<double, 3>(
                                          1.0 / kFocal1, 1.0 / kFocal1, 1.0);
  // Constraint gradient of correspondence i: view 1 in head<2>, view 2 in
  // tail<2>, each in its own view's pixels.
  const auto Gradient = [&](const size_t i) {
    Eigen::Vector4d g;
    g.head<2>() = (M.transpose() * cam_rays2_with_jac[i].ray).head<2>();
    g.tail<2>() = cam_rays2_with_jac[i].jacobian.transpose() *
                  (M * points1[i].homogeneous());
    return g;
  };

  // Displacing a correspondence along the unit normal of the constraint, split
  // across the two views in proportion to their gradients, moves it exactly
  // that far off the variety. This is the defining property of the residual: a
  // distance, in pixels, in the joint measurement space. The offsets stay small
  // because the constraint is only linear in the second view's pixel to first
  // order, so the agreement is first-order rather than exact.
  for (const double offset_pixels : {0.05, 0.2, 1.0}) {
    std::vector<Eigen::Vector2d> shifted_points1 = points1;
    std::vector<CamRayWithJac> shifted_rays2 = cam_rays2_with_jac;
    for (size_t i = 0; i < points1.size(); ++i) {
      const Eigen::Vector4d g = Gradient(i);
      const double scale = offset_pixels / g.norm();
      shifted_points1[i] += scale * g.head<2>();
      shifted_rays2[i] =
          *camera2.CamRayFromImgWithJac(img_points2[i] + scale * g.tail<2>());
    }
    RelativePoseOneSidedFocalEstimator::Residuals(
        shifted_points1, shifted_rays2, model, &residuals);
    for (const double residual : residuals) {
      EXPECT_NEAR(std::sqrt(residual), offset_pixels, 1e-2 * offset_pixels);
    }
  }

  // A displacement confined to the calibrated view is a real error and is
  // scored as one. Measuring only the first view's distance to its epipolar
  // line would report every one of these as a perfect fit.
  constexpr double kOffsetPixels = 2.0;
  std::vector<CamRayWithJac> view2_shifted_rays2 = cam_rays2_with_jac;
  for (size_t i = 0; i < points1.size(); ++i) {
    view2_shifted_rays2[i] = *camera2.CamRayFromImgWithJac(
        img_points2[i] + kOffsetPixels * Gradient(i).tail<2>().normalized());
  }
  RelativePoseOneSidedFocalEstimator::Residuals(
      points1, view2_shifted_rays2, model, &residuals);
  for (size_t i = 0; i < points1.size(); ++i) {
    const Eigen::Vector4d g = Gradient(i);
    const double expected = kOffsetPixels * g.tail<2>().norm() / g.norm();
    // The second view carries a substantial share, so this is not vacuous.
    EXPECT_GT(expected, 0.1 * kOffsetPixels);
    EXPECT_NEAR(std::sqrt(residuals[i]), expected, 1e-2 * expected);
  }

  M_t invalid_model = model;
  invalid_model.focal = 0.0;
  RelativePoseOneSidedFocalEstimator::Residuals(
      points1, cam_rays2_with_jac, invalid_model, &residuals);
  for (const double residual : residuals) {
    EXPECT_EQ(residual, std::numeric_limits<double>::max());
  }

  std::vector<CamRayWithJac> degenerate_rays2 = cam_rays2_with_jac;
  degenerate_rays2[0] = CamRayWithJac::Zero();
  RelativePoseOneSidedFocalEstimator::Residuals(
      points1, degenerate_rays2, model, &residuals);
  EXPECT_EQ(residuals[0], std::numeric_limits<double>::max());
  EXPECT_LT(std::sqrt(residuals[1]), 1e-6);
}

// Refinement pulls a perturbed pose + focal back to the ground truth.
TEST(RelativePoseOneSidedFocalEstimator, RefineFromInitialModel) {
  SetPRNGSeed(0);
  const Camera camera2 = TestCamera2();
  for (size_t k = 0; k < 50; ++k) {
    const Rigid3d cam2_from_cam1 = TestCam2FromCam1();
    const Eigen::Matrix3d expected_E = EssentialMatrixFromPose(cam2_from_cam1);
    std::vector<Eigen::Vector2d> points1;
    std::vector<CamRayWithJac> cam_rays2_with_jac;
    RandomOneSidedFocalCorrespondences(cam2_from_cam1,
                                       camera2,
                                       kFocal1,
                                       50,
                                       /*reject_degenerate=*/false,
                                       points1,
                                       cam_rays2_with_jac);

    const Eigen::Quaterniond seed_rotation =
        cam2_from_cam1.rotation() *
        Eigen::Quaterniond(
            Eigen::AngleAxisd(0.02, RandomEigenVectord<3>().normalized()));
    const Eigen::Vector3d seed_translation =
        (cam2_from_cam1.translation() + 0.02 * RandomEigenVectord<3>())
            .normalized();
    M_t model;
    model.E = EssentialMatrixFromPose(Rigid3d(seed_rotation, seed_translation));
    model.focal = 1.1 * kFocal1;

    ASSERT_TRUE(RelativePoseOneSidedFocalEstimator::Refine(
        points1, cam_rays2_with_jac, &model));

    const std::vector<M_t> models = {model};
    EXPECT_TRUE(HasValidModel(points1,
                              cam_rays2_with_jac,
                              expected_E,
                              kFocal1,
                              models,
                              /*E_eps=*/1e-3,
                              /*focal_rel_eps=*/1e-2,
                              /*r_eps=*/1e-2));
  }
}

// The calibrated view enters as bearing rays, which span the full sphere rather
// than a pinhole image plane: rays with z <= 0, which only a camera seeing
// beyond 180 degrees observes and which no pinhole image plane can represent,
// must be usable correspondences. An equirectangular second camera supplies
// both those rays and the Jacobians that put its share of the residual in its
// own pixels.
TEST(RelativePoseOneSidedFocalEstimator, FullSphereCalibratedRays) {
  SetPRNGSeed(0);
  const Camera camera2 = TestCamera2(EquirectangularCameraModel::model_id);
  // Without the cheirality filter the configurations are harsher than any
  // pinhole pair, and the minimal solve occasionally loses the true root. Never
  // exceeded 2% over 199 measured runs; a real regression exceeds it at once.
  constexpr size_t kNumTrials = 100;
  constexpr double kMaxFailureRate = 0.03;
  size_t num_failures = 0;
  for (size_t k = 0; k < kNumTrials; ++k) {
    Rigid3d cam2_from_cam1;
    std::vector<Eigen::Vector2d> points1;
    std::vector<CamRayWithJac> cam_rays2_with_jac;
    // Resample until the sample really contains such a ray, so that every trial
    // exercises the property rather than a random subset of them. A large
    // rotation puts a good share of the points outside any pinhole frustum.
    do {
      points1.clear();
      cam_rays2_with_jac.clear();
      cam2_from_cam1 = TestCam2FromCam1(/*max_angle_deg=*/140.0);
      RandomOneSidedFocalCorrespondences(
          cam2_from_cam1,
          camera2,
          kFocal1,
          RelativePoseOneSidedFocalEstimator::kMinNumSamples,
          /*reject_degenerate=*/true,
          points1,
          cam_rays2_with_jac,
          /*img_points2=*/nullptr,
          /*require_front_of_cam2=*/false);
    } while (std::none_of(cam_rays2_with_jac.begin(),
                          cam_rays2_with_jac.end(),
                          [](const CamRayWithJac& cam_ray_with_jac) {
                            return cam_ray_with_jac.ray.z() <= 0.0;
                          }));

    std::vector<M_t> models;
    RelativePoseOneSidedFocalEstimator::Estimate(
        points1, cam_rays2_with_jac, &models);

    if (!HasValidModel(points1,
                       cam_rays2_with_jac,
                       EssentialMatrixFromPose(cam2_from_cam1),
                       kFocal1,
                       models)) {
      ++num_failures;
    }
  }
  EXPECT_LE(static_cast<double>(num_failures) / kNumTrials, kMaxFailureRate);
}

}  // namespace
}  // namespace colmap

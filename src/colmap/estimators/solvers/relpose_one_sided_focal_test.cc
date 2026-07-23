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
#include "colmap/util/eigen_alignment.h"

#include <algorithm>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

using M_t = RelativePoseOneSidedFocalEstimator::M_t;

// The unknown focal of the first (uncalibrated) view. The second view enters as
// calibrated rays and therefore has no focal at all.
constexpr double kFocal1 = 1000.0;

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
// by `focal1` for the uncalibrated view, bearing rays for the calibrated one.
// Clearing require_front_of_cam2 admits rays at any angle, as a fisheye or
// spherical second camera would observe.
void RandomOneSidedFocalCorrespondences(
    const Rigid3d& cam2_from_cam1,
    const double focal1,
    const size_t num_points,
    const bool reject_degenerate,
    std::vector<Eigen::Vector2d>& points1,
    std::vector<Eigen::Vector3d>& cam_rays2,
    const bool require_front_of_cam2 = true) {
  for (size_t i = 0; i < num_points; ++i) {
    Eigen::Vector3d point_in_cam1;
    Eigen::Vector3d point_in_cam2;
    bool degenerate;
    do {
      // Moderate field of view (|x/z|, |y/z| <= ~0.5): wide-angle points span a
      // magnitude range that degrades the conditioning of the minimal solve.
      Eigen::Vector3d ray1 = RandomEigenVectord<3>();
      ray1.z() = std::abs(ray1.z()) + 2.0;
      const double depth = RandomUniformReal<double>(1.0, 3.0);
      point_in_cam1 = depth * ray1.normalized();
      point_in_cam2 = cam2_from_cam1 * point_in_cam1;
      const Eigen::Vector3d ray1_in_cam2 =
          cam2_from_cam1.rotation() * point_in_cam1.normalized();
      const double cos_parallax = ray1_in_cam2.dot(point_in_cam2.normalized());
      // Cheirality in cam1 holds by construction; requiring it in cam2 is only
      // meaningful for a pinhole second view.
      degenerate = 1.0 - cos_parallax * cos_parallax < kMinParallax ||
                   (require_front_of_cam2 && point_in_cam2.z() < kMinDepth);
    } while (reject_degenerate && degenerate);
    points1.emplace_back(focal1 * point_in_cam1.x() / point_in_cam1.z(),
                         focal1 * point_in_cam1.y() / point_in_cam1.z());
    cam_rays2.push_back(point_in_cam2.normalized());
  }
}

// Whether at least one model recovers the essential matrix (up to scale/sign)
// and the unknown focal length, with small residuals on the exact
// correspondences. Returns a bool rather than asserting, so that
// FullSphereCalibratedRays can tolerate the solver's intrinsic failure rate
// over many draws.
bool HasValidModel(const std::vector<Eigen::Vector2d>& points1,
                   const std::vector<Eigen::Vector3d>& cam_rays2,
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
        points1, cam_rays2, model, &residuals);
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
  for (size_t k = 0; k < 100; ++k) {
    const Rigid3d cam2_from_cam1 = TestCam2FromCam1();
    const Eigen::Matrix3d expected_E = EssentialMatrixFromPose(cam2_from_cam1);
    std::vector<Eigen::Vector2d> points1;
    std::vector<Eigen::Vector3d> cam_rays2;
    RandomOneSidedFocalCorrespondences(
        cam2_from_cam1,
        kFocal1,
        RelativePoseOneSidedFocalEstimator::kMinNumSamples,
        /*reject_degenerate=*/true,
        points1,
        cam_rays2);

    std::vector<M_t> models;
    RelativePoseOneSidedFocalEstimator::Estimate(points1, cam_rays2, &models);

    EXPECT_TRUE(HasValidModel(points1, cam_rays2, expected_E, kFocal1, models));
  }
}

// Residuals are squared distances in first-view pixels: they vanish on exact
// correspondences and scale exactly with a displacement perpendicular to the
// epipolar line. That is what makes the pixel-valued RANSAC threshold
// meaningful; in ray units they would be ~1/focal of this and any
// correspondence, however wrong, would pass. Degenerate inputs -- a
// non-positive focal, and a zero ray left by a failed undistortion -- are
// scored as outliers rather than as perfect fits.
//
// The epipolar line is rebuilt here as the implementation does, so the scaling
// checks pin the residual's units rather than the formula; the formula is
// pinned by the exact correspondences, which come from projected 3D points.
TEST(RelativePoseOneSidedFocalEstimator, Residuals) {
  SetPRNGSeed(0);
  const Rigid3d cam2_from_cam1 = TestCam2FromCam1();
  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector3d> cam_rays2;
  RandomOneSidedFocalCorrespondences(cam2_from_cam1,
                                     kFocal1,
                                     30,
                                     /*reject_degenerate=*/false,
                                     points1,
                                     cam_rays2);

  M_t model;
  model.E = EssentialMatrixFromPose(cam2_from_cam1);
  model.focal = kFocal1;

  std::vector<double> residuals;
  RelativePoseOneSidedFocalEstimator::Residuals(
      points1, cam_rays2, model, &residuals);
  for (const double residual : residuals) {
    EXPECT_LT(std::sqrt(residual), 1e-6);
  }

  // Spans well below and well above the default 4 pixel inlier threshold.
  const Eigen::Matrix3d M = model.E * Eigen::DiagonalMatrix<double, 3>(
                                          1.0 / kFocal1, 1.0 / kFocal1, 1.0);
  for (const double offset_pixels : {0.5, 3.0, 100.0}) {
    std::vector<Eigen::Vector2d> shifted_points1 = points1;
    for (size_t i = 0; i < shifted_points1.size(); ++i) {
      const Eigen::Vector3d line1 = M.transpose() * cam_rays2[i];
      shifted_points1[i] += offset_pixels * line1.head<2>().normalized();
    }
    RelativePoseOneSidedFocalEstimator::Residuals(
        shifted_points1, cam_rays2, model, &residuals);
    for (const double residual : residuals) {
      EXPECT_NEAR(std::sqrt(residual), offset_pixels, 1e-6);
    }
  }

  M_t invalid_model = model;
  invalid_model.focal = 0.0;
  RelativePoseOneSidedFocalEstimator::Residuals(
      points1, cam_rays2, invalid_model, &residuals);
  for (const double residual : residuals) {
    EXPECT_EQ(residual, std::numeric_limits<double>::max());
  }

  std::vector<Eigen::Vector3d> degenerate_rays2 = cam_rays2;
  degenerate_rays2[0] = Eigen::Vector3d::Zero();
  RelativePoseOneSidedFocalEstimator::Residuals(
      points1, degenerate_rays2, model, &residuals);
  EXPECT_EQ(residuals[0], std::numeric_limits<double>::max());
  EXPECT_LT(std::sqrt(residuals[1]), 1e-6);
}

// Refinement pulls a perturbed pose + focal back to the ground truth.
TEST(RelativePoseOneSidedFocalEstimator, RefineFromInitialModel) {
  SetPRNGSeed(0);
  for (size_t k = 0; k < 50; ++k) {
    const Rigid3d cam2_from_cam1 = TestCam2FromCam1();
    const Eigen::Matrix3d expected_E = EssentialMatrixFromPose(cam2_from_cam1);
    std::vector<Eigen::Vector2d> points1;
    std::vector<Eigen::Vector3d> cam_rays2;
    RandomOneSidedFocalCorrespondences(cam2_from_cam1,
                                       kFocal1,
                                       50,
                                       /*reject_degenerate=*/false,
                                       points1,
                                       cam_rays2);

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

    ASSERT_TRUE(
        RelativePoseOneSidedFocalEstimator::Refine(points1, cam_rays2, &model));

    const std::vector<M_t> models = {model};
    EXPECT_TRUE(HasValidModel(points1,
                              cam_rays2,
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
// must be usable correspondences.
TEST(RelativePoseOneSidedFocalEstimator, FullSphereCalibratedRays) {
  SetPRNGSeed(0);
  // Without the cheirality filter the configurations are harsher than any
  // pinhole pair, and the minimal solve occasionally loses the true root. Never
  // exceeded 2% over 199 measured runs; a real regression exceeds it at once.
  constexpr size_t kNumTrials = 100;
  constexpr double kMaxFailureRate = 0.03;
  size_t num_failures = 0;
  for (size_t k = 0; k < kNumTrials; ++k) {
    Rigid3d cam2_from_cam1;
    std::vector<Eigen::Vector2d> points1;
    std::vector<Eigen::Vector3d> cam_rays2;
    // Resample until the sample really contains such a ray, so that every trial
    // exercises the property rather than a random subset of them. A large
    // rotation puts a good share of the points outside any pinhole frustum.
    do {
      points1.clear();
      cam_rays2.clear();
      cam2_from_cam1 = TestCam2FromCam1(/*max_angle_deg=*/140.0);
      RandomOneSidedFocalCorrespondences(
          cam2_from_cam1,
          kFocal1,
          RelativePoseOneSidedFocalEstimator::kMinNumSamples,
          /*reject_degenerate=*/true,
          points1,
          cam_rays2,
          /*require_front_of_cam2=*/false);
    } while (std::none_of(
        cam_rays2.begin(), cam_rays2.end(), [](const Eigen::Vector3d& ray) {
          return ray.z() <= 0.0;
        }));

    std::vector<M_t> models;
    RelativePoseOneSidedFocalEstimator::Estimate(points1, cam_rays2, &models);

    if (!HasValidModel(points1,
                       cam_rays2,
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

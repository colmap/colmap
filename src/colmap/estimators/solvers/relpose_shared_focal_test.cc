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

#include "colmap/estimators/solvers/relpose_shared_focal.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/math/random_eigen.h"
#include "colmap/util/eigen_alignment.h"

#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

using M_t = RelativePoseSharedFocalEstimator::M_t;

// Rejection thresholds used to condition a minimal sample: minimum depth in
// front of either camera and minimum parallax (sin^2 of the ray angle).
constexpr double kMinDepth = 0.5;
constexpr double kMinParallax = 1e-2;  // ~5.7 degrees.

// Random relative pose with a unit-norm baseline (away from the pure-rotation
// degeneracy) and a bounded rotation. The rotation is bounded so the two view
// frustums overlap: this keeps the sampled points cheirality-consistent (in
// front of both cameras) without the rejection sampler spinning for
// near-opposite orientations.
Rigid3d TestCam2FromCam1() {
  const double max_angle_deg = 60.0;
  // Resample until the two optical axes are clearly non-coplanar, i.e. the
  // shared focal is identifiable (the same precondition the estimator enforces
  // via FocalIdentifiability). For a coplanar pose the focal is unrecoverable
  // and the minimal solver returns a meaningless focal, which no downstream
  // assertion can meaningfully check.
  while (true) {
    const Eigen::Vector3d axis = RandomEigenVectord<3>().normalized();
    const Rigid3d cam2_from_cam1(
        Eigen::Quaterniond(Eigen::AngleAxisd(
            DegToRad(RandomUniformReal<double>(0.0, max_angle_deg)), axis)),
        RandomEigenVectord<3>().normalized());
    if (RelativePoseSharedFocalEstimator::FocalIdentifiability(
            cam2_from_cam1) >=
        RelativePoseSharedFocalEstimator::kMinFocalIdentifiability) {
      return cam2_from_cam1;
    }
  }
}

// Generates principal-point-centered image point pairs (f * X / Z) for a shared
// focal length `focal`, from random 3D points in front of both cameras. When
// reject_degenerate is set, resamples points that make a minimal 6-point solve
// ill-conditioned.
void RandomSharedFocalCorrespondences(const Rigid3d& cam2_from_cam1,
                                      const double focal,
                                      const size_t num_points,
                                      const bool reject_degenerate,
                                      std::vector<Eigen::Vector2d>& points1,
                                      std::vector<Eigen::Vector2d>& points2) {
  for (size_t i = 0; i < num_points; ++i) {
    Eigen::Vector3d point_in_cam1;
    Eigen::Vector3d point_in_cam2;
    bool degenerate;
    do {
      // Point in front of cam1 with a moderate field of view (|x/z|, |y/z| <=
      // ~0.5): wide-angle points span a large magnitude range that degrades the
      // conditioning of the minimal solve.
      Eigen::Vector3d ray1 = RandomEigenVectord<3>();
      ray1.z() = std::abs(ray1.z()) + 2.0;
      const double depth = RandomUniformReal<double>(1.0, 3.0);
      point_in_cam1 = depth * ray1.normalized();
      point_in_cam2 = cam2_from_cam1 * point_in_cam1;
      const Eigen::Vector3d ray1_in_cam2 =
          cam2_from_cam1.rotation() * point_in_cam1.normalized();
      const double cos_parallax = ray1_in_cam2.dot(point_in_cam2.normalized());
      // The point is always in front of cam1; require it in front of cam2 too,
      // with sufficient parallax.
      degenerate = point_in_cam2.z() < kMinDepth ||
                   1.0 - cos_parallax * cos_parallax < kMinParallax;
    } while (reject_degenerate && degenerate);
    points1.emplace_back(focal * point_in_cam1.x() / point_in_cam1.z(),
                         focal * point_in_cam1.y() / point_in_cam1.z());
    points2.emplace_back(focal * point_in_cam2.x() / point_in_cam2.z(),
                         focal * point_in_cam2.y() / point_in_cam2.z());
  }
}

// Asserts that at least one model recovers the essential matrix (up to
// scale/sign) and the focal length, with small residuals on the exact points.
void ExpectAtLeastOneValidModel(const std::vector<Eigen::Vector2d>& points1,
                                const std::vector<Eigen::Vector2d>& points2,
                                const Eigen::Matrix3d& expected_E,
                                const double expected_focal,
                                const std::vector<M_t>& models,
                                const double E_eps = 5e-3,
                                const double focal_rel_eps = 5e-3,
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
    RelativePoseSharedFocalEstimator::Residuals(
        points1, points2, model, &residuals);
    for (const double residual : residuals) {
      EXPECT_LT(residual, r_eps);
    }
    return;
  }
  ADD_FAILURE() << "No shared-focal model matches the ground truth.";
}

// The minimal 6-point solver recovers the pose and focal on clean samples.
TEST(RelativePoseSharedFocalEstimator, Nominal) {
  SetPRNGSeed(0);
  const double kFocal = 1000.0;
  for (size_t k = 0; k < 100; ++k) {
    const Rigid3d cam2_from_cam1 = TestCam2FromCam1();
    const Eigen::Matrix3d expected_E = EssentialMatrixFromPose(cam2_from_cam1);
    std::vector<Eigen::Vector2d> points1;
    std::vector<Eigen::Vector2d> points2;
    RandomSharedFocalCorrespondences(
        cam2_from_cam1,
        kFocal,
        RelativePoseSharedFocalEstimator::kMinNumSamples,
        /*reject_degenerate=*/true,
        points1,
        points2);

    std::vector<M_t> models;
    RelativePoseSharedFocalEstimator::Estimate(points1, points2, &models);

    ExpectAtLeastOneValidModel(points1, points2, expected_E, kFocal, models);
  }
}

// Residuals are near-zero on exact points, grow with a wrong focal, and are
// infinite for a non-positive focal.
TEST(RelativePoseSharedFocalEstimator, Residuals) {
  SetPRNGSeed(0);
  const double kFocal = 1000.0;
  const Rigid3d cam2_from_cam1 = TestCam2FromCam1();
  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  RandomSharedFocalCorrespondences(cam2_from_cam1,
                                   kFocal,
                                   30,
                                   /*reject_degenerate=*/false,
                                   points1,
                                   points2);

  M_t model;
  model.E = EssentialMatrixFromPose(cam2_from_cam1);
  model.focal = kFocal;
  std::vector<double> residuals;
  RelativePoseSharedFocalEstimator::Residuals(
      points1, points2, model, &residuals);
  double sum_exact = 0.0;
  for (const double residual : residuals) {
    EXPECT_LT(residual, 1e-2);
    sum_exact += residual;
  }

  M_t wrong_model = model;
  wrong_model.focal = 1.5 * kFocal;
  std::vector<double> wrong_residuals;
  RelativePoseSharedFocalEstimator::Residuals(
      points1, points2, wrong_model, &wrong_residuals);
  double sum_wrong = 0.0;
  for (const double residual : wrong_residuals) {
    sum_wrong += residual;
  }
  EXPECT_GT(sum_wrong, sum_exact);

  M_t invalid_model = model;
  invalid_model.focal = 0.0;
  std::vector<double> invalid_residuals;
  RelativePoseSharedFocalEstimator::Residuals(
      points1, points2, invalid_model, &invalid_residuals);
  for (const double residual : invalid_residuals) {
    EXPECT_EQ(residual, std::numeric_limits<double>::max());
  }
}

// Refinement pulls a perturbed pose + focal back to the ground truth.
TEST(RelativePoseSharedFocalEstimator, RefineFromInitialModel) {
  SetPRNGSeed(0);
  const double kFocal = 1000.0;
  for (size_t k = 0; k < 50; ++k) {
    const Rigid3d cam2_from_cam1 = TestCam2FromCam1();
    const Eigen::Matrix3d expected_E = EssentialMatrixFromPose(cam2_from_cam1);
    std::vector<Eigen::Vector2d> points1;
    std::vector<Eigen::Vector2d> points2;
    RandomSharedFocalCorrespondences(cam2_from_cam1,
                                     kFocal,
                                     50,
                                     /*reject_degenerate=*/false,
                                     points1,
                                     points2);

    const Eigen::Quaterniond seed_rotation =
        cam2_from_cam1.rotation() *
        Eigen::Quaterniond(
            Eigen::AngleAxisd(0.02, RandomEigenVectord<3>().normalized()));
    const Eigen::Vector3d seed_translation =
        (cam2_from_cam1.translation() + 0.02 * RandomEigenVectord<3>())
            .normalized();
    M_t model;
    model.E = EssentialMatrixFromPose(Rigid3d(seed_rotation, seed_translation));
    model.focal = 1.1 * kFocal;

    ASSERT_TRUE(
        RelativePoseSharedFocalEstimator::Refine(points1, points2, &model));

    const std::vector<M_t> models = {model};
    ExpectAtLeastOneValidModel(points1,
                               points2,
                               expected_E,
                               kFocal,
                               models,
                               /*E_eps=*/1e-3,
                               /*focal_rel_eps=*/1e-2,
                               /*r_eps=*/1e-2);
  }
}

// FocalIdentifiability is ~0 for coplanar (intersecting) axes and large for
// skew axes, and is invariant to the translation scale.
TEST(RelativePoseSharedFocalEstimator, FocalIdentifiability) {
  SetPRNGSeed(0);

  // A relative pose whose two optical axes intersect at a common fixation point
  // on cam1's +z axis, so the axes are coplanar and the focal is
  // unidentifiable.
  const auto intersecting_cam2_from_cam1 = [] {
    const Eigen::Vector3d fixation(0, 0, RandomUniformReal<double>(1.5, 3.0));
    Eigen::Vector3d center2 = RandomEigenVectord<3>();
    center2.z() = std::abs(center2.z());
    center2.normalize();
    const Eigen::Quaterniond rotation = Eigen::Quaterniond::FromTwoVectors(
        (fixation - center2).normalized(), Eigen::Vector3d::UnitZ());
    return Rigid3d(rotation, rotation * -center2);
  };

  for (size_t k = 0; k < 10; ++k) {
    EXPECT_LT(RelativePoseSharedFocalEstimator::FocalIdentifiability(
                  intersecting_cam2_from_cam1()),
              1e-9);
  }

  // 90 deg about x with a lateral baseline: maximally skew axes, score 1.
  const Rigid3d skew(Eigen::Quaterniond(Eigen::AngleAxisd(
                         DegToRad(90.0), Eigen::Vector3d::UnitX())),
                     Eigen::Vector3d(1, 0, 0));
  EXPECT_NEAR(
      RelativePoseSharedFocalEstimator::FocalIdentifiability(skew), 1.0, 1e-9);
  // Scaling the translation does not change the score.
  const Rigid3d skew_scaled(skew.rotation(), 1000.0 * skew.translation());
  EXPECT_NEAR(
      RelativePoseSharedFocalEstimator::FocalIdentifiability(skew_scaled),
      1.0,
      1e-9);
}

}  // namespace
}  // namespace colmap

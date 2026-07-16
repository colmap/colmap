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

#include "colmap/estimators/fundamental_matrix_degensac.h"

#include "colmap/estimators/solvers/fundamental_matrix.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/homography_matrix.h"
#include "colmap/math/random.h"
#include "colmap/math/random_eigen.h"
#include "colmap/optim/loransac.h"

#include <array>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

Eigen::Matrix3d RandomCalibrationMatrix() {
  return (Eigen::Matrix3d() << RandomUniformReal<double>(800, 1200),
          0,
          RandomUniformReal<double>(400, 600),
          0,
          RandomUniformReal<double>(800, 1200),
          RandomUniformReal<double>(400, 600),
          0,
          0,
          1)
      .finished();
}

// Generates a two-view scene with a dominant plane. The first `num_on_plane`
// correspondences lie on the plane `normal . X = distance` (in the first
// camera frame); the remaining ones are at random depths off the plane. All
// correspondences are consistent with the true epipolar geometry; only the
// on-plane ones are additionally consistent with the plane homography.
void GenerateDominantPlaneScene(const Rigid3d& cam2_from_cam1,
                                const Eigen::Matrix3d& K,
                                const Eigen::Vector3d& normal,
                                double distance,
                                size_t num_points,
                                size_t num_on_plane,
                                double noise,
                                std::vector<Eigen::Vector2d>* points1,
                                std::vector<Eigen::Vector2d>* points2,
                                std::vector<char>* on_plane_mask) {
  const Eigen::Matrix3d K_inv = K.inverse();
  points1->clear();
  points2->clear();
  on_plane_mask->clear();
  for (size_t i = 0; i < num_points; ++i) {
    const Eigen::Vector2d point1 =
        K.topRows<2>() * RandomEigenVectord<2>().homogeneous();
    const Eigen::Vector3d ray = K_inv * point1.homogeneous();
    const bool on_plane = i < num_on_plane;
    double depth;
    if (on_plane) {
      depth = distance / normal.dot(ray);
    } else {
      depth = RandomUniformReal<double>(0.5, 3.0);
    }
    const Eigen::Vector3d point3D_in_cam1 = depth * ray;
    const Eigen::Vector2d point2 =
        (K * (cam2_from_cam1 * point3D_in_cam1)).hnormalized();
    points1->push_back(point1 + noise * RandomEigenVectord<2>());
    points2->push_back(point2 + noise * RandomEigenVectord<2>());
    on_plane_mask->push_back(on_plane);
  }
}

double MeanSampsonErrorOnSubset(const std::vector<Eigen::Vector2d>& points1,
                                const std::vector<Eigen::Vector2d>& points2,
                                const Eigen::Matrix3d& F,
                                const std::vector<char>& subset_mask) {
  std::vector<double> residuals;
  ComputeSquaredSampsonError(points1, points2, F, &residuals);
  double sum = 0;
  size_t count = 0;
  for (size_t i = 0; i < residuals.size(); ++i) {
    if (subset_mask[i]) {
      sum += residuals[i];
      ++count;
    }
  }
  return count == 0 ? 0.0 : sum / count;
}

RANSACOptions TestRANSACOptions() {
  RANSACOptions options;
  options.max_error = 1.0;
  options.confidence = 0.9999;
  options.min_inlier_ratio = 0.1;
  options.max_num_trials = 10000;
  options.random_seed = 0;
  return options;
}

TEST(EpipoleFromFundamentalMatrix, Nominal) {
  SetPRNGSeed(0);
  for (int k = 0; k < 20; ++k) {
    const Eigen::Matrix3d K = RandomCalibrationMatrix();
    const Rigid3d cam2_from_cam1(RandomEigenQuaterniond(),
                                 RandomEigenVectord<3>());
    const Eigen::Matrix3d F = FundamentalFromEssentialMatrix(
        K, EssentialMatrixFromPose(cam2_from_cam1), K);
    const Eigen::Vector3d epipole2 = EpipoleFromFundamentalMatrix(F);
    EXPECT_NEAR(epipole2.norm(), 1.0, 1e-9);
    EXPECT_LT((F.transpose() * epipole2).norm(), 1e-6);
  }
}

TEST(HomographyFromFundamentalAndPoints, Nominal) {
  SetPRNGSeed(0);
  const Eigen::Matrix3d K = RandomCalibrationMatrix();
  const Rigid3d cam2_from_cam1(RandomEigenQuaterniond(),
                               RandomEigenVectord<3>());
  const Eigen::Matrix3d F = FundamentalFromEssentialMatrix(
      K, EssentialMatrixFromPose(cam2_from_cam1), K);
  const Eigen::Vector3d epipole2 = EpipoleFromFundamentalMatrix(F);

  const Eigen::Vector3d normal = Eigen::Vector3d(0.2, -0.1, 1.0).normalized();
  constexpr double kDistance = 2.0;
  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  std::vector<char> on_plane_mask;
  GenerateDominantPlaneScene(cam2_from_cam1,
                             K,
                             normal,
                             kDistance,
                             /*num_points=*/6,
                             /*num_on_plane=*/6,
                             /*noise=*/0.0,
                             &points1,
                             &points2,
                             &on_plane_mask);

  const std::array<Eigen::Vector2d, 3> tri_points1 = {
      points1[0], points1[1], points1[2]};
  const std::array<Eigen::Vector2d, 3> tri_points2 = {
      points2[0], points2[1], points2[2]};
  const std::optional<Eigen::Matrix3d> H =
      HomographyFromFundamentalAndPoints(F, epipole2, tri_points1, tri_points2);
  ASSERT_TRUE(H.has_value());

  // All on-plane correspondences must be explained by the recovered homography.
  for (size_t i = 0; i < points1.size(); ++i) {
    EXPECT_LT(ComputeSquaredHomographyError(points1[i], points2[i], *H), 1e-6);
  }
}

TEST(HomographyFromFundamentalAndPoints, CollinearReturnsNullopt) {
  SetPRNGSeed(0);
  const Eigen::Matrix3d K = RandomCalibrationMatrix();
  const Rigid3d cam2_from_cam1(RandomEigenQuaterniond(),
                               RandomEigenVectord<3>());
  const Eigen::Matrix3d F = FundamentalFromEssentialMatrix(
      K, EssentialMatrixFromPose(cam2_from_cam1), K);
  const Eigen::Vector3d epipole2 = EpipoleFromFundamentalMatrix(F);

  // Three collinear first-image points make M rank-deficient.
  const std::array<Eigen::Vector2d, 3> tri_points1 = {
      Eigen::Vector2d(100, 201),
      Eigen::Vector2d(200, 401),
      Eigen::Vector2d(300, 601)};
  const std::array<Eigen::Vector2d, 3> tri_points2 = {
      Eigen::Vector2d(110, 210),
      Eigen::Vector2d(220, 430),
      Eigen::Vector2d(330, 650)};
  EXPECT_FALSE(
      HomographyFromFundamentalAndPoints(F, epipole2, tri_points1, tri_points2)
          .has_value());
}

TEST(IsSampleHDegenerate, DetectsAndRejects) {
  SetPRNGSeed(0);
  const Eigen::Matrix3d K = RandomCalibrationMatrix();
  const Rigid3d cam2_from_cam1(RandomEigenQuaterniond(),
                               RandomEigenVectord<3>());
  const Eigen::Matrix3d F = FundamentalFromEssentialMatrix(
      K, EssentialMatrixFromPose(cam2_from_cam1), K);
  const Eigen::Vector3d normal = Eigen::Vector3d(0.2, -0.1, 1.0).normalized();

  // A sample with 5 of 7 correspondences on the plane is H-degenerate.
  {
    std::vector<Eigen::Vector2d> points1;
    std::vector<Eigen::Vector2d> points2;
    std::vector<char> on_plane_mask;
    GenerateDominantPlaneScene(cam2_from_cam1,
                               K,
                               normal,
                               /*distance=*/2.0,
                               /*num_points=*/7,
                               /*num_on_plane=*/5,
                               /*noise=*/0.0,
                               &points1,
                               &points2,
                               &on_plane_mask);
    EXPECT_TRUE(IsSampleHDegenerate(F,
                                    points1,
                                    points2,
                                    /*h_max_residual=*/4.0,
                                    /*min_sample_h_inlier_ratio=*/5.0 / 7.0));
  }

  // A general (non-planar) sample is not H-degenerate.
  {
    std::vector<Eigen::Vector2d> points1;
    std::vector<Eigen::Vector2d> points2;
    std::vector<char> on_plane_mask;
    GenerateDominantPlaneScene(cam2_from_cam1,
                               K,
                               normal,
                               /*distance=*/2.0,
                               /*num_points=*/7,
                               /*num_on_plane=*/0,
                               /*noise=*/0.0,
                               &points1,
                               &points2,
                               &on_plane_mask);
    EXPECT_FALSE(IsSampleHDegenerate(F,
                                     points1,
                                     points2,
                                     /*h_max_residual=*/4.0,
                                     /*min_sample_h_inlier_ratio=*/5.0 / 7.0));
  }
}

// On a general non-planar scene, DEGENSAC recovers the fundamental matrix as
// well as the plain LO-RANSAC estimator.
TEST(FundamentalMatrixDegensac, NonPlanarParity) {
  SetPRNGSeed(0);
  const Eigen::Matrix3d K = RandomCalibrationMatrix();
  const Rigid3d cam2_from_cam1(RandomEigenQuaterniond(),
                               RandomEigenVectord<3>());
  Eigen::Matrix3d expected_F = FundamentalFromEssentialMatrix(
      K, EssentialMatrixFromPose(cam2_from_cam1), K);
  expected_F /= expected_F(2, 2);

  const Eigen::Vector3d normal = Eigen::Vector3d(0.2, -0.1, 1.0).normalized();
  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  std::vector<char> on_plane_mask;
  GenerateDominantPlaneScene(cam2_from_cam1,
                             K,
                             normal,
                             /*distance=*/2.0,
                             /*num_points=*/200,
                             /*num_on_plane=*/0,
                             /*noise=*/0.0,
                             &points1,
                             &points2,
                             &on_plane_mask);

  FundamentalMatrixDegensacOptions options;
  options.ransac = TestRANSACOptions();
  const auto report =
      EstimateFundamentalMatrixDegensac(points1, points2, options);
  ASSERT_TRUE(report.success);

  Eigen::Matrix3d F = report.model / report.model(2, 2);
  EXPECT_TRUE(F.isApprox(expected_F, 1e-3) || (-F).isApprox(expected_F, 1e-3));
  EXPECT_GE(report.support.num_inliers, points1.size() - 2);
}

// On a dominant-plane scene with real off-plane parallax, DEGENSAC recovers the
// correct fundamental matrix via plane-and-parallax completion, explaining the
// off-plane points well.
TEST(FundamentalMatrixDegensac, RecoversFOnDominantPlane) {
  SetPRNGSeed(0);
  const Eigen::Matrix3d K = RandomCalibrationMatrix();
  const Rigid3d cam2_from_cam1(RandomEigenQuaterniond(),
                               RandomEigenVectord<3>().normalized());
  Eigen::Matrix3d expected_F = FundamentalFromEssentialMatrix(
      K, EssentialMatrixFromPose(cam2_from_cam1), K);
  expected_F /= expected_F(2, 2);

  const Eigen::Vector3d normal = Eigen::Vector3d(0.2, -0.1, 1.0).normalized();
  constexpr size_t kNumPoints = 200;
  constexpr size_t kNumOnPlane = 190;  // 95% dominant plane.
  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  std::vector<char> on_plane_mask;
  GenerateDominantPlaneScene(cam2_from_cam1,
                             K,
                             normal,
                             /*distance=*/2.0,
                             kNumPoints,
                             kNumOnPlane,
                             /*noise=*/0.1,
                             &points1,
                             &points2,
                             &on_plane_mask);

  std::vector<char> off_plane_mask(kNumPoints);
  for (size_t i = 0; i < kNumPoints; ++i) {
    off_plane_mask[i] = !on_plane_mask[i];
  }

  FundamentalMatrixDegensacOptions degensac_options;
  degensac_options.ransac = TestRANSACOptions();
  const auto report =
      EstimateFundamentalMatrixDegensac(points1, points2, degensac_options);
  ASSERT_TRUE(report.success);

  Eigen::Matrix3d F = report.model / report.model(2, 2);
  EXPECT_TRUE(F.isApprox(expected_F, 1e-2) || (-F).isApprox(expected_F, 1e-2));

  // DEGENSAC must explain the off-plane parallax points well, which a
  // plane-corrupted fundamental matrix cannot.
  EXPECT_LT(
      MeanSampsonErrorOnSubset(points1, points2, report.model, off_plane_mask),
      1.0);
}

// Aggregated over many dominant-plane scenes, DEGENSAC recovers the correct
// epipolar geometry at least as often as plain LO-RANSAC, which is prone to
// terminating on a plane-corrupted model.
TEST(FundamentalMatrixDegensac, OutperformsLoRansacOnDominantPlane) {
  constexpr int kNumScenes = 40;
  constexpr size_t kNumPoints = 200;
  constexpr size_t kNumOnPlane = 190;  // 95% dominant plane.
  constexpr double kOffPlaneErrorThreshold = 1.0;

  int degensac_successes = 0;
  int loransac_successes = 0;
  for (int s = 0; s < kNumScenes; ++s) {
    SetPRNGSeed(s);
    const Eigen::Matrix3d K = RandomCalibrationMatrix();
    const Rigid3d cam2_from_cam1(RandomEigenQuaterniond(),
                                 RandomEigenVectord<3>().normalized());
    const Eigen::Vector3d normal = Eigen::Vector3d(0.2, -0.1, 1.0).normalized();
    std::vector<Eigen::Vector2d> points1;
    std::vector<Eigen::Vector2d> points2;
    std::vector<char> on_plane_mask;
    GenerateDominantPlaneScene(cam2_from_cam1,
                               K,
                               normal,
                               /*distance=*/2.0,
                               kNumPoints,
                               kNumOnPlane,
                               /*noise=*/0.1,
                               &points1,
                               &points2,
                               &on_plane_mask);
    std::vector<char> off_plane_mask(kNumPoints);
    for (size_t i = 0; i < kNumPoints; ++i) {
      off_plane_mask[i] = !on_plane_mask[i];
    }

    const RANSACOptions ransac_options = TestRANSACOptions();

    FundamentalMatrixDegensacOptions degensac_options;
    degensac_options.ransac = ransac_options;
    const auto degensac_report =
        EstimateFundamentalMatrixDegensac(points1, points2, degensac_options);
    if (degensac_report.success &&
        MeanSampsonErrorOnSubset(
            points1, points2, degensac_report.model, off_plane_mask) <
            kOffPlaneErrorThreshold) {
      ++degensac_successes;
    }

    LORANSAC<FundamentalMatrixSevenPointEstimator,
             FundamentalMatrixEightPointEstimator>
        loransac(ransac_options);
    const auto loransac_report = loransac.Estimate(points1, points2);
    if (loransac_report.success &&
        MeanSampsonErrorOnSubset(
            points1, points2, loransac_report.model, off_plane_mask) <
            kOffPlaneErrorThreshold) {
      ++loransac_successes;
    }
  }

  EXPECT_GT(degensac_successes, 0);
  EXPECT_GE(degensac_successes, loransac_successes);
}

}  // namespace
}  // namespace colmap

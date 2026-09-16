// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/solvers/affine_transform.h"

#include "colmap/math/random.h"
#include "colmap/math/random_eigen.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/eigen_matchers.h"

#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>>
GenerateData(size_t num_inliers,
             size_t num_outliers,
             const Eigen::Matrix2x3d& tgt_from_src) {
  std::vector<Eigen::Vector2d> src;
  std::vector<Eigen::Vector2d> tgt;

  // Generate inlier data.
  for (size_t i = 0; i < num_inliers; ++i) {
    src.emplace_back(i, std::sqrt(i) + 2);
    tgt.push_back(tgt_from_src * src.back().homogeneous());
  }

  // Add some faulty data.
  for (size_t i = 0; i < num_outliers; ++i) {
    src.emplace_back(i, std::sqrt(i) + 2);
    tgt.emplace_back(RandomUniformReal(-3000.0, -2000.0),
                     RandomUniformReal(-4000.0, -3000.0));
  }

  return {std::move(src), std::move(tgt)};
}

void TestEstimateAffine2dWithNumCoords(const size_t num_coords) {
  const Eigen::Matrix2x3d gt_tgt_from_src = RandomEigenMatrixd<2, 3>();
  const auto [src, tgt] = GenerateData(
      /*num_inliers=*/num_coords,
      /*num_outliers=*/0,
      gt_tgt_from_src);

  Eigen::Matrix2x3d tgt_from_src;
  EXPECT_TRUE(EstimateAffine2d(src, tgt, tgt_from_src));
  EXPECT_THAT(tgt_from_src, EigenMatrixNear(gt_tgt_from_src, 1e-6));
}

TEST(Affine2d, EstimateMinimal) { TestEstimateAffine2dWithNumCoords(3); }

TEST(Affine2d, EstimateOverDetermined) {
  TestEstimateAffine2dWithNumCoords(100);
}

TEST(Affine2d, EstimateMinimalDegenerate) {
  std::vector<Eigen::Vector2d> degenerate_src_tgt(3, Eigen::Vector2d::Zero());
  Eigen::Matrix2x3d tgt_from_src;
  EXPECT_FALSE(
      EstimateAffine2d(degenerate_src_tgt, degenerate_src_tgt, tgt_from_src));
}

TEST(Affine2d, EstimateNonMinimalDegenerate) {
  std::vector<Eigen::Vector2d> degenerate_src_tgt(5, Eigen::Vector2d::Zero());
  Eigen::Matrix2x3d tgt_from_src;
  EXPECT_FALSE(
      EstimateAffine2d(degenerate_src_tgt, degenerate_src_tgt, tgt_from_src));
}

TEST(Affine2d, EstimateRobust) {
  const size_t num_inliers = 1000;
  const size_t num_outliers = 400;

  const Eigen::Matrix2x3d gt_tgt_from_src = RandomEigenMatrixd<2, 3>();
  const auto [src, tgt] = GenerateData(
      /*num_inliers=*/num_inliers,
      /*num_outliers=*/num_outliers,
      gt_tgt_from_src);

  // Robustly estimate transformation using RANSAC.
  RANSACOptions options;
  options.max_error = 10;
  Eigen::Matrix2x3d tgt_from_src;
  const auto report = EstimateAffine2dRobust(src, tgt, options, tgt_from_src);

  EXPECT_TRUE(report.success);
  EXPECT_GT(report.num_trials, 0);

  // Make sure outliers were detected correctly.
  EXPECT_EQ(report.support.num_inliers, num_inliers);
  for (size_t i = 0; i < num_inliers + num_outliers; ++i) {
    if (i >= num_inliers) {
      EXPECT_FALSE(report.inlier_mask[i]);
    } else {
      EXPECT_TRUE(report.inlier_mask[i]);
    }
  }

  EXPECT_THAT(tgt_from_src, EigenMatrixNear(gt_tgt_from_src, 1e-6));
}

}  // namespace
}  // namespace colmap

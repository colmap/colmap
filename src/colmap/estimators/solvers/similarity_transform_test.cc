// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/solvers/similarity_transform.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/geometry/sim3.h"
#include "colmap/geometry/sim3_matchers.h"
#include "colmap/math/random.h"
#include "colmap/math/random_eigen.h"
#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d> >
GenerateData(size_t num_inliers,
             size_t num_outliers,
             const Eigen::Matrix3x4d& tgt_from_src) {
  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> tgt;

  // Generate inlier data.
  for (size_t i = 0; i < num_inliers; ++i) {
    src.emplace_back(i, std::sqrt(i) + 2, std::sqrt(2 * i + 2));
    tgt.push_back(tgt_from_src * src.back().homogeneous());
  }

  // Add some faulty data.
  for (size_t i = 0; i < num_outliers; ++i) {
    src.emplace_back(i, std::sqrt(i) + 2, std::sqrt(2 * i + 2));
    tgt.emplace_back(RandomUniformReal(-3000.0, -2000.0),
                     RandomUniformReal(-4000.0, -3000.0),
                     RandomUniformReal(-5000.0, -4000.0));
  }

  return {std::move(src), std::move(tgt)};
}

void TestEstimateRigid3dWithNumCoords(const size_t num_coords) {
  const Rigid3d gt_tgt_from_src(RandomEigenQuaterniond(),
                                RandomEigenVectord<3>());
  const auto [src, tgt] = GenerateData(
      /*num_inliers=*/num_coords,
      /*num_outliers=*/0,
      gt_tgt_from_src.ToMatrix());

  Rigid3d tgt_from_src;
  EXPECT_TRUE(EstimateRigid3d(src, tgt, tgt_from_src));
  EXPECT_LT(gt_tgt_from_src.rotation().angularDistance(tgt_from_src.rotation()),
            1e-6);
  EXPECT_LT((gt_tgt_from_src.translation() - tgt_from_src.translation()).norm(),
            1e-6);
}

TEST(Rigid3d, EstimateMinimal) { TestEstimateRigid3dWithNumCoords(3); }

TEST(Rigid3d, EstimateOverDetermined) { TestEstimateRigid3dWithNumCoords(100); }

TEST(Rigid3d, EstimateMinimalDegenerate) {
  std::vector<Eigen::Vector3d> degenerate_src_tgt(3, Eigen::Vector3d::Zero());
  Rigid3d tgt_from_src;
  EXPECT_FALSE(
      EstimateRigid3d(degenerate_src_tgt, degenerate_src_tgt, tgt_from_src));
}

TEST(Rigid3d, EstimateNonMinimalDegenerate) {
  std::vector<Eigen::Vector3d> degenerate_src_tgt(5, Eigen::Vector3d::Zero());
  Rigid3d tgt_from_src;
  EXPECT_FALSE(
      EstimateRigid3d(degenerate_src_tgt, degenerate_src_tgt, tgt_from_src));
}

TEST(Rigid3d, EstimateRobust) {
  const size_t num_inliers = 1000;
  const size_t num_outliers = 400;

  const Rigid3d gt_tgt_from_src(RandomEigenQuaterniond(),
                                Eigen::Vector3d(100, 10, 10));
  const auto [src, tgt] = GenerateData(
      /*num_inliers=*/num_inliers,
      /*num_outliers=*/num_outliers,
      gt_tgt_from_src.ToMatrix());

  // Robustly estimate transformation using RANSAC.
  RANSACOptions options;
  options.max_error = 10;
  Rigid3d tgt_from_src;
  const auto report = EstimateRigid3dRobust(src, tgt, options, tgt_from_src);

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

  EXPECT_THAT(tgt_from_src,
              Rigid3dNear(gt_tgt_from_src, /*rtol=*/1e-6, /*ttol=*/1e-6));
}

void TestEstimateSim3dWithNumCoords(const size_t num_coords) {
  const Sim3d gt_tgt_from_src(RandomUniformReal<double>(0.1, 10),
                              RandomEigenQuaterniond(),
                              RandomEigenVectord<3>());
  const auto [src, tgt] = GenerateData(
      /*num_inliers=*/num_coords,
      /*num_outliers=*/0,
      gt_tgt_from_src.ToMatrix());

  Sim3d tgt_from_src;
  EXPECT_TRUE(EstimateSim3d(src, tgt, tgt_from_src));
  EXPECT_NEAR(gt_tgt_from_src.scale(), tgt_from_src.scale(), 1e-6);
  EXPECT_LT(gt_tgt_from_src.rotation().angularDistance(tgt_from_src.rotation()),
            1e-6);
  EXPECT_LT((gt_tgt_from_src.translation() - tgt_from_src.translation()).norm(),
            1e-6);
}

TEST(Sim3d, EstimateMinimal) { TestEstimateSim3dWithNumCoords(3); }

TEST(Sim3d, EstimateOverDetermined) { TestEstimateSim3dWithNumCoords(100); }

TEST(Sim3d, EstimateMinimalDegenerate) {
  std::vector<Eigen::Vector3d> degenerate_src_tgt(3, Eigen::Vector3d::Zero());
  Sim3d tgt_from_src;
  EXPECT_FALSE(
      EstimateSim3d(degenerate_src_tgt, degenerate_src_tgt, tgt_from_src));
}

TEST(Sim3d, EstimateNonMinimalDegenerate) {
  std::vector<Eigen::Vector3d> degenerate_src_tgt(5, Eigen::Vector3d::Zero());
  Sim3d tgt_from_src;
  EXPECT_FALSE(
      EstimateSim3d(degenerate_src_tgt, degenerate_src_tgt, tgt_from_src));
}

template <int kDim, bool kEstimateScale>
void TestAffineRank() {
  using Estimator = SimilarityTransformEstimator<kDim, kEstimateScale>;
  using Point = typename Estimator::X_t;
  for (const double offset : {0.0, 10.0}) {
    std::vector<Point> src(kDim, Point::Constant(offset));
    for (int i = 1; i < kDim; ++i) {
      src[i](i - 1) += 1;
    }
    const double scale = kEstimateScale ? 2.0 : 1.0;
    std::vector<Point> tgt;
    tgt.reserve(src.size());
    for (const auto& point : src) {
      tgt.push_back(scale * point + Point::Constant(3));
    }
    for (int i = 0; i < 2; ++i) {
      std::vector<typename Estimator::M_t> models;
      Estimator::Estimate(src, tgt, &models);
      ASSERT_EQ(models.size(), 1);
      for (size_t j = 0; j < src.size(); ++j) {
        EXPECT_LT((models[0] * src[j].homogeneous() - tgt[j]).norm(), 1e-10);
      }
      // Test a plane through the origin in either the source or target frame.
      std::swap(src, tgt);
    }
  }
}

TEST(SimilarityTransformEstimator, AffineRank) {
  TestAffineRank<2, false>();
  TestAffineRank<2, true>();
  TestAffineRank<3, false>();
  TestAffineRank<3, true>();
}

TEST(SimilarityTransformEstimator, Collinear3d) {
  // An offset line has linear rank two but still cannot determine a rotation.
  const std::vector<Eigen::Vector3d> points = {{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  std::vector<Eigen::Matrix3x4d> models;
  SimilarityTransformEstimator<3, false>::Estimate(points, points, &models);
  EXPECT_TRUE(models.empty());
  SimilarityTransformEstimator<3, true>::Estimate(points, points, &models);
  EXPECT_TRUE(models.empty());
}

TEST(Sim3d, EstimatePlanarRobust) {
  const Sim3d expected(
      2,
      Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX())),
      Eigen::Vector3d(3, 4, 5));
  const std::vector<Eigen::Vector3d> src = {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
  std::vector<Eigen::Vector3d> tgt;
  tgt.reserve(src.size());
  for (const auto& point : src) {
    tgt.push_back(expected * point);
  }
  RANSACOptions options;
  options.max_error = 1e-6;
  options.max_num_trials = 20;
  options.min_num_trials = 1;
  Sim3d estimated;
  const auto report = EstimateSim3dRobust(src, tgt, options, estimated);
  ASSERT_TRUE(report.success);
  EXPECT_EQ(report.support.num_inliers, src.size());
  EXPECT_THAT(estimated, Sim3dNear(expected, 1e-10, 1e-10, 1e-10));
}

TEST(Sim3d, EstimateRobust) {
  const size_t num_inliers = 1000;
  const size_t num_outliers = 400;

  const Sim3d gt_tgt_from_src(
      2, RandomEigenQuaterniond(), Eigen::Vector3d(100, 10, 10));
  const auto [src, tgt] = GenerateData(
      /*num_inliers=*/num_inliers,
      /*num_outliers=*/num_outliers,
      gt_tgt_from_src.ToMatrix());

  // Robustly estimate transformation using RANSAC.
  RANSACOptions options;
  options.max_error = 10;
  Sim3d tgt_from_src;
  const auto report = EstimateSim3dRobust(src, tgt, options, tgt_from_src);

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

  EXPECT_THAT(
      tgt_from_src,
      Sim3dNear(gt_tgt_from_src, /*stol=*/1e-8, /*rtol=*/1e-8, /*ttol=*/1e-8));
}

}  // namespace
}  // namespace colmap

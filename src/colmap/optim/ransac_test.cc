// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/ransac.h"

#include "colmap/estimators/solvers/similarity_transform.h"
#include "colmap/geometry/sim3.h"
#include "colmap/math/random.h"
#include "colmap/math/random_eigen.h"
#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

struct SimilarityTransformTestData {
  Sim3d expected_tgt_from_src;
  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> tgt;
  size_t num_samples;
  size_t num_outliers;
};

SimilarityTransformTestData GenerateTestData(const size_t num_samples = 1000,
                                             const size_t num_outliers = 400) {
  SimilarityTransformTestData data;
  data.num_samples = num_samples;
  data.num_outliers = num_outliers;

  data.expected_tgt_from_src =
      Sim3d(2, RandomEigenQuaterniond(), Eigen::Vector3d(100, 10, 10));

  for (size_t i = 0; i < num_samples; ++i) {
    data.src.emplace_back(i, std::sqrt(i) + 2, std::sqrt(2 * i + 2));
    data.tgt.push_back(data.expected_tgt_from_src * data.src.back());
  }

  for (size_t i = 0; i < num_outliers; ++i) {
    data.tgt[i] = Eigen::Vector3d(RandomUniformReal(-3000.0, -2000.0),
                                  RandomUniformReal(-4000.0, -3000.0),
                                  RandomUniformReal(-5000.0, -4000.0));
  }

  return data;
}

template <typename Report>
void ValidateReport(const Report& report,
                    const SimilarityTransformTestData& data) {
  EXPECT_TRUE(report.success);
  EXPECT_GT(report.num_trials, 0);

  EXPECT_EQ(report.support.num_inliers, data.num_samples - data.num_outliers);
  for (size_t i = 0; i < data.num_samples; ++i) {
    if (i < data.num_outliers) {
      EXPECT_FALSE(report.inlier_mask[i]);
    } else {
      EXPECT_TRUE(report.inlier_mask[i]);
    }
  }

  const double matrix_diff =
      (data.expected_tgt_from_src.ToMatrix() - report.model).norm();
  EXPECT_LT(matrix_diff, 1e-6);
}

TEST(RANSAC, Options) {
  RANSACOptions options;
  EXPECT_EQ(options.max_error, 0);
  EXPECT_EQ(options.min_inlier_ratio, 0.1);
  EXPECT_EQ(options.confidence, 0.99);
  EXPECT_EQ(options.min_num_trials, 0);
  EXPECT_EQ(options.max_num_trials, std::numeric_limits<int>::max());
}

TEST(RANSAC, Report) {
  RANSAC<SimilarityTransformEstimator<3>>::Report report;
  EXPECT_FALSE(report.success);
  EXPECT_EQ(report.num_trials, 0);
  EXPECT_EQ(report.support.num_inliers, 0);
  EXPECT_EQ(report.support.residual_sum, std::numeric_limits<double>::max());
  EXPECT_EQ(report.inlier_mask.size(), 0);
}

TEST(RANSAC, NumTrials) {
  EXPECT_EQ(RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(
                1, 100, 0.99, 1.0),
            18446744073709551615llu);
  EXPECT_EQ(RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(
                10, 100, 0.99, 1.0),
            6204);
  EXPECT_EQ(RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(
                10, 100, 0.999, 1.0),
            9305);
  EXPECT_EQ(RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(
                10, 100, 0.999, 2.0),
            18610);
  EXPECT_EQ(RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(
                50, 100, 0.99, 1.0),
            36);
  EXPECT_EQ(RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(
                50, 100, 0.999, 1.0),
            54);
  EXPECT_EQ(RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(
                100, 100, 0.99, 1.0),
            1);
  EXPECT_EQ(RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(
                100, 100, 0.999, 1.0),
            1);
  EXPECT_EQ(RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(
                100, 100, 0, 1.0),
            1);
}

TEST(RANSAC, SimilarityTransform) {
  const auto data = GenerateTestData();

  RANSACOptions options;
  options.max_error = 10;
  options.random_seed = kDefaultPRNGSeed;
  RANSAC<SimilarityTransformEstimator<3>> ransac(options);
  const auto report = ransac.Estimate(data.src, data.tgt);

  ValidateReport(report, data);
}

TEST(RANSAC, ParallelSimilarityTransform) {
  const auto data = GenerateTestData();

  RANSACOptions options;
  options.max_error = 10;
  options.random_seed = kDefaultPRNGSeed;
  options.num_threads = 4;
  RANSAC<SimilarityTransformEstimator<3>> ransac(options);
  const auto report = ransac.Estimate(data.src, data.tgt);

  ValidateReport(report, data);
}

TEST(RANSAC, ReproducibilityWithRandomSeed) {
  const auto data = GenerateTestData();

  RANSACOptions options1;
  options1.max_error = 10;
  options1.random_seed = 42;
  RANSAC<SimilarityTransformEstimator<3>> ransac1(options1);
  const auto report1 = ransac1.Estimate(data.src, data.tgt);

  RANSACOptions options2 = options1;
  RANSAC<SimilarityTransformEstimator<3>> ransac2(options2);
  const auto report2 = ransac2.Estimate(data.src, data.tgt);

  ASSERT_TRUE(report1.success);
  ASSERT_TRUE(report2.success);

  // Results should be exactly the same.
  EXPECT_EQ(report1.support.num_inliers, report2.support.num_inliers);
  EXPECT_EQ(report1.inlier_mask, report2.inlier_mask);
  EXPECT_EQ(report1.model, report2.model);

  // Now change the seed.
  options2.random_seed = 123;
  RANSAC<SimilarityTransformEstimator<3>> ransac3(options2);
  const auto report3 = ransac3.Estimate(data.src, data.tgt);

  ASSERT_TRUE(report3.success);

  // Results should now differ.
  EXPECT_NE(report1.model, report3.model);
}

}  // namespace
}  // namespace colmap

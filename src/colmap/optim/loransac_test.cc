// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/loransac.h"

#include "colmap/estimators/solvers/similarity_transform.h"
#include "colmap/geometry/sim3.h"
#include "colmap/math/random.h"
#include "colmap/math/random_eigen.h"
#include "colmap/util/eigen_alignment.h"

#include <algorithm>

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

class ShrinkingInlierEstimator {
 public:
  using X_t = double;
  using Y_t = double;
  using M_t = int;

  static const int kMinNumSamples = 5;

  static void Estimate(const std::vector<X_t>& X,
                       const std::vector<Y_t>& Y,
                       std::vector<M_t>* models) {
    THROW_CHECK_EQ(X.size(), Y.size());
    THROW_CHECK_GE(X.size(), kMinNumSamples);
    models->assign(1, 0);
  }

  static bool Refine(const std::vector<X_t>& X,
                     const std::vector<Y_t>& Y,
                     M_t* model) {
    THROW_CHECK_EQ(X.size(), Y.size());
    THROW_CHECK_GE(X.size(), kMinNumSamples);
    *model = 1;
    return true;
  }

  static void Residuals(const std::vector<X_t>& X,
                        const std::vector<Y_t>& Y,
                        const M_t model,
                        std::vector<double>* residuals) {
    THROW_CHECK_EQ(X.size(), Y.size());
    residuals->assign(X.size(), 0.9);
    if (model == 1) {
      residuals->assign(X.size(), 2.0);
      std::fill_n(residuals->begin(), 4, 0.0);
    }
  }
};

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

TEST(LORANSAC, Report) {
  LORANSAC<SimilarityTransformEstimator<3>,
           SimilarityTransformEstimator<3>>::Report report;
  EXPECT_FALSE(report.success);
  EXPECT_EQ(report.num_trials, 0);
  EXPECT_EQ(report.support.num_inliers, 0);
  EXPECT_EQ(report.support.residual_sum, std::numeric_limits<double>::max());
  EXPECT_EQ(report.inlier_mask.size(), 0);
}

TEST(LORANSAC, SimilarityTransform) {
  const auto data = GenerateTestData();

  RANSACOptions options;
  options.max_error = 10;
  options.random_seed = kDefaultPRNGSeed;
  LORANSAC<SimilarityTransformEstimator<3>, SimilarityTransformEstimator<3>>
      loransac(options);
  const auto report = loransac.Estimate(data.src, data.tgt);

  ValidateReport(report, data);
}

TEST(LORANSAC, ParallelSimilarityTransform) {
  const auto data = GenerateTestData();

  RANSACOptions options;
  options.max_error = 10;
  options.random_seed = kDefaultPRNGSeed;
  options.num_threads = 4;
  LORANSAC<SimilarityTransformEstimator<3>, SimilarityTransformEstimator<3>>
      loransac(options);
  const auto report = loransac.Estimate(data.src, data.tgt);

  ValidateReport(report, data);
}

TEST(LORANSAC, RecursiveRefinementStopsBelowMinimumSampleCount) {
  RANSACOptions options;
  options.max_error = 1.0;
  options.min_num_trials = 1;
  options.max_num_trials = 1;
  options.random_seed = kDefaultPRNGSeed;

  LORANSAC<ShrinkingInlierEstimator,
           ShrinkingInlierEstimator,
           MEstimatorSupportMeasurer>
      loransac(options);
  const std::vector<double> samples(6, 0.0);

  LORANSAC<ShrinkingInlierEstimator,
           ShrinkingInlierEstimator,
           MEstimatorSupportMeasurer>::Report report;
  EXPECT_NO_THROW(report = loransac.Estimate(samples, samples));
  EXPECT_FALSE(report.success);
  EXPECT_EQ(report.support.num_inliers, 4);
}

}  // namespace
}  // namespace colmap

// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/sprt.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(SPRT, EvaluateAllInliers) {
  SPRT::Options options;
  options.delta = 0.05;
  options.epsilon = 0.5;
  SPRT sprt(options);

  // All residuals are small (inliers)
  std::vector<double> residuals(100, 0.1);
  size_t num_inliers = 0;
  size_t num_eval_samples = 0;
  const bool accepted =
      sprt.Evaluate(residuals, 1.0, &num_inliers, &num_eval_samples);

  EXPECT_TRUE(accepted);
  EXPECT_EQ(num_inliers, 100);
  EXPECT_EQ(num_eval_samples, 100);
}

TEST(SPRT, EvaluateAllOutliers) {
  SPRT::Options options;
  options.delta = 0.05;
  options.epsilon = 0.5;
  SPRT sprt(options);

  // All residuals are large (outliers) - should trigger early rejection
  std::vector<double> residuals(100, 10.0);
  size_t num_inliers = 0;
  size_t num_eval_samples = 0;
  const bool accepted =
      sprt.Evaluate(residuals, 1.0, &num_inliers, &num_eval_samples);

  EXPECT_FALSE(accepted);
  EXPECT_EQ(num_inliers, 0);
  EXPECT_LT(num_eval_samples, 100);
}

TEST(SPRT, EvaluateMixedEarlyReject) {
  SPRT::Options options;
  options.delta = 0.05;
  options.epsilon = 0.9;
  SPRT sprt(options);

  // Mostly outliers - should reject early
  std::vector<double> residuals(1000, 10.0);
  // Sprinkle a few inliers
  residuals[0] = 0.1;
  residuals[10] = 0.1;

  size_t num_inliers = 0;
  size_t num_eval_samples = 0;
  const bool accepted =
      sprt.Evaluate(residuals, 1.0, &num_inliers, &num_eval_samples);

  EXPECT_FALSE(accepted);
  // With epsilon=0.9 and delta=0.05, the likelihood ratio exceeds the decision
  // threshold after processing the inlier at index 0 and 4 subsequent outliers.
  EXPECT_EQ(num_inliers, 1);
  EXPECT_EQ(num_eval_samples, 5);
}

TEST(SPRT, EvaluateEmpty) {
  SPRT::Options options;
  SPRT sprt(options);

  std::vector<double> residuals;
  size_t num_inliers = 0;
  size_t num_eval_samples = 0;
  const bool accepted =
      sprt.Evaluate(residuals, 1.0, &num_inliers, &num_eval_samples);

  EXPECT_TRUE(accepted);
  EXPECT_EQ(num_inliers, 0);
  EXPECT_EQ(num_eval_samples, 0);
}

}  // namespace
}  // namespace colmap

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

// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/optim/support_measurement.h"

#include "colmap/math/math.h"

#include <unordered_set>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(InlierSupportMeasurer, Nominal) {
  InlierSupportMeasurer::Support support1;
  EXPECT_EQ(support1.num_inliers, 0);
  EXPECT_EQ(support1.residual_sum, std::numeric_limits<double>::max());
  InlierSupportMeasurer measurer;
  std::vector<double> residuals = {-1.0, 0.0, 1.0, 2.0};
  support1 = measurer.Evaluate(residuals, 1.0);
  EXPECT_EQ(support1.num_inliers, 3);
  EXPECT_EQ(support1.residual_sum, 0.0);
  InlierSupportMeasurer::Support support2;
  support2.num_inliers = 2;
  EXPECT_TRUE(measurer.Compare(support1, support2));
  EXPECT_FALSE(measurer.Compare(support2, support1));
  support2.residual_sum = support1.residual_sum;
  EXPECT_TRUE(measurer.Compare(support1, support2));
  EXPECT_FALSE(measurer.Compare(support2, support1));
  support2.num_inliers = support1.num_inliers;
  support2.residual_sum += 0.01;
  EXPECT_TRUE(measurer.Compare(support1, support2));
  EXPECT_FALSE(measurer.Compare(support2, support1));
  support2.residual_sum -= 0.01;
  EXPECT_FALSE(measurer.Compare(support1, support2));
  EXPECT_FALSE(measurer.Compare(support2, support1));
  support2.residual_sum -= 0.01;
  EXPECT_FALSE(measurer.Compare(support1, support2));
  EXPECT_TRUE(measurer.Compare(support2, support1));
}

TEST(UniqueInlierSupportMeasurer, Nominal) {
  UniqueInlierSupportMeasurer::Support support1;
  EXPECT_EQ(support1.num_inliers, 0);
  EXPECT_EQ(support1.num_unique_inliers, 0);
  EXPECT_EQ(support1.residual_sum, std::numeric_limits<double>::max());

  UniqueInlierSupportMeasurer measurer;
  const std::vector<size_t> sample_ids = {1, 2, 2, 3};
  measurer.SetUniqueSampleIds(sample_ids);
  const std::vector<double> residuals = {-1.0, 0.0, 1.0, 2.0};
  support1 = measurer.Evaluate(residuals, 1.0);
  EXPECT_EQ(support1.num_inliers, 3);
  EXPECT_EQ(support1.num_unique_inliers, 2);
  EXPECT_EQ(support1.residual_sum, 0.0);

  UniqueInlierSupportMeasurer::Support support2;
  support2.num_unique_inliers = support1.num_unique_inliers - 1;
  EXPECT_TRUE(measurer.Compare(support1, support2));
  EXPECT_FALSE(measurer.Compare(support2, support1));
  support2.num_inliers = support1.num_inliers + 1;
  EXPECT_TRUE(measurer.Compare(support1, support2));
  EXPECT_FALSE(measurer.Compare(support2, support1));
  support2.num_inliers = support1.num_inliers;
  EXPECT_TRUE(measurer.Compare(support1, support2));
  EXPECT_FALSE(measurer.Compare(support2, support1));
  support2.residual_sum = support1.residual_sum - 0.01;
  EXPECT_TRUE(measurer.Compare(support1, support2));
  EXPECT_FALSE(measurer.Compare(support2, support1));
  support2.residual_sum = support1.residual_sum;
  EXPECT_TRUE(measurer.Compare(support1, support2));
  EXPECT_FALSE(measurer.Compare(support2, support1));
  support2.num_unique_inliers = support1.num_unique_inliers;
  EXPECT_FALSE(measurer.Compare(support1, support2));
  EXPECT_FALSE(measurer.Compare(support2, support1));
  support2.residual_sum = support1.residual_sum - 0.01;
  EXPECT_FALSE(measurer.Compare(support1, support2));
  EXPECT_TRUE(measurer.Compare(support2, support1));
  support2.num_inliers = support1.num_inliers + 1;
  support2.residual_sum = support1.residual_sum + 0.01;
  EXPECT_FALSE(measurer.Compare(support1, support2));
  EXPECT_TRUE(measurer.Compare(support2, support1));
  support2.num_unique_inliers = support1.num_unique_inliers + 1;
  support2.num_inliers = support1.num_inliers - 1;
  support2.residual_sum = support1.residual_sum + 0.01;
  EXPECT_FALSE(measurer.Compare(support1, support2));
  EXPECT_TRUE(measurer.Compare(support2, support1));
}

TEST(MEstimatorSupportMeasurer, Nominal) {
  MEstimatorSupportMeasurer::Support support1;
  EXPECT_EQ(support1.num_inliers, 0);
  EXPECT_EQ(support1.score, std::numeric_limits<double>::max());
  MEstimatorSupportMeasurer measurer;
  std::vector<double> residuals = {-1.0, 0.0, 1.0, 2.0};
  support1 = measurer.Evaluate(residuals, 1.0);
  EXPECT_EQ(support1.num_inliers, 3);
  EXPECT_EQ(support1.score, 1.0);
  MEstimatorSupportMeasurer::Support support2 = support1;
  EXPECT_FALSE(measurer.Compare(support1, support2));
  EXPECT_FALSE(measurer.Compare(support2, support1));
  support2.num_inliers -= 1;
  support2.score += 0.01;
  EXPECT_TRUE(measurer.Compare(support1, support2));
  EXPECT_FALSE(measurer.Compare(support2, support1));
  support2.score -= 0.01;
  EXPECT_FALSE(measurer.Compare(support1, support2));
  EXPECT_FALSE(measurer.Compare(support2, support1));
  support2.score -= 0.01;
  EXPECT_FALSE(measurer.Compare(support1, support2));
  EXPECT_TRUE(measurer.Compare(support2, support1));
}

}  // namespace
}  // namespace colmap

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

#include "colmap/optim/loransac.h"

#include "colmap/estimators/solvers/similarity_transform.h"
#include "colmap/geometry/sim3.h"
#include "colmap/math/random.h"
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

  SetPRNGSeed(0);
  data.expected_tgt_from_src =
      Sim3d(2, Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d(100, 10, 10));

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

}  // namespace
}  // namespace colmap

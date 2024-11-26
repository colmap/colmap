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

#include "colmap/estimators/similarity_transform.h"

#include "colmap/geometry/sim3.h"
#include "colmap/math/random.h"
#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

void TestEstimateSim3dWithNumCoords(const size_t num_coords) {
  const Sim3d gt_tgt_from_src(RandomUniformReal<double>(0.1, 10),
                              Eigen::Quaterniond::UnitRandom(),
                              Eigen::Vector3d::Random());

  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> dst;
  for (size_t i = 0; i < num_coords; ++i) {
    src.emplace_back(i, i + 2, i * i);
    dst.push_back(gt_tgt_from_src * src.back());
  }

  Sim3d tgt_from_src;
  EXPECT_TRUE(EstimateSim3d(src, dst, tgt_from_src));
  EXPECT_NEAR(gt_tgt_from_src.scale, tgt_from_src.scale, 1e-6);
  EXPECT_LT(gt_tgt_from_src.rotation.angularDistance(tgt_from_src.rotation),
            1e-6);
  EXPECT_LT((gt_tgt_from_src.translation - tgt_from_src.translation).norm(),
            1e-6);
}

TEST(Sim3d, EstimateMinimal) { TestEstimateSim3dWithNumCoords(3); }

TEST(Sim3d, EstimateOverDetermined) { TestEstimateSim3dWithNumCoords(100); }

TEST(Sim3d, EstimateDegenerate) {
  std::vector<Eigen::Vector3d> invalid_src_dst(3, Eigen::Vector3d::Zero());
  Sim3d tgt_from_src;
  EXPECT_FALSE(EstimateSim3d(invalid_src_dst, invalid_src_dst, tgt_from_src));
}

TEST(Sim3d, EstimateRobust) {
  SetPRNGSeed(0);

  const size_t num_samples = 1000;
  const size_t num_outliers = 400;

  // Create some arbitrary transformation.
  const Sim3d expectedTgtFromSrc(
      2, Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d(100, 10, 10));

  // Generate exact data
  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> tgt;
  for (size_t i = 0; i < num_samples; ++i) {
    src.emplace_back(i, std::sqrt(i) + 2, std::sqrt(2 * i + 2));
    tgt.push_back(expectedTgtFromSrc * src.back());
  }

  // Add some faulty data.
  for (size_t i = 0; i < num_outliers; ++i) {
    tgt[i] = Eigen::Vector3d(RandomUniformReal(-3000.0, -2000.0),
                             RandomUniformReal(-4000.0, -3000.0),
                             RandomUniformReal(-5000.0, -4000.0));
  }

  // Robustly estimate transformation using RANSAC.
  RANSACOptions options;
  options.max_error = 10;
  Sim3d tgt_from_src;
  const auto report = EstimateSim3dRobust(src, tgt, options, tgt_from_src);

  EXPECT_TRUE(report.success);
  EXPECT_GT(report.num_trials, 0);

  // Make sure outliers were detected correctly.
  EXPECT_EQ(report.support.num_inliers, num_samples - num_outliers);
  for (size_t i = 0; i < num_samples; ++i) {
    if (i < num_outliers) {
      EXPECT_FALSE(report.inlier_mask[i]);
    } else {
      EXPECT_TRUE(report.inlier_mask[i]);
    }
  }

  // Make sure original transformation is estimated correctly.
  const double matrix_diff =
      (expectedTgtFromSrc.ToMatrix() - tgt_from_src.ToMatrix()).norm();
  EXPECT_LT(matrix_diff, 1e-6);
}

}  // namespace
}  // namespace colmap

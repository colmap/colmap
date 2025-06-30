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

#include "colmap/estimators/similarity_transform.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/geometry/sim3.h"
#include "colmap/geometry/sim3_matchers.h"
#include "colmap/math/random.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/eigen_matchers.h"

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
  const Rigid3d gt_tgt_from_src(Eigen::Quaterniond::UnitRandom(),
                                Eigen::Vector3d::Random());
  const auto [src, tgt] = GenerateData(
      /*num_inliers=*/num_coords,
      /*num_outliers=*/0,
      gt_tgt_from_src.ToMatrix());

  Rigid3d tgt_from_src;
  EXPECT_TRUE(EstimateRigid3d(src, tgt, tgt_from_src));
  EXPECT_LT(gt_tgt_from_src.rotation.angularDistance(tgt_from_src.rotation),
            1e-6);
  EXPECT_LT((gt_tgt_from_src.translation - tgt_from_src.translation).norm(),
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
  SetPRNGSeed(0);

  const size_t num_inliers = 1000;
  const size_t num_outliers = 400;

  const Rigid3d gt_tgt_from_src(Eigen::Quaterniond::UnitRandom(),
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
                              Eigen::Quaterniond::UnitRandom(),
                              Eigen::Vector3d::Random());
  const auto [src, tgt] = GenerateData(
      /*num_inliers=*/num_coords,
      /*num_outliers=*/0,
      gt_tgt_from_src.ToMatrix());

  Sim3d tgt_from_src;
  EXPECT_TRUE(EstimateSim3d(src, tgt, tgt_from_src));
  EXPECT_NEAR(gt_tgt_from_src.scale, tgt_from_src.scale, 1e-6);
  EXPECT_LT(gt_tgt_from_src.rotation.angularDistance(tgt_from_src.rotation),
            1e-6);
  EXPECT_LT((gt_tgt_from_src.translation - tgt_from_src.translation).norm(),
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

TEST(Sim3d, EstimateRobust) {
  SetPRNGSeed(0);

  const size_t num_inliers = 1000;
  const size_t num_outliers = 400;

  const Sim3d gt_tgt_from_src(
      2, Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d(100, 10, 10));
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

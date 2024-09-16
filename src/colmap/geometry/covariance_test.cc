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

#include "colmap/geometry/covariance.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(IsPointWithinUncertaintyInterval, Degenerate) {
  EXPECT_FALSE(IsPointWithinUncertaintyInterval(Eigen::Vector2d::Zero(),
                                                Eigen::Matrix2d::Zero(),
                                                Eigen::Vector2d(0, 0),
                                                1));
  EXPECT_FALSE(IsPointWithinUncertaintyInterval(
      Eigen::Vector2d::Zero(),
      (Eigen::Matrix2d() << 1, 0, 0, 0).finished(),
      Eigen::Vector2d(0, 0),
      1));
  EXPECT_FALSE(IsPointWithinUncertaintyInterval(
      Eigen::Vector2d::Zero(),
      (Eigen::Matrix2d() << 0, 0, 0, 1).finished(),
      Eigen::Vector2d(0, 0),
      1));
}

bool AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
    const Eigen::Matrix2d& cov, const Eigen::Vector2d& x, double sigma_factor) {
  return IsPointWithinUncertaintyInterval(
             Eigen::Vector2d::Zero(), cov, x, sigma_factor) &&
         IsPointWithinUncertaintyInterval(
             Eigen::Vector2d::Zero(), cov, -x, sigma_factor);
}

TEST(IsPointWithinUncertaintyInterval, Identity) {
  const Eigen::Matrix2d cov = Eigen::Matrix2d::Identity();
  EXPECT_TRUE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(0, 0), 1));
  EXPECT_TRUE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(1, 0), 1));
  EXPECT_TRUE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(0, 1), 1));
  EXPECT_TRUE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(2, 0), 2));
  EXPECT_TRUE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(0, 2), 2));
  EXPECT_FALSE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(1.01, 0), 1));
  EXPECT_FALSE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(0, 1.01), 1));
  EXPECT_FALSE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(2.01, 0), 2));
  EXPECT_FALSE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(0, 2.01), 2));
  EXPECT_TRUE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(2.01, 0), 2.01));
  EXPECT_TRUE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(0, 2.01), 2.01));
}

TEST(IsPointWithinUncertaintyInterval, NonZeroMean) {
  const Eigen::Vector2d mean = Eigen::Vector2d(2, 3);
  const Eigen::Matrix2d cov = Eigen::Matrix2d::Identity();
  EXPECT_FALSE(
      IsPointWithinUncertaintyInterval(mean, cov, Eigen::Vector2d(0, 0), 1));
  EXPECT_TRUE(
      IsPointWithinUncertaintyInterval(mean, cov, Eigen::Vector2d(2, 3), 1));
  EXPECT_TRUE(
      IsPointWithinUncertaintyInterval(mean, cov, Eigen::Vector2d(1, 3), 1));
  EXPECT_TRUE(
      IsPointWithinUncertaintyInterval(mean, cov, Eigen::Vector2d(2, 2), 1));
}

TEST(IsPointWithinUncertaintyInterval, Isotropic) {
  const Eigen::Matrix2d cov = (Eigen::Matrix2d() << 1, 0, 0, 4).finished();
  EXPECT_TRUE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(0, 0), 1));
  EXPECT_TRUE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(1, 0), 1));
  EXPECT_TRUE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(0, 2), 1));
  EXPECT_FALSE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(1.01, 0), 1));
  EXPECT_FALSE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(0, 2.01), 1));
  EXPECT_TRUE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(1.01, 0), 1.01));
  EXPECT_TRUE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(0, 2.01), 2.01));
}

TEST(IsPointWithinUncertaintyInterval, Anisotropic) {
  const Eigen::Matrix2d cov = (Eigen::Matrix2d() << 1, 0.5, 0.5, 1).finished();
  EXPECT_TRUE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(0, 0), 1));
  EXPECT_FALSE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(1, 0), 1));
  EXPECT_FALSE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(0, 1), 1));
  EXPECT_TRUE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(std::sqrt(0.75), std::sqrt(0.75)), 1));
  EXPECT_TRUE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, -Eigen::Vector2d(std::sqrt(0.75), std::sqrt(0.75)), 1));
  EXPECT_FALSE(AreSymmetricPointsWithinZeroMeanUncertaintyInterval(
      cov, Eigen::Vector2d(std::sqrt(0.75) + 0.01, std::sqrt(0.75) + 0.01), 1));
}

}  // namespace
}  // namespace colmap

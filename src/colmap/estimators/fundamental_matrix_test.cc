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

#include "colmap/estimators/fundamental_matrix.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(FundamentalMatrix, SevenPoint) {
  const double points1_raw[] = {0.4964,
                                1.0577,
                                0.3650,
                                -0.0919,
                                -0.5412,
                                0.0159,
                                -0.5239,
                                0.9467,
                                0.3467,
                                0.5301,
                                0.2797,
                                0.0012,
                                -0.1986,
                                0.0460};

  const double points2_raw[] = {0.7570,
                                2.7340,
                                0.3961,
                                0.6981,
                                -0.6014,
                                0.7110,
                                -0.7385,
                                2.2712,
                                0.4177,
                                1.2132,
                                0.3052,
                                0.4835,
                                -0.2171,
                                0.5057};

  const size_t kNumPoints = 7;

  std::vector<Eigen::Vector2d> points1(kNumPoints);
  std::vector<Eigen::Vector2d> points2(kNumPoints);
  for (size_t i = 0; i < kNumPoints; ++i) {
    points1[i] = Eigen::Vector2d(points1_raw[2 * i], points1_raw[2 * i + 1]);
    points2[i] = Eigen::Vector2d(points2_raw[2 * i], points2_raw[2 * i + 1]);
  }

  FundamentalMatrixSevenPointEstimator estimator;
  std::vector<Eigen::Matrix3d> models;
  estimator.Estimate(points1, points2, &models);

  ASSERT_EQ(models.size(), 1);
  const auto& F = models[0];

  // Reference values obtained from Matlab.
  EXPECT_NEAR(F(0, 0), 4.81441976, 1e-6);
  EXPECT_NEAR(F(0, 1), -8.16978909, 1e-6);
  EXPECT_NEAR(F(0, 2), 6.73133404, 1e-6);
  EXPECT_NEAR(F(1, 0), 5.16247992, 1e-6);
  EXPECT_NEAR(F(1, 1), 0.19325606, 1e-6);
  EXPECT_NEAR(F(1, 2), -2.87239381, 1e-6);
  EXPECT_NEAR(F(2, 0), -9.92570126, 1e-6);
  EXPECT_NEAR(F(2, 1), 3.64159554, 1e-6);
  EXPECT_NEAR(F(2, 2), 1., 1e-6);
}

TEST(FundamentalMatrix, EightPoint) {
  const double points1_raw[] = {1.839035,
                                1.924743,
                                0.543582,
                                0.375221,
                                0.473240,
                                0.142522,
                                0.964910,
                                0.598376,
                                0.102388,
                                0.140092,
                                15.994343,
                                9.622164,
                                0.285901,
                                0.430055,
                                0.091150,
                                0.254594};

  const double points2_raw[] = {
      1.002114,
      1.129644,
      1.521742,
      1.846002,
      1.084332,
      0.275134,
      0.293328,
      0.588992,
      0.839509,
      0.087290,
      1.779735,
      1.116857,
      0.878616,
      0.602447,
      0.642616,
      1.028681,
  };

  const size_t kNumPoints = 8;
  std::vector<Eigen::Vector2d> points1(kNumPoints);
  std::vector<Eigen::Vector2d> points2(kNumPoints);
  for (size_t i = 0; i < kNumPoints; ++i) {
    points1[i] = Eigen::Vector2d(points1_raw[2 * i], points1_raw[2 * i + 1]);
    points2[i] = Eigen::Vector2d(points2_raw[2 * i], points2_raw[2 * i + 1]);
  }

  FundamentalMatrixEightPointEstimator estimator;
  std::vector<Eigen::Matrix3d> models;
  estimator.Estimate(points1, points2, &models);

  ASSERT_EQ(models.size(), 1);
  const auto& F = models[0];

  // Reference values obtained from Matlab.
  EXPECT_TRUE(std::abs(F(0, 0) - -0.217859) < 1e-5);
  EXPECT_TRUE(std::abs(F(0, 1) - 0.419282) < 1e-5);
  EXPECT_TRUE(std::abs(F(0, 2) - -0.0343075) < 1e-5);
  EXPECT_TRUE(std::abs(F(1, 0) - -0.0717941) < 1e-5);
  EXPECT_TRUE(std::abs(F(1, 1) - 0.0451643) < 1e-5);
  EXPECT_TRUE(std::abs(F(1, 2) - 0.0216073) < 1e-5);
  EXPECT_TRUE(std::abs(F(2, 0) - 0.248062) < 1e-5);
  EXPECT_TRUE(std::abs(F(2, 1) - -0.429478) < 1e-5);
  EXPECT_TRUE(std::abs(F(2, 2) - 0.0221019) < 1e-5);
}

}  // namespace
}  // namespace colmap

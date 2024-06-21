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

#include "colmap/estimators/homography_matrix.h"

#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

class HomographyMatrixTests : public ::testing::TestWithParam<size_t> {};

TEST_P(HomographyMatrixTests, Nominal) {
  const size_t kNumPoints = GetParam();
  for (int x = 0; x < 10; ++x) {
    Eigen::Matrix3d expected_H;
    expected_H << x, 0.2, 0.3, 30, 0.2, 0.1, 0.3, 20, 1;

    std::vector<Eigen::Vector2d> src;
    std::vector<Eigen::Vector2d> dst;
    for (size_t i = 0; i < kNumPoints; ++i) {
      src.push_back(Eigen::Vector2d::Random());
      dst.push_back((expected_H * src[i].homogeneous()).hnormalized());
    }

    HomographyMatrixEstimator estimator;
    std::vector<Eigen::Matrix3d> models;
    estimator.Estimate(src, dst, &models);

    ASSERT_EQ(models.size(), 1);

    std::vector<double> residuals;
    estimator.Residuals(src, dst, models[0], &residuals);

    for (size_t i = 0; i < kNumPoints; ++i) {
      EXPECT_LT(residuals[i], 1e-6);
    }
  }
}

// Test numerical stability with large coordinates. This is to ensure that the
// homography matrix estimator is numerically stable despite not using
// coordinate normalization. We can do this because of double precision.
TEST_P(HomographyMatrixTests, NumericalStability) {
  const size_t kNumPoints = GetParam();
  constexpr double kCoordinateScale = 1e6;
  for (int x = 1; x < 10; ++x) {
    Eigen::Matrix3d expected_H = Eigen::Matrix3d::Identity();
    expected_H(0, 0) = x;

    std::vector<Eigen::Vector2d> src;
    std::vector<Eigen::Vector2d> dst;
    for (size_t i = 0; i < kNumPoints; ++i) {
      src.push_back(Eigen::Vector2d::Random() * kCoordinateScale);
      dst.push_back((expected_H * src[i].homogeneous()).hnormalized());
    }

    HomographyMatrixEstimator estimator;
    std::vector<Eigen::Matrix3d> models;
    estimator.Estimate(src, dst, &models);
    ASSERT_EQ(models.size(), 1);

    std::vector<double> residuals;
    estimator.Residuals(src, dst, models[0], &residuals);

    for (size_t i = 0; i < kNumPoints; ++i) {
      EXPECT_LT(residuals[i], 1e-6);
    }
  }
}

TEST_P(HomographyMatrixTests, NoiseStability) {
  const size_t kNumPoints = GetParam();
  constexpr double kNoise = 1e-3;
  for (int x = 1; x < 10; ++x) {
    Eigen::Matrix3d expected_H = Eigen::Matrix3d::Identity();
    expected_H(0, 0) = x;

    std::vector<Eigen::Vector2d> src;
    std::vector<Eigen::Vector2d> dst;
    for (size_t i = 0; i < kNumPoints; ++i) {
      src.push_back(Eigen::Vector2d::Random());
      dst.push_back((expected_H * src[i].homogeneous()).hnormalized() +
                    Eigen::Vector2d::Random() * kNoise);
    }

    HomographyMatrixEstimator estimator;
    std::vector<Eigen::Matrix3d> models;
    estimator.Estimate(src, dst, &models);
    ASSERT_EQ(models.size(), 1);

    std::vector<double> residuals;
    estimator.Residuals(src, dst, models[0], &residuals);

    for (size_t i = 0; i < kNumPoints; ++i) {
      EXPECT_LT(residuals[i], 1e-5);
    }
  }
}

TEST_P(HomographyMatrixTests, Degenerate) {
  const size_t kNumPoints = GetParam();
  constexpr double kNoise = 1e-3;

  for (int x = 0; x < 10; ++x) {
    Eigen::Matrix3d expected_H;
    expected_H << x, 0.2, 0.3, 30, 0.2, 0.1, 0.3, 20, 1;

    std::vector<Eigen::Vector2d> src;
    src.emplace_back(2, 1);
    src.emplace_back(3, 1);
    src.emplace_back(10, 30);
    ASSERT_GE(kNumPoints, 4);
    const size_t num_redundant_points = kNumPoints - src.size();
    for (size_t i = 0; i < num_redundant_points; ++i) {
      src.emplace_back(src.front());
    }

    std::vector<Eigen::Vector2d> dst;
    for (size_t i = 0; i < src.size(); ++i) {
      const Eigen::Vector3d dsth = expected_H * src[i].homogeneous();
      dst.push_back(dsth.hnormalized() + Eigen::Vector2d::Random() * kNoise);
    }

    HomographyMatrixEstimator estimator;
    std::vector<Eigen::Matrix3d> models;
    estimator.Estimate(src, dst, &models);

    ASSERT_EQ(models.size(), 0);
  }
}

INSTANTIATE_TEST_SUITE_P(HomographyMatrix,
                         HomographyMatrixTests,
                         ::testing::Values(4, 8, 64, 1024));

}  // namespace
}  // namespace colmap

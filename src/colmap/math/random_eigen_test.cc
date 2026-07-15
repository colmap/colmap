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

#include "colmap/math/random_eigen.h"

#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(RandomEigenVectord, Range) {
  SetPRNGSeed(0);
  for (int i = 0; i < 1000; ++i) {
    const Eigen::Vector3d vector = RandomEigenVectord<3>();
    EXPECT_TRUE((vector.array() >= -1).all());
    EXPECT_TRUE((vector.array() <= 1).all());
  }
}

TEST(RandomEigenVectord, Deterministic) {
  SetPRNGSeed(42);
  const Eigen::Vector4d vector1 = RandomEigenVectord<4>();
  SetPRNGSeed(42);
  const Eigen::Vector4d vector2 = RandomEigenVectord<4>();
  EXPECT_EQ(vector1, vector2);
}

TEST(RandomEigenVectorf, Range) {
  SetPRNGSeed(0);
  for (int i = 0; i < 1000; ++i) {
    const Eigen::Vector2f vector = RandomEigenVectorf<2>();
    EXPECT_TRUE((vector.array() >= -1).all());
    EXPECT_TRUE((vector.array() <= 1).all());
  }
}

TEST(RandomEigenVectorXd, Dynamic) {
  SetPRNGSeed(0);
  const Eigen::VectorXd vector = RandomEigenVectorXd(7);
  EXPECT_EQ(vector.size(), 7);
  EXPECT_TRUE((vector.array() >= -1).all());
  EXPECT_TRUE((vector.array() <= 1).all());
}

TEST(RandomEigenMatrixd, Range) {
  SetPRNGSeed(0);
  const Eigen::Matrix<double, 3, 4> matrix = RandomEigenMatrixd<3, 4>();
  EXPECT_TRUE((matrix.array() >= -1).all());
  EXPECT_TRUE((matrix.array() <= 1).all());
}

TEST(RandomEigenMatrixXf, Dynamic) {
  SetPRNGSeed(0);
  const Eigen::MatrixXf matrix = RandomEigenMatrixXf(3, 5);
  EXPECT_EQ(matrix.rows(), 3);
  EXPECT_EQ(matrix.cols(), 5);
  EXPECT_TRUE((matrix.array() >= -1).all());
  EXPECT_TRUE((matrix.array() <= 1).all());
}

TEST(RandomEigenQuaterniond, Unit) {
  SetPRNGSeed(0);
  for (int i = 0; i < 1000; ++i) {
    const Eigen::Quaterniond quat = RandomEigenQuaterniond();
    EXPECT_NEAR(quat.norm(), 1.0, 1e-9);
  }
}

TEST(RandomEigenQuaterniond, Deterministic) {
  SetPRNGSeed(42);
  const Eigen::Quaterniond quat1 = RandomEigenQuaterniond();
  SetPRNGSeed(42);
  const Eigen::Quaterniond quat2 = RandomEigenQuaterniond();
  EXPECT_EQ(quat1.coeffs(), quat2.coeffs());
}

}  // namespace
}  // namespace colmap

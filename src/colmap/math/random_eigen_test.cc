// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/math/random_eigen.h"

#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(RandomEigenVectord, Range) {
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
  for (int i = 0; i < 1000; ++i) {
    const Eigen::Vector2f vector = RandomEigenVectorf<2>();
    EXPECT_TRUE((vector.array() >= -1).all());
    EXPECT_TRUE((vector.array() <= 1).all());
  }
}

TEST(RandomEigenVectorXd, Dynamic) {
  const Eigen::VectorXd vector = RandomEigenVectorXd(7);
  EXPECT_EQ(vector.size(), 7);
  EXPECT_TRUE((vector.array() >= -1).all());
  EXPECT_TRUE((vector.array() <= 1).all());
}

TEST(RandomEigenMatrixd, Range) {
  const Eigen::Matrix<double, 3, 4> matrix = RandomEigenMatrixd<3, 4>();
  EXPECT_TRUE((matrix.array() >= -1).all());
  EXPECT_TRUE((matrix.array() <= 1).all());
}

TEST(RandomEigenMatrixXf, Dynamic) {
  const Eigen::MatrixXf matrix = RandomEigenMatrixXf(3, 5);
  EXPECT_EQ(matrix.rows(), 3);
  EXPECT_EQ(matrix.cols(), 5);
  EXPECT_TRUE((matrix.array() >= -1).all());
  EXPECT_TRUE((matrix.array() <= 1).all());
}

TEST(RandomEigenQuaterniond, Unit) {
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

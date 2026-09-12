// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/geometry/sim3.h"

#include "colmap/math/random.h"
#include "colmap/math/random_eigen.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/eigen_matchers.h"
#include "colmap/util/testing.h"

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

Sim3d TestSim3d() {
  return Sim3d(RandomUniformReal<double>(0.1, 10),
               RandomEigenQuaterniond(),
               RandomEigenVectord<3>());
}

TEST(Sim3d, Default) {
  const Sim3d tform;
  EXPECT_EQ(tform.scale(), 1);
  EXPECT_EQ(tform.rotation().coeffs(), Eigen::Quaterniond::Identity().coeffs());
  EXPECT_EQ(tform.translation(), Eigen::Vector3d::Zero());
}

TEST(Sim3d, Equals) {
  Sim3d tform;
  Sim3d other = tform;
  EXPECT_EQ(tform, other);
  tform.translation().x() = 1;
  EXPECT_NE(tform, other);
  other.translation().x() = 1;
  EXPECT_EQ(tform, other);
}

TEST(Sim3d, Print) {
  Sim3d tform;
  std::ostringstream stream;
  stream << tform;
  EXPECT_EQ(
      stream.str(),
      "Sim3d(scale=1, rotation_xyzw=[0, 0, 0, 1], translation=[0, 0, 0])");
}

TEST(Sim3d, Inverse) {
  const Sim3d b_from_a = TestSim3d();
  const Sim3d a_from_b = Inverse(b_from_a);
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector3d x_in_a = RandomEigenVectord<3>();
    const Eigen::Vector3d x_in_b = b_from_a * x_in_a;
    EXPECT_THAT(a_from_b * x_in_b, EigenMatrixNear(x_in_a, 1e-6));
  }
}

TEST(Sim3d, ToMatrix) {
  const Sim3d b_from_a = TestSim3d();
  const Eigen::Matrix3x4d b_from_a_mat = b_from_a.ToMatrix();
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector3d x_in_a = RandomEigenVectord<3>();
    EXPECT_LT((b_from_a * x_in_a - b_from_a_mat * x_in_a.homogeneous()).norm(),
              1e-6);
  }
}

TEST(Sim3d, FromMatrix) {
  const Sim3d b1_from_a = TestSim3d();
  const Sim3d b2_from_a = Sim3d::FromMatrix(b1_from_a.ToMatrix());
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector3d x_in_a = RandomEigenVectord<3>();
    EXPECT_THAT(b1_from_a * x_in_a, EigenMatrixNear(b2_from_a * x_in_a, 1e-6));
  }
}

TEST(Sim3d, ApplyScaleOnly) {
  const Sim3d b_from_a(
      2, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  EXPECT_LT(
      (b_from_a * Eigen::Vector3d(1, 2, 3) - Eigen::Vector3d(2, 4, 6)).norm(),
      1e-6);
}

TEST(Sim3d, ApplyTranslationOnly) {
  const Sim3d b_from_a(
      1, Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 2, 3));
  EXPECT_LT(
      (b_from_a * Eigen::Vector3d(1, 2, 3) - Eigen::Vector3d(2, 4, 6)).norm(),
      1e-6);
}

TEST(Sim3d, ApplyRotationOnly) {
  const Sim3d b_from_a(1,
                       Eigen::Quaterniond(Eigen::AngleAxisd(
                           EIGEN_PI / 2, Eigen::Vector3d::UnitX())),
                       Eigen::Vector3d::Zero());
  EXPECT_LT(
      (b_from_a * Eigen::Vector3d(1, 2, 3) - Eigen::Vector3d(1, -3, 2)).norm(),
      1e-6);
}

TEST(Sim3d, ApplyScaleRotationTranslation) {
  const Sim3d b_from_a(2,
                       Eigen::Quaterniond(Eigen::AngleAxisd(
                           EIGEN_PI / 2, Eigen::Vector3d::UnitX())),
                       Eigen::Vector3d(1, 2, 3));
  EXPECT_LT(
      (b_from_a * Eigen::Vector3d(1, 2, 3) - Eigen::Vector3d(3, -4, 7)).norm(),
      1e-6);
}

TEST(Rigid3d, ApplyChain) {
  const Sim3d b_from_a = TestSim3d();
  const Sim3d c_from_b = TestSim3d();
  const Sim3d d_from_c = TestSim3d();
  const Eigen::Vector3d x_in_a = RandomEigenVectord<3>();
  const Eigen::Vector3d x_in_b = b_from_a * x_in_a;
  const Eigen::Vector3d x_in_c = c_from_b * x_in_b;
  const Eigen::Vector3d x_in_d = d_from_c * x_in_c;
  EXPECT_EQ((d_from_c * (c_from_b * (b_from_a * x_in_a))), x_in_d);
}

TEST(Sim3d, Compose) {
  const Sim3d b_from_a = TestSim3d();
  const Sim3d c_from_b = TestSim3d();
  const Sim3d d_from_c = TestSim3d();
  const Sim3d d_from_a = d_from_c * c_from_b * b_from_a;
  const Eigen::Vector3d x_in_a = RandomEigenVectord<3>();
  const Eigen::Vector3d x_in_b = b_from_a * x_in_a;
  const Eigen::Vector3d x_in_c = c_from_b * x_in_b;
  const Eigen::Vector3d x_in_d = d_from_c * x_in_c;
  EXPECT_THAT(d_from_a * x_in_a, EigenMatrixNear(x_in_d, 1e-6));
}

TEST(Sim3d, ToFromFile) {
  const auto path = CreateTestDir() / "file.txt";
  const Sim3d written = TestSim3d();
  written.ToFile(path);
  const Sim3d read = Sim3d::FromFile(path);
  EXPECT_EQ(written.scale(), read.scale());
  EXPECT_EQ(written.rotation().coeffs(), read.rotation().coeffs());
  EXPECT_EQ(written.translation(), read.translation());
}

}  // namespace
}  // namespace colmap

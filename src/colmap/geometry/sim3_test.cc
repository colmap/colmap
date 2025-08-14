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

#include "colmap/geometry/sim3.h"

#include "colmap/math/random.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/eigen_matchers.h"
#include "colmap/util/testing.h"

#include <fstream>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

Sim3d TestSim3d() {
  return Sim3d(RandomUniformReal<double>(0.1, 10),
               Eigen::Quaterniond::UnitRandom(),
               Eigen::Vector3d::Random());
}

TEST(Sim3d, Default) {
  const Sim3d tform;
  EXPECT_EQ(tform.scale, 1);
  EXPECT_EQ(tform.rotation.coeffs(), Eigen::Quaterniond::Identity().coeffs());
  EXPECT_EQ(tform.translation, Eigen::Vector3d::Zero());
}

TEST(Sim3d, Equals) {
  Sim3d tform;
  Sim3d other = tform;
  EXPECT_EQ(tform, other);
  tform.translation.x() = 1;
  EXPECT_NE(tform, other);
  other.translation.x() = 1;
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
    const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
    const Eigen::Vector3d x_in_b = b_from_a * x_in_a;
    EXPECT_LT((a_from_b * x_in_b - x_in_a).norm(), 1e-6);
  }
}

TEST(Sim3d, ToMatrix) {
  const Sim3d b_from_a = TestSim3d();
  const Eigen::Matrix3x4d b_from_a_mat = b_from_a.ToMatrix();
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
    EXPECT_LT((b_from_a * x_in_a - b_from_a_mat * x_in_a.homogeneous()).norm(),
              1e-6);
  }
}

TEST(Sim3d, FromMatrix) {
  const Sim3d b1_from_a = TestSim3d();
  const Sim3d b2_from_a = Sim3d::FromMatrix(b1_from_a.ToMatrix());
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
    EXPECT_LT((b1_from_a * x_in_a - b2_from_a * x_in_a).norm(), 1e-6);
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
  const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
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
  const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
  const Eigen::Vector3d x_in_b = b_from_a * x_in_a;
  const Eigen::Vector3d x_in_c = c_from_b * x_in_b;
  const Eigen::Vector3d x_in_d = d_from_c * x_in_c;
  EXPECT_LT((d_from_a * x_in_a - x_in_d).norm(), 1e-6);
}

TEST(Sim3d, Adjoint) {
  const Sim3d b_from_a = TestSim3d();
  const Eigen::Matrix<double, 7, 7> adjoint = b_from_a.Adjoint();
  const Eigen::Matrix<double, 7, 7> adjoint_inv = b_from_a.AdjointInverse();
  EXPECT_LT(
      (adjoint * adjoint_inv - Eigen::Matrix<double, 7, 7>::Identity()).norm(),
      1e-6);
  const Sim3d a_from_b = Inverse(b_from_a);
  const Eigen::Matrix<double, 7, 7> adjoint_a_from_b = a_from_b.Adjoint();
  EXPECT_THAT(adjoint_inv, EigenMatrixNear(adjoint_a_from_b, 1e-6));
}

TEST(Sim3d, CovarianceForInverse) {
  const Sim3d b_from_a = TestSim3d();
  const Eigen::Matrix<double, 7, 7> A = Eigen::Matrix<double, 7, 7>::Random();
  const Eigen::Matrix<double, 7, 7> cov_b_from_a = A * A.transpose();
  const Eigen::Matrix<double, 7, 7> cov_a_from_b =
      PropagateCovarianceForInverse(b_from_a, cov_b_from_a);
  const Sim3d a_from_b = Inverse(b_from_a);
  const Eigen::Matrix<double, 7, 7> cov_b_from_a_test =
      PropagateCovarianceForInverse(a_from_b, cov_a_from_b);
  EXPECT_THAT(cov_b_from_a_test, EigenMatrixNear(cov_b_from_a, 1e-6));
}

TEST(Sim3d, CovarianceForRelativeSim3d_PerfectCorrelation) {
  const Sim3d world_from_a = TestSim3d();
  const Sim3d world_from_b = TestSim3d();
  const Eigen::Matrix<double, 7, 7> A = Eigen::Matrix<double, 7, 7>::Random();
  const Eigen::Matrix<double, 7, 7> covar_subblock = A * A.transpose();

  Eigen::Matrix<double, 14, 14> covar_world_from_cam;
  covar_world_from_cam.block<7, 7>(0, 0) = covar_subblock;
  covar_world_from_cam.block<7, 7>(0, 7) = covar_subblock;
  covar_world_from_cam.block<7, 7>(7, 0) = covar_subblock;
  covar_world_from_cam.block<7, 7>(7, 7) = covar_subblock;

  const Sim3d a_from_world = Inverse(world_from_a);
  const Sim3d b_from_world = Inverse(world_from_b);
  Eigen::Matrix<double, 14, 14> J0;
  J0.setZero();
  J0.block<7, 7>(0, 0) = -world_from_a.AdjointInverse();
  J0.block<7, 7>(7, 7) = -world_from_b.AdjointInverse();
  const Eigen::Matrix<double, 14, 14> covar_cam_from_world =
      J0 * covar_world_from_cam * J0.transpose();

  const Eigen::Matrix<double, 7, 7> b_cov_from_a =
      PropagateCovarianceForRelative(
          a_from_world, b_from_world, covar_cam_from_world);
  EXPECT_LT(b_cov_from_a.norm(), 1e-6);
}

TEST(Sim3d, CovarianceForRelativeSim3d) {
  const Sim3d a_from_world = TestSim3d();
  const Sim3d b_from_world = TestSim3d();
  const Eigen::Matrix<double, 14, 14> A =
      Eigen::Matrix<double, 14, 14>::Random();
  const Eigen::Matrix<double, 14, 14> covar = A * A.transpose();

  const Eigen::Matrix<double, 7, 7> b_cov_from_a =
      PropagateCovarianceForRelative(a_from_world, b_from_world, covar);

  Eigen::Matrix<double, 14, 14> J0;
  J0.setZero();
  J0.block<7, 7>(0, 0) = -a_from_world.AdjointInverse();
  J0.block<7, 7>(7, 7) = Eigen::Matrix<double, 7, 7>::Identity();
  const Eigen::Matrix<double, 14, 14> covar_in_right =
      J0 * covar * J0.transpose();

  Eigen::Matrix<double, 7, 14> J_in_right;
  J_in_right.block<7, 7>(0, 0) = b_from_world.Adjoint();
  J_in_right.block<7, 7>(0, 7) = Eigen::Matrix<double, 7, 7>::Identity();
  const Eigen::Matrix<double, 7, 7> a_cov_from_b_right =
      J_in_right * covar_in_right * J_in_right.transpose();
  EXPECT_THAT(b_cov_from_a, EigenMatrixNear(a_cov_from_b_right, 1e-6));
}

TEST(Sim3d, CovariancePropagation_Composed_vs_Relative) {
  const Sim3d a_from_b = TestSim3d();
  const Sim3d b_from_c = TestSim3d();
  const Eigen::Matrix<double, 14, 14> A =
      Eigen::Matrix<double, 14, 14>::Random();
  const Eigen::Matrix<double, 14, 14> covar = A * A.transpose();

  const Eigen::Matrix<double, 7, 7> a_cov_from_c_composed =
      PropagateCovarianceForCompose(a_from_b, covar);

  const Sim3d c_from_b = Inverse(b_from_c);
  Eigen::Matrix<double, 14, 14> J0;
  J0.setZero();
  J0.block<7, 7>(7, 0) = Eigen::Matrix<double, 7, 7>::Identity();
  J0.block<7, 7>(0, 7) = -b_from_c.AdjointInverse();
  const Eigen::Matrix<double, 14, 14> covar_x_from_b =
      J0 * covar * J0.transpose();
  const Eigen::Matrix<double, 7, 7> a_cov_from_c_relative =
      PropagateCovarianceForRelative(c_from_b, a_from_b, covar_x_from_b);

  EXPECT_THAT(a_cov_from_c_composed,
              EigenMatrixNear(a_cov_from_c_relative, 1e-6));
}

TEST(Rigid3d, CovariancePropagationForTransformPoint) {
  const Sim3d sim3 = TestSim3d();

  const Eigen::Matrix3d A = Eigen::Matrix3d::Random();
  const Eigen::Matrix3d cov_point = A * A.transpose();

  const Eigen::Matrix3d cov_transformed =
      PropagateCovarianceForTransformPoint(sim3, cov_point);
  const Eigen::Matrix3d cov_recovered =
      PropagateCovarianceForTransformPoint(Inverse(sim3), cov_transformed);

  EXPECT_THAT(cov_recovered, EigenMatrixNear(cov_point, 1e-6));
}

TEST(Sim3d, ToFromFile) {
  const std::string path = CreateTestDir() + "/file.txt";
  const Sim3d written = TestSim3d();
  written.ToFile(path);
  const Sim3d read = Sim3d::FromFile(path);
  EXPECT_EQ(written.scale, read.scale);
  EXPECT_EQ(written.rotation.coeffs(), read.rotation.coeffs());
  EXPECT_EQ(written.translation, read.translation);
}

}  // namespace
}  // namespace colmap

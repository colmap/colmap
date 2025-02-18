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

#include "colmap/geometry/rigid3.h"

#include "colmap/util/eigen_matchers.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

Rigid3d TestRigid3d() {
  return Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
}

TEST(CrossProductMatrix, Nominal) {
  EXPECT_EQ(CrossProductMatrix(Eigen::Vector3d(0, 0, 0)),
            Eigen::Matrix3d::Zero());
  Eigen::Matrix3d ref_matrix;
  ref_matrix << 0, -3, 2, 3, 0, -1, -2, 1, 0;
  EXPECT_EQ(CrossProductMatrix(Eigen::Vector3d(1, 2, 3)), ref_matrix);
}

TEST(Rigid3d, Default) {
  const Rigid3d tform;
  EXPECT_EQ(tform.rotation.coeffs(), Eigen::Quaterniond::Identity().coeffs());
  EXPECT_EQ(tform.translation, Eigen::Vector3d::Zero());
}

TEST(Rigid3d, Equals) {
  Rigid3d tform;
  Rigid3d other = tform;
  EXPECT_EQ(tform, other);
  tform.translation.x() = 1;
  EXPECT_NE(tform, other);
  other.translation.x() = 1;
  EXPECT_EQ(tform, other);
}

TEST(Rigid3d, Print) {
  Rigid3d tform;
  std::ostringstream stream;
  stream << tform;
  EXPECT_EQ(stream.str(),
            "Rigid3d(rotation_xyzw=[0, 0, 0, 1], translation=[0, 0, 0])");
}

TEST(Rigid3d, Inverse) {
  const Rigid3d b_from_a = TestRigid3d();
  const Rigid3d a_from_b = Inverse(b_from_a);
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
    const Eigen::Vector3d x_in_b = b_from_a * x_in_a;
    EXPECT_LT((a_from_b * x_in_b - x_in_a).norm(), 1e-6);
  }
}

TEST(Rigid3d, ToMatrix) {
  const Rigid3d b_from_a = TestRigid3d();
  const Eigen::Matrix3x4d b_from_a_mat = b_from_a.ToMatrix();
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
    EXPECT_LT((b_from_a * x_in_a - b_from_a_mat * x_in_a.homogeneous()).norm(),
              1e-6);
  }
}

TEST(Rigid3d, FromMatrix) {
  const Rigid3d b1_from_a = TestRigid3d();
  const Rigid3d b2_from_a = Rigid3d::FromMatrix(b1_from_a.ToMatrix());
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
    EXPECT_LT((b1_from_a * x_in_a - b2_from_a * x_in_a).norm(), 1e-6);
  }
}

TEST(Rigid3d, ApplyNoRotation) {
  const Rigid3d b_from_a(Eigen::Quaterniond::Identity(),
                         Eigen::Vector3d(1, 2, 3));
  EXPECT_LT(
      (b_from_a * Eigen::Vector3d(1, 2, 3) - Eigen::Vector3d(2, 4, 6)).norm(),
      1e-6);
}

TEST(Rigid3d, ApplyNoTranslation) {
  const Rigid3d b_from_a(Eigen::Quaterniond(Eigen::AngleAxisd(
                             EIGEN_PI / 2, Eigen::Vector3d::UnitX())),
                         Eigen::Vector3d::Zero());
  EXPECT_LT(
      (b_from_a * Eigen::Vector3d(1, 2, 3) - Eigen::Vector3d(1, -3, 2)).norm(),
      1e-6);
}

TEST(Rigid3d, ApplyRotationTranslation) {
  const Rigid3d b_from_a(Eigen::Quaterniond(Eigen::AngleAxisd(
                             EIGEN_PI / 2, Eigen::Vector3d::UnitX())),
                         Eigen::Vector3d(1, 2, 3));
  EXPECT_LT(
      (b_from_a * Eigen::Vector3d(1, 2, 3) - Eigen::Vector3d(2, -1, 5)).norm(),
      1e-6);
}

TEST(Rigid3d, ApplyChain) {
  const Rigid3d b_from_a = TestRigid3d();
  const Rigid3d c_from_b = TestRigid3d();
  const Rigid3d d_from_c = TestRigid3d();
  const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
  const Eigen::Vector3d x_in_b = b_from_a * x_in_a;
  const Eigen::Vector3d x_in_c = c_from_b * x_in_b;
  const Eigen::Vector3d x_in_d = d_from_c * x_in_c;
  EXPECT_EQ((d_from_c * (c_from_b * (b_from_a * x_in_a))), x_in_d);
}

TEST(Rigid3d, Compose) {
  const Rigid3d b_from_a = TestRigid3d();
  const Rigid3d c_from_b = TestRigid3d();
  const Rigid3d d_from_c = TestRigid3d();
  const Rigid3d d_from_a = d_from_c * c_from_b * b_from_a;
  const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
  const Eigen::Vector3d x_in_b = b_from_a * x_in_a;
  const Eigen::Vector3d x_in_c = c_from_b * x_in_b;
  const Eigen::Vector3d x_in_d = d_from_c * x_in_c;
  EXPECT_LT((d_from_a * x_in_a - x_in_d).norm(), 1e-6);
}

TEST(Rigid3d, Adjoint) {
  const Rigid3d b_from_a = TestRigid3d();
  const Eigen::Matrix6d adjoint = b_from_a.Adjoint();
  const Eigen::Matrix6d adjoint_inv = b_from_a.AdjointInverse();
  EXPECT_LT((adjoint * adjoint_inv - Eigen::Matrix6d::Identity()).norm(), 1e-6);
  const Rigid3d a_from_b = Inverse(b_from_a);
  const Eigen::Matrix6d adjoint_a_from_b = a_from_b.Adjoint();
  EXPECT_THAT(adjoint_inv, EigenMatrixNear(adjoint_a_from_b, 1e-6));
}

TEST(Rigid3d, CovarianceForInverse) {
  const Rigid3d b_from_a = TestRigid3d();
  const Eigen::Matrix6d A = Eigen::Matrix6d::Random();
  const Eigen::Matrix6d cov_b_from_a = A * A.transpose();
  const Eigen::Matrix6d cov_a_from_b =
      GetCovarianceForRigid3dInverse(b_from_a, cov_b_from_a);
  const Rigid3d a_from_b = Inverse(b_from_a);
  const Eigen::Matrix6d cov_b_from_a_test =
      GetCovarianceForRigid3dInverse(a_from_b, cov_a_from_b);
  EXPECT_THAT(cov_b_from_a_test, EigenMatrixNear(cov_b_from_a, 1e-6));
}

TEST(Rigid3d, CovarianceForRelativeRigid3d_PerfectCorrelation) {
  const Rigid3d world_from_a = TestRigid3d();
  const Rigid3d world_from_b = TestRigid3d();
  const Eigen::Matrix6d A = Eigen::Matrix6d::Random();
  const Eigen::Matrix6d covar_subblock = A * A.transpose();
  // Two poses are perfectly correlated in world frame
  Eigen::Matrix<double, 12, 12> covar_world_from_cam;
  covar_world_from_cam.block<6, 6>(0, 0) = covar_subblock;
  covar_world_from_cam.block<6, 6>(0, 6) = covar_subblock;
  covar_world_from_cam.block<6, 6>(6, 0) = covar_subblock;
  covar_world_from_cam.block<6, 6>(6, 6) = covar_subblock;
  // Invert poses
  const Rigid3d a_from_world = Inverse(world_from_a);
  const Rigid3d b_from_world = Inverse(world_from_b);
  Eigen::Matrix<double, 12, 12> J0;
  J0.setZero();
  J0.block<6, 6>(0, 0) = -world_from_a.AdjointInverse();
  J0.block<6, 6>(6, 6) = -world_from_b.AdjointInverse();
  const Eigen::Matrix<double, 12, 12> covar_cam_from_world =
      J0 * covar_world_from_cam * J0.transpose();
  // Calculate relative pose covariance, which should be a zero matrix.
  const Eigen::Matrix6d b_cov_from_a = GetCovarianceForRelativeRigid3d(
      a_from_world, b_from_world, covar_cam_from_world);
  EXPECT_LT(b_cov_from_a.norm(), 1e-6);
}

TEST(Rigid3d, CovarianceForRelativeRigid3d) {
  const Rigid3d a_from_world = TestRigid3d();
  const Rigid3d b_from_world = TestRigid3d();
  const Eigen::Matrix<double, 12, 12> A =
      Eigen::Matrix<double, 12, 12>::Random();
  const Eigen::Matrix<double, 12, 12> covar = A * A.transpose();

  // Ours (in left convention)
  const Eigen::Matrix6d b_cov_from_a =
      GetCovarianceForRelativeRigid3d(a_from_world, b_from_world, covar);

  // Use the equations from the right convention as a reference.
  // The covariance in left (right) equals to the covariance of pose inverse in
  // right (left).

  // Convert to right convention. To estimate covariance of T_2T_1^{-1} in left,
  // We can equivalently estimate covariance of T_1T_2^{-1} in right.
  Eigen::Matrix<double, 12, 12> J0;
  J0.setZero();
  // the covariance of T_1^{-1} in left corresponds to the covariance of T_1 in
  // right
  J0.block<6, 6>(0, 0) = -a_from_world.AdjointInverse();
  // the covariance of T_2 in left corresponds to the covariance of T_2^{-1} in
  // right
  J0.block<6, 6>(6, 6) = Eigen::Matrix6d::Identity();
  // Get the covariance of (T_1, T_2^{-1}) in right
  const Eigen::Matrix<double, 12, 12> covar_in_right =
      J0 * covar * J0.transpose();

  // Compose T_1T_2^{-1} in right
  // [Reference] Joan Solà, Jeremie Deray, Dinesh Atchuthan, A micro Lie theory
  // for state estimation in robotics, 2018.
  // Eqs. (177) and (178)
  Eigen::Matrix<double, 6, 12> J_in_right;
  J_in_right.block<6, 6>(0, 0) = b_from_world.Adjoint();
  J_in_right.block<6, 6>(0, 6) = Eigen::Matrix6d::Identity();
  const Eigen::Matrix6d a_cov_from_b_right =
      J_in_right * covar_in_right * J_in_right.transpose();
  EXPECT_THAT(b_cov_from_a, EigenMatrixNear(a_cov_from_b_right, 1e-6));
}

TEST(Rigid3d, CovariancePropagation_Composed_vs_Relative) {
  const Rigid3d a_from_b = TestRigid3d();
  const Rigid3d b_from_c = TestRigid3d();
  const Eigen::Matrix<double, 12, 12> A =
      Eigen::Matrix<double, 12, 12>::Random();
  const Eigen::Matrix<double, 12, 12> covar = A * A.transpose();

  // Covariance for the composed rigid3d
  const Eigen::Matrix6d a_cov_from_c_composed =
      GetCovarianceForComposedRigid3d(a_from_b, covar);

  // Invert b_from_c and switch order
  const Rigid3d c_from_b = Inverse(b_from_c);
  Eigen::Matrix<double, 12, 12> J0;
  J0.setZero();
  J0.block<6, 6>(6, 0) = Eigen::Matrix6d::Identity();
  J0.block<6, 6>(0, 6) = -b_from_c.AdjointInverse();
  const Eigen::Matrix<double, 12, 12> covar_x_from_b =
      J0 * covar * J0.transpose();
  const Eigen::Matrix6d a_cov_from_c_relative =
      GetCovarianceForRelativeRigid3d(c_from_b, a_from_b, covar_x_from_b);

  // Check consistency
  EXPECT_THAT(a_cov_from_c_composed,
              EigenMatrixNear(a_cov_from_c_relative, 1e-6));
}

}  // namespace
}  // namespace colmap

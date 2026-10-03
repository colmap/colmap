// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/cost_functions/quaternion_utils.h"

#include "colmap/math/random_eigen.h"
#include "colmap/util/eigen_matchers.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(QuaternionLeftMultMatrix, Nominal) {
  constexpr double kEps = 1e-7;

  for (int i = 0; i < 100; ++i) {
    Eigen::Quaterniond q = RandomEigenQuaterniond();
    Eigen::Quaterniond p = RandomEigenQuaterniond();
    Eigen::Vector4d p_vec(p.x(), p.y(), p.z(), p.w());

    // L(q) * p = q * p.
    Eigen::Vector4d result = QuaternionLeftMultMatrix(q) * p_vec;
    Eigen::Quaterniond qp = q * p;
    Eigen::Vector4d expected(qp.x(), qp.y(), qp.z(), qp.w());
    EXPECT_THAT(result, EigenMatrixNear(expected, 1e-12));

    // L(q) = d(q*p)/dp (Jacobian w.r.t. second argument).
    Eigen::Matrix4d J_numeric;
    for (int k = 0; k < 4; ++k) {
      Eigen::Vector4d p_plus = p_vec, p_minus = p_vec;
      p_plus(k) += kEps;
      p_minus(k) -= kEps;
      Eigen::Quaterniond pp(p_plus(3), p_plus(0), p_plus(1), p_plus(2));
      Eigen::Quaterniond pm(p_minus(3), p_minus(0), p_minus(1), p_minus(2));
      Eigen::Quaterniond rp = q * pp, rm = q * pm;
      J_numeric.col(k) = (Eigen::Vector4d(rp.x(), rp.y(), rp.z(), rp.w()) -
                          Eigen::Vector4d(rm.x(), rm.y(), rm.z(), rm.w())) /
                         (2.0 * kEps);
    }
    EXPECT_THAT(QuaternionLeftMultMatrix(q), EigenMatrixNear(J_numeric, 1e-5));
  }
}

TEST(QuaternionRightMultMatrix, Nominal) {
  constexpr double kEps = 1e-7;

  for (int i = 0; i < 100; ++i) {
    Eigen::Quaterniond q = RandomEigenQuaterniond();
    Eigen::Quaterniond p = RandomEigenQuaterniond();
    Eigen::Vector4d q_vec(q.x(), q.y(), q.z(), q.w());

    // R(p) * q = q * p.
    Eigen::Vector4d result = QuaternionRightMultMatrix(p) * q_vec;
    Eigen::Quaterniond qp = q * p;
    Eigen::Vector4d expected(qp.x(), qp.y(), qp.z(), qp.w());
    EXPECT_THAT(result, EigenMatrixNear(expected, 1e-12));

    // R(p) = d(q*p)/dq (Jacobian w.r.t. first argument).
    Eigen::Matrix4d J_numeric;
    for (int k = 0; k < 4; ++k) {
      Eigen::Vector4d q_plus = q_vec, q_minus = q_vec;
      q_plus(k) += kEps;
      q_minus(k) -= kEps;
      Eigen::Quaterniond qp_p(q_plus(3), q_plus(0), q_plus(1), q_plus(2));
      Eigen::Quaterniond qm_p(q_minus(3), q_minus(0), q_minus(1), q_minus(2));
      Eigen::Quaterniond rp = qp_p * p, rm = qm_p * p;
      J_numeric.col(k) = (Eigen::Vector4d(rp.x(), rp.y(), rp.z(), rp.w()) -
                          Eigen::Vector4d(rm.x(), rm.y(), rm.z(), rm.w())) /
                         (2.0 * kEps);
    }
    EXPECT_THAT(QuaternionRightMultMatrix(p), EigenMatrixNear(J_numeric, 1e-5));
  }
}

TEST(QuaternionRotatePointWithJac, Nominal) {
  constexpr double kEps = 1e-7;

  for (int i = 0; i < 100; ++i) {
    Eigen::Quaterniond q = RandomEigenQuaterniond();
    Eigen::Vector3d pt = RandomEigenVectord<3>();
    double q_arr[4] = {q.x(), q.y(), q.z(), q.w()};

    // R(q) * pt matches Eigen.
    Eigen::Matrix<double, 3, 4, Eigen::RowMajor> J_analytical;
    Eigen::Vector3d result =
        QuaternionRotatePointWithJac(q_arr, pt.data(), J_analytical.data());
    EXPECT_THAT(result, EigenMatrixNear(Eigen::Vector3d(q * pt), 1e-12));

    // Jacobian d(R(q)*pt)/dq matches numeric.
    Eigen::Matrix<double, 3, 4> J_numeric;
    for (int k = 0; k < 4; ++k) {
      double q_plus[4] = {q_arr[0], q_arr[1], q_arr[2], q_arr[3]};
      double q_minus[4] = {q_arr[0], q_arr[1], q_arr[2], q_arr[3]};
      q_plus[k] += kEps;
      q_minus[k] -= kEps;
      J_numeric.col(k) =
          (QuaternionRotatePointWithJac(q_plus, pt.data(), nullptr) -
           QuaternionRotatePointWithJac(q_minus, pt.data(), nullptr)) /
          (2.0 * kEps);
    }
    EXPECT_THAT(J_analytical, EigenMatrixNear(J_numeric, 1e-5));
  }
}

TEST(EigenQuaternionAngleAxis, Roundtrip) {
  for (int i = 0; i < 100; ++i) {
    const Eigen::Quaterniond q = RandomEigenQuaterniond();
    const double q_arr[4] = {q.x(), q.y(), q.z(), q.w()};

    // quaternion -> angle-axis -> quaternion recovers the original rotation.
    double angle_axis[3];
    AngleAxisFromEigenQuaternion(q_arr, angle_axis);
    double q_out[4];
    EigenQuaternionFromAngleAxis(angle_axis, q_out);

    // Compare as rotations to avoid the quaternion double-cover sign ambiguity.
    const Eigen::Quaterniond q_recovered(
        q_out[3], q_out[0], q_out[1], q_out[2]);
    EXPECT_NEAR(q.angularDistance(q_recovered), 0.0, 1e-10);
  }
}

}  // namespace
}  // namespace colmap

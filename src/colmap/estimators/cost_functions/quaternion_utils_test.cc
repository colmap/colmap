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

#include "colmap/estimators/cost_functions/quaternion_utils.h"

#include "colmap/math/random.h"
#include "colmap/util/eigen_matchers.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(QuaternionLeftMultMatrix, Nominal) {
  SetPRNGSeed(42);
  const double eps = 1e-7;

  for (int i = 0; i < 100; ++i) {
    Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom();
    Eigen::Quaterniond p = Eigen::Quaterniond::UnitRandom();
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
      p_plus(k) += eps;
      p_minus(k) -= eps;
      Eigen::Quaterniond pp(p_plus(3), p_plus(0), p_plus(1), p_plus(2));
      Eigen::Quaterniond pm(p_minus(3), p_minus(0), p_minus(1), p_minus(2));
      Eigen::Quaterniond rp = q * pp, rm = q * pm;
      J_numeric.col(k) =
          (Eigen::Vector4d(rp.x(), rp.y(), rp.z(), rp.w()) -
           Eigen::Vector4d(rm.x(), rm.y(), rm.z(), rm.w())) /
          (2.0 * eps);
    }
    EXPECT_THAT(QuaternionLeftMultMatrix(q), EigenMatrixNear(J_numeric, 1e-5));
  }
}

TEST(QuaternionRightMultMatrix, Nominal) {
  SetPRNGSeed(42);
  const double eps = 1e-7;

  for (int i = 0; i < 100; ++i) {
    Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom();
    Eigen::Quaterniond p = Eigen::Quaterniond::UnitRandom();
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
      q_plus(k) += eps;
      q_minus(k) -= eps;
      Eigen::Quaterniond qp_p(q_plus(3), q_plus(0), q_plus(1), q_plus(2));
      Eigen::Quaterniond qm_p(q_minus(3), q_minus(0), q_minus(1), q_minus(2));
      Eigen::Quaterniond rp = qp_p * p, rm = qm_p * p;
      J_numeric.col(k) =
          (Eigen::Vector4d(rp.x(), rp.y(), rp.z(), rp.w()) -
           Eigen::Vector4d(rm.x(), rm.y(), rm.z(), rm.w())) /
          (2.0 * eps);
    }
    EXPECT_THAT(QuaternionRightMultMatrix(p), EigenMatrixNear(J_numeric, 1e-5));
  }
}

TEST(QuaternionRotatePointWithJac, Nominal) {
  SetPRNGSeed(42);
  const double eps = 1e-7;

  for (int i = 0; i < 100; ++i) {
    Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom();
    Eigen::Vector3d pt = Eigen::Vector3d::Random();
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
      q_plus[k] += eps;
      q_minus[k] -= eps;
      J_numeric.col(k) =
          (QuaternionRotatePointWithJac(q_plus, pt.data(), nullptr) -
           QuaternionRotatePointWithJac(q_minus, pt.data(), nullptr)) /
          (2.0 * eps);
    }
    EXPECT_THAT(J_analytical, EigenMatrixNear(J_numeric, 1e-5));
  }
}

}  // namespace
}  // namespace colmap

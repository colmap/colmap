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

#pragma once

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <ostream>

#include <Eigen/Geometry>

namespace colmap {

// Compose the skew symmetric cross product matrix from a vector.
Eigen::Matrix3d CrossProductMatrix(const Eigen::Vector3d& vector);

// 3D rigid transform with 6 degrees of freedom.
// Transforms point x from a to b as: x_in_b = R * x_in_a + t.
struct Rigid3d {
 public:
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();

  Rigid3d() = default;
  Rigid3d(const Eigen::Quaterniond& rotation,
          const Eigen::Vector3d& translation)
      : rotation(rotation), translation(translation) {}

  inline Eigen::Matrix3x4d ToMatrix() const {
    Eigen::Matrix3x4d matrix;
    matrix.leftCols<3>() = rotation.toRotationMatrix();
    matrix.col(3) = translation;
    return matrix;
  }

  static inline Rigid3d FromMatrix(const Eigen::Matrix3x4d& matrix) {
    Rigid3d t;
    t.rotation = Eigen::Quaterniond(matrix.leftCols<3>()).normalized();
    t.translation = matrix.rightCols<1>();
    return t;
  }

  // Adjoint matrix to propagate uncertainty on Rigid3d
  // [Reference] https://gtsam.org/2021/02/23/uncertainties-part3.html
  inline Eigen::Matrix6d Adjoint() const {
    Eigen::Matrix6d adjoint;
    adjoint.block<3, 3>(0, 0) = rotation.toRotationMatrix();
    adjoint.block<3, 3>(0, 3).setZero();
    adjoint.block<3, 3>(3, 0) =
        adjoint.block<3, 3>(0, 0).colwise().cross(-translation);  // t x R
    adjoint.block<3, 3>(3, 3) = adjoint.block<3, 3>(0, 0);
    return adjoint;
  }

  inline Eigen::Matrix6d AdjointInverse() const {
    Eigen::Matrix6d adjoint_inv;
    adjoint_inv.block<3, 3>(0, 0) = rotation.toRotationMatrix().transpose();
    adjoint_inv.block<3, 3>(0, 3).setZero();
    adjoint_inv.block<3, 3>(3, 0) =
        -adjoint_inv.block<3, 3>(0, 0) * CrossProductMatrix(translation);
    adjoint_inv.block<3, 3>(3, 3) = adjoint_inv.block<3, 3>(0, 0);
    return adjoint_inv;
  }
};

// Return inverse transform.
inline Rigid3d Inverse(const Rigid3d& b_from_a) {
  Rigid3d a_from_b;
  a_from_b.rotation = b_from_a.rotation.inverse();
  a_from_b.translation = a_from_b.rotation * -b_from_a.translation;
  return a_from_b;
}

// Update covariance (6 x 6) for rigid3d.inverse()
//
// [Reference] Joan Solà, Jeremie Deray, Dinesh Atchuthan, A micro Lie theory
// for state estimation in robotics, 2018.
// In COLMAP we follow the left convention (rather than the right convention as
// in the reference paper and GTSAM). With the left convention the Jacobian of
// the inverse is -Ad(X^{-1}). This can be easily derived combining Eqs. (62)
// and (57) in the paper.
inline Eigen::Matrix6d GetCovarianceForRigid3dInverse(
    const Rigid3d& rigid3, const Eigen::Matrix6d& covar) {
  const Eigen::Matrix6d adjoint_inv = rigid3.AdjointInverse();
  return adjoint_inv * covar * adjoint_inv.transpose();
}

// Given a (12 x 12) covariance on two rigid3d objects (a_from_b, b_from_c),
// this function calculates the (6 x 6) covariance of the composed
// transformation a_from_c (a_T_b * b_T_c). b_T_c does not contribute to the
// covariance propagation and is thus not required.
inline Eigen::Matrix6d GetCovarianceForComposedRigid3d(
    const Rigid3d& a_from_b, const Eigen::Matrix<double, 12, 12>& covar) {
  Eigen::Matrix<double, 6, 12> J;
  J.block<6, 6>(0, 0) = Eigen::Matrix6d::Identity();
  J.block<6, 6>(0, 6) = a_from_b.Adjoint();
  return J * covar * J.transpose();
}

// Given a (12 x 12) covariance on two rigid3d objects (a_from_c, b_from_c),
// this function calculates the (6 x 6) covariance of the relative
// transformation b_from_a (b_T_c * a_T_c^{-1}).
inline Eigen::Matrix6d GetCovarianceForRelativeRigid3d(
    const Rigid3d& a_from_c,
    const Rigid3d& b_from_c,
    const Eigen::Matrix<double, 12, 12>& covar) {
  Eigen::Matrix<double, 6, 12> J;
  J.block<6, 6>(0, 0) = -b_from_c.Adjoint() * a_from_c.AdjointInverse();
  J.block<6, 6>(0, 6) = Eigen::Matrix6d::Identity();
  return J * covar * J.transpose();
}

// Apply transform to point such that one can write expressions like:
//      x_in_b = b_from_a * x_in_a
//
// Be careful when including multiple transformations in the same expression, as
// the multiply operator in C++ is evaluated left-to-right.
// For example, the following expression:
//      x_in_c = d_from_c * c_from_b * b_from_a * x_in_a
// will be executed in the following order:
//      x_in_c = ((d_from_c * c_from_b) * b_from_a) * x_in_a
// This will first concatenate all transforms and then apply it to the point.
// While you may want to instead write and execute it as:
//      x_in_c = d_from_c * (c_from_b * (b_from_a * x_in_a))
// which will apply the transformations as a chain on the point.
inline Eigen::Vector3d operator*(const Rigid3d& t, const Eigen::Vector3d& x) {
  return t.rotation * x + t.translation;
}

// Concatenate transforms such one can write expressions like:
//      d_from_a = d_from_c * c_from_b * b_from_a
inline Rigid3d operator*(const Rigid3d& c_from_b, const Rigid3d& b_from_a) {
  Rigid3d c_from_a;
  c_from_a.rotation = (c_from_b.rotation * b_from_a.rotation).normalized();
  c_from_a.translation =
      c_from_b.translation + (c_from_b.rotation * b_from_a.translation);
  return c_from_a;
}

inline bool operator==(const Rigid3d& left, const Rigid3d& right) {
  return left.rotation.coeffs() == right.rotation.coeffs() &&
         left.translation == right.translation;
}
inline bool operator!=(const Rigid3d& left, const Rigid3d& right) {
  return !(left == right);
}

std::ostream& operator<<(std::ostream& stream, const Rigid3d& tform);

}  // namespace colmap

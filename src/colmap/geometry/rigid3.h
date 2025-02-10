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

#pragma once

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <ostream>

#include <Eigen/Geometry>

namespace colmap {

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
};

// Return inverse transform.
inline Rigid3d Inverse(const Rigid3d& b_from_a) {
  Rigid3d a_from_b;
  a_from_b.rotation = b_from_a.rotation.inverse();
  a_from_b.translation = a_from_b.rotation * -b_from_a.translation;
  return a_from_b;
}

// Update covariance (6x6) for rigid3d.inverse()
inline Eigen::Matrix6d GetCovarianceForRigid3dInverse(
    const Rigid3d& rigid3, const Eigen::Matrix6d& covar) {
  const Eigen::Matrix6d adjoint = rigid3.Adjoint();
  return adjoint * covar * adjoint.transpose();
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

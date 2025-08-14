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

#include "colmap/geometry/rigid3.h"
#include "colmap/math/matrix.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colmap {

// 3D similarity transform with 7 degrees of freedom.
// Transforms point x from a to b as: x_in_b = scale * R * x_in_a + t.
struct Sim3d {
  double scale = 1;
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();

  Sim3d() = default;
  Sim3d(double scale,
        const Eigen::Quaterniond& rotation,
        const Eigen::Vector3d& translation)
      : scale(scale), rotation(rotation), translation(translation) {}

  inline Eigen::Matrix3x4d ToMatrix() const;

  static inline Sim3d FromMatrix(const Eigen::Matrix3x4d& matrix);

  // Adjoint matrix to propagate uncertainty on Sim3d
  //
  // [Reference] Ethan Eade, "Lie Groups for 2D and 3D Transformations"
  //             https://www.ethaneade.org/lie.pdf
  // Parameter ordering here is [\omega; v; \sigma], i.e. rotation vector
  // \omega first, then translation v, then scale \sigma, which is different
  // from the ordering in the reference.
  inline Eigen::Matrix<double, 7, 7> Adjoint() const;
  inline Eigen::Matrix<double, 7, 7> AdjointInverse() const;

  // Read from or write to text file without loss of precision.
  void ToFile(const std::string& path) const;
  static Sim3d FromFile(const std::string& path);
};

// Return inverse transform.
inline Sim3d Inverse(const Sim3d& b_from_a);

// Update covariance (7 x 7) for Inverse(sim3d)
inline Eigen::Matrix<double, 7, 7> PropagateCovarianceForInverse(
    const Sim3d& sim3, const Eigen::Matrix<double, 7, 7>& covar);

// Given a (14 x 14) covariance on two sim3d objects (a_from_b, b_from_c),
// this function calculates the (7 x 7) covariance of the composed
// transformation a_from_c (a_T_b * b_T_c).
inline Eigen::Matrix<double, 7, 7> PropagateCovarianceForCompose(
    const Sim3d& a_from_b, const Eigen::Matrix<double, 14, 14>& covar);

// Given a (14 x 14) covariance on two rigid3d objects (a_from_c, b_from_c),
// this function calculates the (7 x 7) covariance of the relative
// transformation b_from_a (b_T_c * a_T_c^{-1}).
inline Eigen::Matrix<double, 7, 7> PropagateCovarianceForRelative(
    const Sim3d& a_from_c,
    const Sim3d& b_from_c,
    const Eigen::Matrix<double, 14, 14>& covar);

// Given a point p and its 3x3 covariance, this function propagates the
// covariance through a Sim3 transformation, it computes the covariance of the
// transformed point: p_transformed = sim3 * p
inline Eigen::Matrix3d PropagateCovarianceForTransformPoint(
    const Sim3d& sim3, const Eigen::Matrix3d& covar);

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
inline Eigen::Vector3d operator*(const Sim3d& t, const Eigen::Vector3d& x);

// Concatenate transforms such one can write expressions like:
//      d_from_a = d_from_c * c_from_b * b_from_a
inline Sim3d operator*(const Sim3d& c_from_b, const Sim3d& b_from_a);

inline bool operator==(const Sim3d& left, const Sim3d& right);
inline bool operator!=(const Sim3d& left, const Sim3d& right);

std::ostream& operator<<(std::ostream& stream, const Sim3d& tform);

// Convert Sim3d to Rigid3d by dropping the similarity scale.
// Rotation is unchanged; if scale_translation = true, scale the translation.
Rigid3d Sim3dToRigid3d(const Sim3d& sim3, bool scale_translation = false);

// Convert to Sim3d with unit scale. If scale_translation = true,
// divide the translation vector by the given scale.
Sim3d Rigid3dToSim3d(const Rigid3d& rigid3,
                     bool scale_translation = false,
                     double scale = 1.0);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

Eigen::Matrix3x4d Sim3d::ToMatrix() const {
  Eigen::Matrix3x4d matrix;
  matrix.leftCols<3>() = scale * rotation.toRotationMatrix();
  matrix.col(3) = translation;
  return matrix;
}

Sim3d Sim3d::FromMatrix(const Eigen::Matrix3x4d& matrix) {
  Sim3d t;
  t.scale = matrix.col(0).norm();
  t.rotation = Eigen::Quaterniond(matrix.leftCols<3>() / t.scale).normalized();
  t.translation = matrix.rightCols<1>();
  return t;
}

Eigen::Matrix<double, 7, 7> Sim3d::Adjoint() const {
  Eigen::Matrix<double, 7, 7> adjoint = Eigen::Matrix<double, 7, 7>::Zero();
  const Eigen::Matrix3d R = rotation.toRotationMatrix();
  adjoint.block<3, 3>(0, 0) = R;
  adjoint.block<3, 3>(3, 0) = CrossProductMatrix(translation) * R;
  adjoint.block<3, 3>(3, 3) = scale * R;
  adjoint.block<3, 1>(3, 6) = -translation;
  adjoint(6, 6) = 1.0;
  return adjoint;
}

Eigen::Matrix<double, 7, 7> Sim3d::AdjointInverse() const {
  Eigen::Matrix<double, 7, 7> adjoint_inv = Eigen::Matrix<double, 7, 7>::Zero();
  const double inv_s = 1.0 / scale;
  const Eigen::Matrix3d R = rotation.toRotationMatrix();
  const Eigen::Matrix3d RT = R.transpose();
  adjoint_inv.block<3, 3>(0, 0) = RT;
  adjoint_inv.block<3, 3>(3, 0) = -inv_s * RT * CrossProductMatrix(translation);
  adjoint_inv.block<3, 3>(3, 3) = inv_s * RT;
  adjoint_inv.block<3, 1>(3, 6) = inv_s * RT * translation;
  adjoint_inv(6, 6) = 1.0;
  return adjoint_inv;
}

Sim3d Inverse(const Sim3d& b_from_a) {
  Sim3d a_from_b;
  a_from_b.scale = 1 / b_from_a.scale;
  a_from_b.rotation = b_from_a.rotation.inverse();
  a_from_b.translation =
      (a_from_b.rotation * b_from_a.translation) / -b_from_a.scale;
  return a_from_b;
}

Eigen::Matrix<double, 7, 7> PropagateCovarianceForInverse(
    const Sim3d& sim3, const Eigen::Matrix<double, 7, 7>& covar) {
  const Eigen::Matrix<double, 7, 7> adjoint_inv = sim3.AdjointInverse();
  return adjoint_inv * covar * adjoint_inv.transpose();
}

Eigen::Matrix<double, 7, 7> PropagateCovarianceForCompose(
    const Sim3d& a_from_b, const Eigen::Matrix<double, 14, 14>& covar) {
  Eigen::Matrix<double, 7, 14> J;
  J.block<7, 7>(0, 0) = Eigen::Matrix<double, 7, 7>::Identity();
  J.block<7, 7>(0, 7) = a_from_b.Adjoint();
  return J * covar * J.transpose();
}

Eigen::Matrix<double, 7, 7> PropagateCovarianceForRelative(
    const Sim3d& a_from_c,
    const Sim3d& b_from_c,
    const Eigen::Matrix<double, 14, 14>& covar) {
  Eigen::Matrix<double, 7, 14> J;
  J.block<7, 7>(0, 0) = -b_from_c.Adjoint() * a_from_c.AdjointInverse();
  J.block<7, 7>(0, 7) = Eigen::Matrix<double, 7, 7>::Identity();
  return J * covar * J.transpose();
}

Eigen::Matrix3d PropagateCovarianceForTransformPoint(
    const Sim3d& sim3, const Eigen::Matrix3d& covar) {
  Eigen::Matrix3d J = sim3.Adjoint().block<3, 3>(3, 3);
  return J * covar * J.transpose();
}

Eigen::Vector3d operator*(const Sim3d& t, const Eigen::Vector3d& x) {
  return t.scale * (t.rotation * x) + t.translation;
}

Sim3d operator*(const Sim3d& c_from_b, const Sim3d& b_from_a) {
  Sim3d c_from_a;
  c_from_a.scale = c_from_b.scale * b_from_a.scale;
  c_from_a.rotation = (c_from_b.rotation * b_from_a.rotation).normalized();
  c_from_a.translation =
      c_from_b.translation +
      (c_from_b.scale * (c_from_b.rotation * b_from_a.translation));
  return c_from_a;
}

bool operator==(const Sim3d& left, const Sim3d& right) {
  return left.scale == right.scale &&
         left.rotation.coeffs() == right.rotation.coeffs() &&
         left.translation == right.translation;
}

bool operator!=(const Sim3d& left, const Sim3d& right) {
  return !(left == right);
}
}  // namespace colmap

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

  inline Eigen::Matrix3x4d ToMatrix() const {
    Eigen::Matrix3x4d matrix;
    matrix.leftCols<3>() = scale * rotation.toRotationMatrix();
    matrix.col(3) = translation;
    return matrix;
  }

  static inline Sim3d FromMatrix(const Eigen::Matrix3x4d& matrix) {
    Sim3d t;
    t.scale = matrix.col(0).norm();
    t.rotation =
        Eigen::Quaterniond(matrix.leftCols<3>() / t.scale).normalized();
    t.translation = matrix.rightCols<1>();
    return t;
  }

  // Read from or write to text file without loss of precision.
  void ToFile(const std::string& path) const;
  static Sim3d FromFile(const std::string& path);
};

// Return inverse transform.
inline Sim3d Inverse(const Sim3d& b_from_a) {
  Sim3d a_from_b;
  a_from_b.scale = 1 / b_from_a.scale;
  a_from_b.rotation = b_from_a.rotation.inverse();
  a_from_b.translation =
      (a_from_b.rotation * b_from_a.translation) / -b_from_a.scale;
  return a_from_b;
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
inline Eigen::Vector3d operator*(const Sim3d& t, const Eigen::Vector3d& x) {
  return t.scale * (t.rotation * x) + t.translation;
}

// Concatenate transforms such one can write expressions like:
//      d_from_a = d_from_c * c_from_b * b_from_a
inline Sim3d operator*(const Sim3d& c_from_b, const Sim3d& b_from_a) {
  Sim3d c_from_a;
  c_from_a.scale = c_from_b.scale * b_from_a.scale;
  c_from_a.rotation = (c_from_b.rotation * b_from_a.rotation).normalized();
  c_from_a.translation =
      c_from_b.translation +
      (c_from_b.scale * (c_from_b.rotation * b_from_a.translation));
  return c_from_a;
}

inline bool operator==(const Sim3d& left, const Sim3d& right) {
  return left.scale == right.scale &&
         left.rotation.coeffs() == right.rotation.coeffs() &&
         left.translation == right.translation;
}
inline bool operator!=(const Sim3d& left, const Sim3d& right) {
  return !(left == right);
}

std::ostream& operator<<(std::ostream& stream, const Sim3d& tform);

}  // namespace colmap

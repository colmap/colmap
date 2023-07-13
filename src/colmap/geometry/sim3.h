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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#pragma once

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

  inline Sim3d Inverse() const {
    Sim3d inverse;
    inverse.scale = 1 / scale;
    inverse.rotation = rotation.inverse();
    inverse.translation = (inverse.rotation * translation) / -scale;
    return inverse;
  }

  inline Eigen::Matrix3x4d Matrix() const {
    Eigen::Matrix3x4d matrix;
    matrix.leftCols<3>() = scale * rotation.toRotationMatrix();
    matrix.col(3) = translation;
    return matrix;
  }

  // Estimate tgtFromSrc transform. Return true if successful.
  bool Estimate(const std::vector<Eigen::Vector3d>& src,
                const std::vector<Eigen::Vector3d>& tgt);

  // Transform world for camFromWorld pose.
  // TODO(jsch): Rename and refactor with future RigidTransform class.
  void TransformPose(Eigen::Vector4d* qvec, Eigen::Vector3d* tvec) const;

  // Read from or write to text file without loss of precision.
  void ToFile(const std::string& path) const;
  static Sim3d FromFile(const std::string& path);
};

// Apply transform to point such that one can write expressions like:
//      x_in_b = bFromA * x_in_a
inline Eigen::Vector3d operator*(const Sim3d& t, const Eigen::Vector3d& x) {
  return t.scale * (t.rotation * x) + t.translation;
}

// Concatenate transforms such one can write expressions like:
//      dFromA = dFromC * cFromB * bFromA
inline Sim3d operator*(const Sim3d& cFromB, const Sim3d& bFromA) {
  Sim3d cFromA;
  cFromA.scale = cFromB.scale * bFromA.scale;
  cFromA.rotation = (cFromB.rotation * bFromA.rotation).normalized();
  cFromA.translation = cFromB.translation +
                       (cFromB.scale * (cFromB.rotation * bFromA.translation));
  return cFromA;
}

}  // namespace colmap

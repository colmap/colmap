// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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

#include "retrieval/geometry.h"

namespace colmap {
namespace retrieval {

FeatureGeometryTransform FeatureGeometry::TransformFromMatch(
    const FeatureGeometry& feature1, const FeatureGeometry& feature2) {
  FeatureGeometryTransform tform;

  tform.scale = feature2.scale / feature1.scale;
  tform.angle = feature2.orientation - feature1.orientation;

  const float sin_angle = std::sin(tform.angle);
  const float cos_angle = std::cos(tform.angle);

  Eigen::Matrix2f R;
  R << cos_angle, -sin_angle, sin_angle, cos_angle;

  const Eigen::Vector2f t =
      Eigen::Vector2f(feature2.x, feature2.y) -
      tform.scale * R * Eigen::Vector2f(feature1.x, feature1.y);
  tform.tx = t.x();
  tform.ty = t.y();

  return tform;
}

Eigen::Matrix<float, 2, 3> FeatureGeometry::TransformMatrixFromMatch(
    const FeatureGeometry& feature1, const FeatureGeometry& feature2) {
  Eigen::Matrix<float, 2, 3> T;

  const float scale = feature2.scale / feature1.scale;
  const float angle = feature2.orientation - feature1.orientation;

  const float sin_angle = std::sin(angle);
  const float cos_angle = std::cos(angle);

  T.leftCols<2>() << cos_angle, -sin_angle, sin_angle, cos_angle;
  T.leftCols<2>() *= scale;

  T.rightCols<1>() =
      Eigen::Vector2f(feature2.x, feature2.y) -
      T.leftCols<2>() * Eigen::Vector2f(feature1.x, feature1.y);

  return T;
}

float FeatureGeometry::GetArea() const {
  return 1.0f / std::sqrt(4.0f / (scale * scale * scale * scale));
}

float FeatureGeometry::GetAreaUnderTransform(const Eigen::Matrix2f& A) const {
  const Eigen::Matrix2f M = Eigen::Matrix2f::Identity() / (scale * scale);
  const Eigen::Matrix2f N = A.transpose() * M * A;
  const float B = N(1, 0) + N(0, 1);
  return 1.0f / std::sqrt(4.0f * N(0, 0) * N(1, 1) - B * B);
}

}  // namespace retrieval
}  // namespace colmap

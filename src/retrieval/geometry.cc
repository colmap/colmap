// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
      scale * T.leftCols<2>() * Eigen::Vector2f(feature1.x, feature1.y);

  return T;
}

float FeatureGeometry::GetArea() const {
  return 1.0f / std::sqrt(4.0f / (scale * scale * scale * scale));
}

float FeatureGeometry::GetAreaUnderTransform(const Eigen::Matrix2f A) const {
  const Eigen::Matrix2f M = Eigen::Matrix2f::Identity() / (scale * scale);
  const Eigen::Matrix2f N = A.transpose() * M * A;
  const float B = N(1, 0) + N(0, 1);
  return 1.0f / std::sqrt(4.0f * N(0, 0) * N(1, 1) - B * B);
}

}  // namespace retrieval
}  // namespace colmap

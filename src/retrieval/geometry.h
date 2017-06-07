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

#ifndef COLMAP_SRC_RETRIEVAL_GEOMETRY_H_
#define COLMAP_SRC_RETRIEVAL_GEOMETRY_H_

#include <vector>

#include <Eigen/Core>

namespace colmap {
namespace retrieval {

struct FeatureGeometryTransform {
  float scale = 0.0f;
  float angle = 0.0f;
  float tx = 0.0f;
  float ty = 0.0f;
};

struct FeatureGeometry {
  // Compute the similarity that transforms the shape of feature 1 to feature 2.
  static FeatureGeometryTransform TransformFromMatch(
      const FeatureGeometry& feature1, const FeatureGeometry& feature2);
  static Eigen::Matrix<float, 2, 3> TransformMatrixFromMatch(
      const FeatureGeometry& feature1, const FeatureGeometry& feature2);

  // Get the approximate area occupied by the feature.
  float GetArea() const;

  // Get the approximate area occupied by the feature after applying an affine
  // transformation to the feature geometry.
  float GetAreaUnderTransform(const Eigen::Matrix2f A) const;

  float x = 0.0f;
  float y = 0.0f;
  float scale = 0.0f;
  float orientation = 0.0f;
};

// 1-to-M feature geometry match.
struct FeatureGeometryMatch {
  FeatureGeometry geometry1;
  std::vector<FeatureGeometry> geometries2;
};

}  // namespace retrieval
}  // namespace colmap

#endif  // COLMAP_SRC_RETRIEVAL_GEOMETRY_H_

// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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
  float GetAreaUnderTransform(const Eigen::Matrix2f& A) const;

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

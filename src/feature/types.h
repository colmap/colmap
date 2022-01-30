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

#ifndef COLMAP_SRC_FEATURE_TYPES_H_
#define COLMAP_SRC_FEATURE_TYPES_H_

#include <vector>

#include <Eigen/Core>

#include "util/types.h"

namespace colmap {

struct FeatureKeypoint {
  FeatureKeypoint();
  FeatureKeypoint(const float x, const float y);
  FeatureKeypoint(const float x, const float y, const float scale,
                  const float orientation);
  FeatureKeypoint(const float x, const float y, const float a11,
                  const float a12, const float a21, const float a22);

  static FeatureKeypoint FromParameters(const float x, const float y,
                                        const float scale_x,
                                        const float scale_y,
                                        const float orientation,
                                        const float shear);

  // Rescale the feature location and shape size by the given scale factor.
  void Rescale(const float scale);
  void Rescale(const float scale_x, const float scale_y);

  // Compute similarity shape parameters from affine shape.
  float ComputeScale() const;
  float ComputeScaleX() const;
  float ComputeScaleY() const;
  float ComputeOrientation() const;
  float ComputeShear() const;

  // Location of the feature, with the origin at the upper left image corner,
  // i.e. the upper left pixel has the coordinate (0.5, 0.5).
  float x;
  float y;

  // Affine shape of the feature.
  float a11;
  float a12;
  float a21;
  float a22;
};

typedef Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor>
    FeatureDescriptor;

struct FeatureMatch {
  FeatureMatch()
      : point2D_idx1(kInvalidPoint2DIdx), point2D_idx2(kInvalidPoint2DIdx) {}
  FeatureMatch(const point2D_t point2D_idx1, const point2D_t point2D_idx2)
      : point2D_idx1(point2D_idx1), point2D_idx2(point2D_idx2) {}

  // Feature index in first image.
  point2D_t point2D_idx1 = kInvalidPoint2DIdx;

  // Feature index in second image.
  point2D_t point2D_idx2 = kInvalidPoint2DIdx;
};

typedef std::vector<FeatureKeypoint> FeatureKeypoints;
typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    FeatureDescriptors;
typedef std::vector<FeatureMatch> FeatureMatches;

}  // namespace colmap

#endif  // COLMAP_SRC_FEATURE_TYPES_H_

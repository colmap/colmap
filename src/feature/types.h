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

#ifndef COLMAP_SRC_FEATURE_TYPES_H_
#define COLMAP_SRC_FEATURE_TYPES_H_

#include <vector>

#include <Eigen/Core>

#include "util/types.h"

namespace colmap {

struct FeatureKeypoint {
  // Location of the feature, with the origin at the upper left image corner,
  // i.e. the upper left pixel has the coordinate (0.5, 0.5).
  float x = 0.0f;
  float y = 0.0f;
  // Shape of the feature.
  float scale = 0.0f;
  float orientation = 0.0f;
};

struct FeatureMatch {
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

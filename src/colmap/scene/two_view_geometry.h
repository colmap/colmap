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

#include "colmap/feature/types.h"
#include "colmap/geometry/rigid3.h"

namespace colmap {

// Two-view geometry.
struct TwoViewGeometry {
  // The configuration of the two-view geometry.
  enum ConfigurationType {
    UNDEFINED = 0,
    // Degenerate configuration (e.g., no overlap or not enough inliers).
    DEGENERATE = 1,
    // Essential matrix.
    CALIBRATED = 2,
    // Fundamental matrix.
    UNCALIBRATED = 3,
    // Homography, planar scene with baseline.
    PLANAR = 4,
    // Homography, pure rotation without baseline.
    PANORAMIC = 5,
    // Homography, planar or panoramic.
    PLANAR_OR_PANORAMIC = 6,
    // Watermark, pure 2D translation in image borders.
    WATERMARK = 7,
    // Multi-model configuration, i.e. the inlier matches result from multiple
    // individual, non-degenerate configurations.
    MULTIPLE = 8,
  };

  // One of `ConfigurationType`.
  int config = ConfigurationType::UNDEFINED;

  // Essential matrix.
  Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
  // Fundamental matrix.
  Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
  // Homography matrix.
  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();

  // Relative pose.
  Rigid3d cam2_from_cam1;

  // Inlier matches of the configuration.
  FeatureMatches inlier_matches;

  // Median triangulation angle.
  double tri_angle = -1;

  // Invert the geometry to match swapped cameras.
  void Invert();
};

}  // namespace colmap

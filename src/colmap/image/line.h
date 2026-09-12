// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/sensor/bitmap.h"
#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>

namespace colmap {

struct LineSegment {
  Eigen::Vector2d start;
  Eigen::Vector2d end;
};

enum class LineSegmentOrientation {
  HORIZONTAL = 1,
  VERTICAL = -1,
  UNDEFINED = 0,
};

#ifdef COLMAP_LSD_ENABLED
// Detect line segments in the given bitmap image.
std::vector<LineSegment> DetectLineSegments(const Bitmap& bitmap,
                                            double min_length = 3);
#endif

// Classify line segments into horizontal/vertical.
std::vector<LineSegmentOrientation> ClassifyLineSegmentOrientations(
    const std::vector<LineSegment>& segments, double tolerance = 0.25);

}  // namespace colmap

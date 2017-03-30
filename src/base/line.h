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

#ifndef COLMAP_SRC_BASE_LINE_H_
#define COLMAP_SRC_BASE_LINE_H_

#include <Eigen/Core>

#include "util/alignment.h"
#include "util/bitmap.h"

namespace colmap {

struct LineSegment {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector2d start;
  Eigen::Vector2d end;
};

enum class LineSegmentOrientation {
  HORIZONTAL = 1,
  VERTICAL = -1,
  UNDEFINED = 0,
};

// Detect line segments in the given bitmap image.
std::vector<LineSegment> DetectLineSegments(const Bitmap& bitmap,
                                            const double min_length = 3);

// Classify line segments into horizontal/vertical.
std::vector<LineSegmentOrientation> ClassifyLineSegmentOrientations(
    const std::vector<LineSegment>& segments, const double tolerance = 0.25);

}  // namespace colmap

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(colmap::LineSegment)

#endif  // COLMAP_SRC_BASE_LINE_H_

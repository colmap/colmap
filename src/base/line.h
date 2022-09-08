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

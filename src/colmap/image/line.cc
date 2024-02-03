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

#include "colmap/image/line.h"

#include "colmap/util/logging.h"

extern "C" {
#include "thirdparty/LSD/lsd.h"
}

#include <memory>

namespace colmap {
namespace {

struct RawDeleter {
  void operator()(double* p) { free(p); }
};

}  // namespace

std::vector<LineSegment> DetectLineSegments(const Bitmap& bitmap,
                                            const double min_length) {
  const double min_length_squared = min_length * min_length;

  std::vector<uint8_t> bitmap_data;
  if (bitmap.IsGrey()) {
    bitmap_data = bitmap.ConvertToRowMajorArray();
  } else {
    const Bitmap bitmap_gray = bitmap.CloneAsGrey();
    bitmap_data = bitmap_gray.ConvertToRowMajorArray();
  }

  std::vector<double> bitmap_data_double(bitmap_data.begin(),
                                         bitmap_data.end());

  int num_segments;
  std::unique_ptr<double, RawDeleter> segments_data(
      lsd(&num_segments,
          bitmap_data_double.data(),
          bitmap.Width(),
          bitmap.Height()));

  std::vector<LineSegment> segments;
  segments.reserve(num_segments);
  for (int i = 0; i < num_segments; ++i) {
    const Eigen::Vector2d start(segments_data.get()[i * 7],
                                segments_data.get()[i * 7 + 1]);
    const Eigen::Vector2d end(segments_data.get()[i * 7 + 2],
                              segments_data.get()[i * 7 + 3]);
    if ((start - end).squaredNorm() >= min_length_squared) {
      segments.emplace_back();
      segments.back().start = start;
      segments.back().end = end;
    }
  }

  return segments;
}

std::vector<LineSegmentOrientation> ClassifyLineSegmentOrientations(
    const std::vector<LineSegment>& segments, const double tolerance) {
  THROW_CHECK_LE(tolerance, 0.5);

  std::vector<LineSegmentOrientation> orientations;
  orientations.reserve(segments.size());

  for (const auto& segment : segments) {
    const Eigen::Vector2d direction =
        (segment.end - segment.start).normalized();
    if (std::abs(direction.x()) + tolerance > 1) {
      orientations.push_back(LineSegmentOrientation::HORIZONTAL);
    } else if (std::abs(direction.y()) + tolerance > 1) {
      orientations.push_back(LineSegmentOrientation::VERTICAL);
    } else {
      orientations.push_back(LineSegmentOrientation::UNDEFINED);
    }
  }

  return orientations;
}

}  // namespace colmap

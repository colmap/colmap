// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/image/line.h"

#include "colmap/util/logging.h"

#ifdef COLMAP_LSD_ENABLED
extern "C" {
#include "thirdparty/LSD/lsd.h"
}
#endif

#include <memory>

namespace colmap {
namespace {

struct RawDeleter {
  void operator()(double* p) { free(p); }
};

}  // namespace

#ifdef COLMAP_LSD_ENABLED

std::vector<LineSegment> DetectLineSegments(const Bitmap& bitmap,
                                            const double min_length) {
  const double min_length_squared = min_length * min_length;

  std::vector<double> bitmap_data_double;
  if (bitmap.IsGrey()) {
    bitmap_data_double = {bitmap.RowMajorData().begin(),
                          bitmap.RowMajorData().end()};
  } else {
    const Bitmap bitmap_gray = bitmap.CloneAsGrey();
    bitmap_data_double = {bitmap_gray.RowMajorData().begin(),
                          bitmap_gray.RowMajorData().end()};
  }

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
#endif

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

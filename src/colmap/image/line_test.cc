// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/image/line.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

#ifdef COLMAP_LSD_ENABLED

TEST(DetectLineSegments, Nominal) {
  Bitmap bitmap(100, 100, false);
  for (size_t i = 0; i < 100; ++i) {
    bitmap.SetPixel(i, i, BitmapColor<uint8_t>(255));
  }

  const auto line_segments = DetectLineSegments(bitmap, 0);

  EXPECT_EQ(line_segments.size(), 2);

  const Eigen::Vector2d ref_start(0, 0);
  const Eigen::Vector2d ref_end(100, 100);
  EXPECT_LT((line_segments[0].start - ref_start).norm(), 5);
  EXPECT_LT((line_segments[0].end - ref_end).norm(), 5);
  EXPECT_LT((line_segments[1].start - ref_end).norm(), 5);
  EXPECT_LT((line_segments[1].end - ref_start).norm(), 5);

  EXPECT_EQ(DetectLineSegments(bitmap, 150).size(), 0);
}

TEST(ClassifyLineSegmentOrientations, Nominal) {
  Bitmap bitmap(100, 100, false);
  for (size_t i = 60; i < 100; ++i) {
    bitmap.SetPixel(i, 50, BitmapColor<uint8_t>(255));
    bitmap.SetPixel(50, i, BitmapColor<uint8_t>(255));
    bitmap.SetPixel(i, i, BitmapColor<uint8_t>(255));
  }

  const auto line_segments = DetectLineSegments(bitmap, 0);
  EXPECT_EQ(line_segments.size(), 6);

  const auto orientations = ClassifyLineSegmentOrientations(line_segments);
  EXPECT_EQ(orientations.size(), 6);

  EXPECT_TRUE(orientations[0] == LineSegmentOrientation::VERTICAL);
  EXPECT_TRUE(orientations[1] == LineSegmentOrientation::VERTICAL);
  EXPECT_TRUE(orientations[2] == LineSegmentOrientation::HORIZONTAL);
  EXPECT_TRUE(orientations[3] == LineSegmentOrientation::HORIZONTAL);
  EXPECT_TRUE(orientations[4] == LineSegmentOrientation::UNDEFINED);
  EXPECT_TRUE(orientations[5] == LineSegmentOrientation::UNDEFINED);
}
#endif

}  // namespace
}  // namespace colmap

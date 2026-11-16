// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/image/line.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

#ifdef COLMAP_LSD_ENABLED

TEST(DetectLineSegments, NoDefinedGradients) {
  for (const int value : {0, 128, 255}) {
    SCOPED_TRACE(value);
    Bitmap bitmap(32, 32, false);
    bitmap.Fill(BitmapColor<uint8_t>(static_cast<uint8_t>(value)));
    EXPECT_TRUE(DetectLineSegments(bitmap, 0).empty());
  }

  // A weak ramp also has no gradient above LSD's default threshold.
  Bitmap ramp(32, 32, false);
  for (size_t y = 0; y < 32; ++y) {
    for (size_t x = 0; x < 32; ++x) {
      ramp.SetPixel(x, y, BitmapColor<uint8_t>(static_cast<uint8_t>(x)));
    }
  }
  EXPECT_TRUE(DetectLineSegments(ramp, 0).empty());
}

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

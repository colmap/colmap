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

#include <gtest/gtest.h>

namespace colmap {
namespace {

#ifdef COLMAP_LSD_ENABLED
TEST(DetectLineSegments, Nominal) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, false);
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
  Bitmap bitmap;
  bitmap.Allocate(100, 100, false);
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

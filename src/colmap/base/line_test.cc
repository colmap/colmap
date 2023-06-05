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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#define TEST_NAME "base/line"
#include "colmap/base/line.h"

#include "colmap/util/testing.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestDetectLineSegments) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, false);
  for (size_t i = 0; i < 100; ++i) {
    bitmap.SetPixel(i, i, BitmapColor<uint8_t>(255));
  }

  const auto line_segments = DetectLineSegments(bitmap, 0);

  BOOST_CHECK_EQUAL(line_segments.size(), 2);

  const Eigen::Vector2d ref_start(0, 0);
  const Eigen::Vector2d ref_end(100, 100);
  BOOST_CHECK_LT((line_segments[0].start - ref_start).norm(), 5);
  BOOST_CHECK_LT((line_segments[0].end - ref_end).norm(), 5);
  BOOST_CHECK_LT((line_segments[1].start - ref_end).norm(), 5);
  BOOST_CHECK_LT((line_segments[1].end - ref_start).norm(), 5);

  BOOST_CHECK_EQUAL(DetectLineSegments(bitmap, 150).size(), 0);
}

BOOST_AUTO_TEST_CASE(TestClassifyLineSegmentOrientations) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, false);
  for (size_t i = 60; i < 100; ++i) {
    bitmap.SetPixel(i, 50, BitmapColor<uint8_t>(255));
    bitmap.SetPixel(50, i, BitmapColor<uint8_t>(255));
    bitmap.SetPixel(i, i, BitmapColor<uint8_t>(255));
  }

  const auto line_segments = DetectLineSegments(bitmap, 0);
  BOOST_CHECK_EQUAL(line_segments.size(), 6);

  const auto orientations = ClassifyLineSegmentOrientations(line_segments);
  BOOST_CHECK_EQUAL(orientations.size(), 6);

  BOOST_CHECK(orientations[0] == LineSegmentOrientation::VERTICAL);
  BOOST_CHECK(orientations[1] == LineSegmentOrientation::VERTICAL);
  BOOST_CHECK(orientations[2] == LineSegmentOrientation::HORIZONTAL);
  BOOST_CHECK(orientations[3] == LineSegmentOrientation::HORIZONTAL);
  BOOST_CHECK(orientations[4] == LineSegmentOrientation::UNDEFINED);
  BOOST_CHECK(orientations[5] == LineSegmentOrientation::UNDEFINED);
}

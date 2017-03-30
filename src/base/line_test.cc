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

#define TEST_NAME "base/line"
#include "util/testing.h"

#include "base/line.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestDetectLineSegments) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, false);
  for (size_t i = 0; i < 100; ++i) {
    bitmap.SetPixel(i, i, BitmapColor<uint8_t>(255, 255, 255));
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
    bitmap.SetPixel(i, 50, BitmapColor<uint8_t>(255, 255, 255));
    bitmap.SetPixel(50, i, BitmapColor<uint8_t>(255, 255, 255));
    bitmap.SetPixel(i, i, BitmapColor<uint8_t>(255, 255, 255));
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

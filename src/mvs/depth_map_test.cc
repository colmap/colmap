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

#define TEST_NAME "mvs/depth_map_test"
#include "util/testing.h"

#include "mvs/depth_map.h"

using namespace colmap;
using namespace colmap::mvs;

BOOST_AUTO_TEST_CASE(TestEmpty) {
  DepthMap depth_map;
  BOOST_CHECK_EQUAL(depth_map.GetWidth(), 0);
  BOOST_CHECK_EQUAL(depth_map.GetHeight(), 0);
  BOOST_CHECK_EQUAL(depth_map.GetDepth(), 1);
  BOOST_CHECK_EQUAL(depth_map.GetDepthMin(), -1);
  BOOST_CHECK_EQUAL(depth_map.GetDepthMax(), -1);
}

BOOST_AUTO_TEST_CASE(TestNonEmpty) {
  DepthMap depth_map(1, 2, 0, 1);
  BOOST_CHECK_EQUAL(depth_map.GetWidth(), 1);
  BOOST_CHECK_EQUAL(depth_map.GetHeight(), 2);
  BOOST_CHECK_EQUAL(depth_map.GetDepth(), 1);
  BOOST_CHECK_EQUAL(depth_map.GetDepthMin(), 0);
  BOOST_CHECK_EQUAL(depth_map.GetDepthMax(), 1);
}

BOOST_AUTO_TEST_CASE(TestRescale) {
  DepthMap depth_map(6, 7, 0, 1);
  depth_map.Rescale(0.5);
  BOOST_CHECK_EQUAL(depth_map.GetWidth(), 3);
  BOOST_CHECK_EQUAL(depth_map.GetHeight(), 4);
  BOOST_CHECK_EQUAL(depth_map.GetDepth(), 1);
  BOOST_CHECK_EQUAL(depth_map.GetDepthMin(), 0);
  BOOST_CHECK_EQUAL(depth_map.GetDepthMax(), 1);
}

BOOST_AUTO_TEST_CASE(TestDownsize) {
  DepthMap depth_map(6, 7, 0, 1);
  depth_map.Downsize(2, 4);
  BOOST_CHECK_EQUAL(depth_map.GetWidth(), 2);
  BOOST_CHECK_EQUAL(depth_map.GetHeight(), 2);
  BOOST_CHECK_EQUAL(depth_map.GetDepth(), 1);
  BOOST_CHECK_EQUAL(depth_map.GetDepthMin(), 0);
  BOOST_CHECK_EQUAL(depth_map.GetDepthMax(), 1);
}

BOOST_AUTO_TEST_CASE(TestToBitmap) {
  DepthMap depth_map(2, 2, 0.1, 0.9);
  depth_map.Fill(0.9);
  depth_map.Set(0, 0, 0, 0.1);
  depth_map.Set(0, 1, 0, 0.5);
  const Bitmap bitmap = depth_map.ToBitmap(0, 100);
  BOOST_CHECK_EQUAL(bitmap.Width(), depth_map.GetWidth());
  BOOST_CHECK_EQUAL(bitmap.Height(), depth_map.GetHeight());
  BOOST_CHECK_EQUAL(bitmap.IsRGB(), true);
  BitmapColor<uint8_t> color;
  BOOST_CHECK(bitmap.GetPixel(0, 0, &color));
  BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(0, 0, 128));
  BOOST_CHECK(bitmap.GetPixel(0, 1, &color));
  BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(128, 0, 0));
  BOOST_CHECK(bitmap.GetPixel(1, 0, &color));
  BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(128, 255, 127));
  BOOST_CHECK(bitmap.GetPixel(1, 1, &color));
  BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(128, 0, 0));
}

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

#define TEST_NAME "mvs/normal_map_test"
#include "util/testing.h"

#include "mvs/normal_map.h"

using namespace colmap;
using namespace colmap::mvs;

BOOST_AUTO_TEST_CASE(TestEmpty) {
  NormalMap normal_map;
  BOOST_CHECK_EQUAL(normal_map.GetWidth(), 0);
  BOOST_CHECK_EQUAL(normal_map.GetHeight(), 0);
  BOOST_CHECK_EQUAL(normal_map.GetDepth(), 3);
}

BOOST_AUTO_TEST_CASE(TestNonEmpty) {
  NormalMap normal_map(1, 2);
  BOOST_CHECK_EQUAL(normal_map.GetWidth(), 1);
  BOOST_CHECK_EQUAL(normal_map.GetHeight(), 2);
  BOOST_CHECK_EQUAL(normal_map.GetDepth(), 3);
}

BOOST_AUTO_TEST_CASE(TestRescale) {
  NormalMap normal_map(6, 7);
  normal_map.Rescale(0.5);
  BOOST_CHECK_EQUAL(normal_map.GetWidth(), 3);
  BOOST_CHECK_EQUAL(normal_map.GetHeight(), 4);
  BOOST_CHECK_EQUAL(normal_map.GetDepth(), 3);
}

BOOST_AUTO_TEST_CASE(TestDownsize) {
  NormalMap normal_map(6, 7);
  normal_map.Downsize(2, 4);
  BOOST_CHECK_EQUAL(normal_map.GetWidth(), 2);
  BOOST_CHECK_EQUAL(normal_map.GetHeight(), 2);
  BOOST_CHECK_EQUAL(normal_map.GetDepth(), 3);
}

BOOST_AUTO_TEST_CASE(TestToBitmap) {
  NormalMap normal_map(2, 2);
  normal_map.Set(0, 0, 0, 0);
  normal_map.Set(0, 0, 1, 0);
  normal_map.Set(0, 0, 2, 1);
  normal_map.Set(0, 1, 0, 0);
  normal_map.Set(0, 1, 1, 1);
  normal_map.Set(0, 1, 2, 0);
  normal_map.Set(1, 0, 0, 1);
  normal_map.Set(1, 0, 1, 0);
  normal_map.Set(1, 0, 2, 0);
  normal_map.Set(1, 1, 0, 1 / std::sqrt(2.0f));
  normal_map.Set(1, 1, 1, 1 / std::sqrt(2.0f));
  normal_map.Set(1, 1, 2, 0);
  const Bitmap bitmap = normal_map.ToBitmap();
  BOOST_CHECK_EQUAL(bitmap.Width(), normal_map.GetWidth());
  BOOST_CHECK_EQUAL(bitmap.Height(), normal_map.GetHeight());
  BOOST_CHECK_EQUAL(bitmap.IsRGB(), true);
  BitmapColor<uint8_t> color;
  BOOST_CHECK(bitmap.GetPixel(0, 0, &color));
  BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(128, 128, 0));
  BOOST_CHECK(bitmap.GetPixel(0, 1, &color));
  BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(0, 128, 0));
  BOOST_CHECK(bitmap.GetPixel(1, 0, &color));
  BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(128, 0, 0));
  BOOST_CHECK(bitmap.GetPixel(1, 1, &color));
  BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(37, 37, 0));
}

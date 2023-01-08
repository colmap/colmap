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

// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "base/warp"
#include <boost/test/unit_test.hpp>

#include "base/warp.h"
#include "util/random.h"

using namespace colmap;
namespace {

void GenerateRandomBitmap(const int width, const int height, const bool as_rgb,
                          Bitmap* bitmap) {
  bitmap->Allocate(width, height, as_rgb);
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      BitmapColor<uint8_t> color;
      color.r = RandomInteger<uint8_t>(0, 255);
      color.g = RandomInteger<uint8_t>(0, 255);
      color.b = RandomInteger<uint8_t>(0, 255);
      bitmap->SetPixel(x, y, color);
    }
  }
}

void CheckBitmapsEqual(const Bitmap& bitmap1, const Bitmap& bitmap2) {
  BOOST_REQUIRE_EQUAL(bitmap1.IsGrey(), bitmap2.IsGrey());
  BOOST_REQUIRE_EQUAL(bitmap1.IsRGB(), bitmap2.IsRGB());
  BOOST_REQUIRE_EQUAL(bitmap1.Width(), bitmap2.Width());
  BOOST_REQUIRE_EQUAL(bitmap1.Height(), bitmap2.Height());
  for (int x = 0; x < bitmap1.Width(); ++x) {
    for (int y = 0; y < bitmap1.Height(); ++y) {
      BitmapColor<uint8_t> color1;
      BitmapColor<uint8_t> color2;
      BOOST_CHECK(bitmap1.GetPixel(x, y, &color1));
      BOOST_CHECK(bitmap1.GetPixel(x, y, &color2));
      BOOST_CHECK_EQUAL(color1, color2);
    }
  }
}

}  // namespace

BOOST_AUTO_TEST_CASE(TestIdenticalCameras) {
  Camera source_camera;
  source_camera.InitializeWithName("PINHOLE", 1, 100, 100);
  Camera target_camera = source_camera;
  Bitmap source_image_gray;
  GenerateRandomBitmap(100, 100, false, &source_image_gray);
  Bitmap target_image_gray;
  WarpImageBetweenCameras(source_camera, target_camera, source_image_gray,
                          &target_image_gray);
  CheckBitmapsEqual(source_image_gray, target_image_gray);
  Bitmap source_image_rgb;
  GenerateRandomBitmap(100, 100, true, &source_image_rgb);
  Bitmap target_image_rgb;
  WarpImageBetweenCameras(source_camera, target_camera, source_image_rgb,
                          &target_image_rgb);
  CheckBitmapsEqual(source_image_rgb, target_image_rgb);
}

BOOST_AUTO_TEST_CASE(TestShiftedCameras) {
  Camera source_camera;
  source_camera.InitializeWithName("PINHOLE", 1, 100, 100);
  Camera target_camera = source_camera;
  target_camera.SetPrincipalPointX(0.0);
  Bitmap source_image_gray;
  GenerateRandomBitmap(100, 100, true, &source_image_gray);
  Bitmap target_image_gray;
  WarpImageBetweenCameras(source_camera, target_camera, source_image_gray,
                          &target_image_gray);
  for (int x = 0; x < target_image_gray.Width(); ++x) {
    for (int y = 0; y < target_image_gray.Height(); ++y) {
      BitmapColor<uint8_t> color;
      BOOST_CHECK(target_image_gray.GetPixel(x, y, &color));
      if (x >= 50) {
        BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(0, 0, 0));
      } else {
        BitmapColor<uint8_t> source_color;
        if (source_image_gray.GetPixel(x + 50, y, &source_color) &&
            color != BitmapColor<uint8_t>(0, 0, 0)) {
          BOOST_CHECK_EQUAL(color, source_color);
        }
      }
    }
  }
}

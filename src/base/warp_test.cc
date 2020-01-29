// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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

#define TEST_NAME "base/warp"
#include "util/testing.h"

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
      color.r = RandomInteger<int>(0, 255);
      color.g = RandomInteger<int>(0, 255);
      color.b = RandomInteger<int>(0, 255);
      bitmap->SetPixel(x, y, color);
    }
  }
}

// Check that the two bitmaps are equal, ignoring a 1px boundary.
void CheckBitmapsEqual(const Bitmap& bitmap1, const Bitmap& bitmap2) {
  BOOST_REQUIRE_EQUAL(bitmap1.IsGrey(), bitmap2.IsGrey());
  BOOST_REQUIRE_EQUAL(bitmap1.IsRGB(), bitmap2.IsRGB());
  BOOST_REQUIRE_EQUAL(bitmap1.Width(), bitmap2.Width());
  BOOST_REQUIRE_EQUAL(bitmap1.Height(), bitmap2.Height());
  for (int x = 1; x < bitmap1.Width() - 1; ++x) {
    for (int y = 1; y < bitmap1.Height() - 1; ++y) {
      BitmapColor<uint8_t> color1;
      BitmapColor<uint8_t> color2;
      BOOST_CHECK(bitmap1.GetPixel(x, y, &color1));
      BOOST_CHECK(bitmap2.GetPixel(x, y, &color2));
      BOOST_CHECK_EQUAL(color1, color2);
    }
  }
}

// Check that the two bitmaps are equal, ignoring a 1px boundary.
void CheckBitmapsTransposed(const Bitmap& bitmap1, const Bitmap& bitmap2) {
  BOOST_REQUIRE_EQUAL(bitmap1.IsGrey(), bitmap2.IsGrey());
  BOOST_REQUIRE_EQUAL(bitmap1.IsRGB(), bitmap2.IsRGB());
  BOOST_REQUIRE_EQUAL(bitmap1.Width(), bitmap2.Width());
  BOOST_REQUIRE_EQUAL(bitmap1.Height(), bitmap2.Height());
  for (int x = 1; x < bitmap1.Width() - 1; ++x) {
    for (int y = 1; y < bitmap1.Height() - 1; ++y) {
      BitmapColor<uint8_t> color1;
      BitmapColor<uint8_t> color2;
      BOOST_CHECK(bitmap1.GetPixel(x, y, &color1));
      BOOST_CHECK(bitmap2.GetPixel(y, x, &color2));
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
        BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(0));
      } else {
        BitmapColor<uint8_t> source_color;
        if (source_image_gray.GetPixel(x + 50, y, &source_color) &&
            color != BitmapColor<uint8_t>(0)) {
          BOOST_CHECK_EQUAL(color, source_color);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(TestWarpImageWithHomographyIdentity) {
  Bitmap source_image_gray;
  GenerateRandomBitmap(100, 100, false, &source_image_gray);
  Bitmap target_image_gray;
  target_image_gray.Allocate(100, 100, false);
  WarpImageWithHomography(Eigen::Matrix3d::Identity(), source_image_gray,
                          &target_image_gray);
  CheckBitmapsEqual(source_image_gray, target_image_gray);

  Bitmap source_image_rgb;
  GenerateRandomBitmap(100, 100, true, &source_image_rgb);
  Bitmap target_image_rgb;
  target_image_rgb.Allocate(100, 100, true);
  WarpImageWithHomography(Eigen::Matrix3d::Identity(), source_image_rgb,
                          &target_image_rgb);
  CheckBitmapsEqual(source_image_rgb, target_image_rgb);
}

BOOST_AUTO_TEST_CASE(TestWarpImageWithHomographyTransposed) {
  Eigen::Matrix3d H;
  H << 0, 1, 0, 1, 0, 0, 0, 0, 1;

  Bitmap source_image_gray;
  GenerateRandomBitmap(100, 100, false, &source_image_gray);
  Bitmap target_image_gray;
  target_image_gray.Allocate(100, 100, false);
  WarpImageWithHomography(H, source_image_gray, &target_image_gray);
  CheckBitmapsTransposed(source_image_gray, target_image_gray);

  Bitmap source_image_rgb;
  GenerateRandomBitmap(100, 100, true, &source_image_rgb);
  Bitmap target_image_rgb;
  target_image_rgb.Allocate(100, 100, true);
  WarpImageWithHomography(H, source_image_rgb, &target_image_rgb);
  CheckBitmapsTransposed(source_image_rgb, target_image_rgb);
}

BOOST_AUTO_TEST_CASE(TestWarpImageWithHomographyBetweenCamerasIdentity) {
  Camera source_camera;
  source_camera.InitializeWithName("PINHOLE", 1, 100, 100);
  Camera target_camera = source_camera;

  Bitmap source_image_gray;
  GenerateRandomBitmap(100, 100, false, &source_image_gray);
  Bitmap target_image_gray;
  target_image_gray.Allocate(100, 100, false);
  WarpImageWithHomographyBetweenCameras(Eigen::Matrix3d::Identity(),
                                        source_camera, target_camera,
                                        source_image_gray, &target_image_gray);
  CheckBitmapsEqual(source_image_gray, target_image_gray);

  Bitmap source_image_rgb;
  GenerateRandomBitmap(100, 100, true, &source_image_rgb);
  Bitmap target_image_rgb;
  target_image_rgb.Allocate(100, 100, true);
  WarpImageWithHomographyBetweenCameras(Eigen::Matrix3d::Identity(),
                                        source_camera, target_camera,
                                        source_image_rgb, &target_image_rgb);
  CheckBitmapsEqual(source_image_rgb, target_image_rgb);
}

BOOST_AUTO_TEST_CASE(TestWarpImageWithHomographyBetweenCamerasTransposed) {
  Camera source_camera;
  source_camera.InitializeWithName("PINHOLE", 1, 100, 100);
  Camera target_camera = source_camera;

  Eigen::Matrix3d H;
  H << 0, 1, 0, 1, 0, 0, 0, 0, 1;

  Bitmap source_image_gray;
  GenerateRandomBitmap(100, 100, false, &source_image_gray);
  Bitmap target_image_gray;
  target_image_gray.Allocate(100, 100, false);
  WarpImageWithHomographyBetweenCameras(H, source_camera, target_camera,
                                        source_image_gray, &target_image_gray);
  CheckBitmapsTransposed(source_image_gray, target_image_gray);

  Bitmap source_image_rgb;
  GenerateRandomBitmap(100, 100, true, &source_image_rgb);
  Bitmap target_image_rgb;
  target_image_rgb.Allocate(100, 100, true);
  WarpImageWithHomographyBetweenCameras(H, source_camera, target_camera,
                                        source_image_rgb, &target_image_rgb);
  CheckBitmapsTransposed(source_image_rgb, target_image_rgb);
}

BOOST_AUTO_TEST_CASE(TestResampleImageBilinear) {
  std::vector<float> image(16);
  for (size_t i = 0; i < image.size(); ++i) {
    image[i] = i;
  }

  std::vector<float> resampled(4);
  ResampleImageBilinear(image.data(), 4, 4, 2, 2, resampled.data());

  BOOST_CHECK_EQUAL(resampled[0], 2.5);
  BOOST_CHECK_EQUAL(resampled[1], 4.5);
  BOOST_CHECK_EQUAL(resampled[2], 10.5);
  BOOST_CHECK_EQUAL(resampled[3], 12.5);
}

BOOST_AUTO_TEST_CASE(TestSmoothImage) {
  std::vector<float> image(16);
  for (size_t i = 0; i < image.size(); ++i) {
    image[i] = i;
  }

  std::vector<float> smoothed(16);
  SmoothImage(image.data(), 4, 4, 1, 1, smoothed.data());

  BOOST_CHECK_CLOSE(smoothed[0], 1.81673253, 1e-3);
  BOOST_CHECK_CLOSE(smoothed[1], 2.51182437, 1e-3);
  BOOST_CHECK_CLOSE(smoothed[2], 3.39494729, 1e-3);
  BOOST_CHECK_CLOSE(smoothed[3], 4.09003973, 1e-3);
  BOOST_CHECK_CLOSE(smoothed[4], 4.59710073, 1e-3);
  BOOST_CHECK_CLOSE(smoothed[5], 5.29219341, 1e-3);
  BOOST_CHECK_CLOSE(smoothed[6], 6.17531633, 1e-3);
  BOOST_CHECK_CLOSE(smoothed[7], 6.87040806, 1e-3);
}

BOOST_AUTO_TEST_CASE(TestDownsampleImage) {
  std::vector<float> image(16);
  for (size_t i = 0; i < image.size(); ++i) {
    image[i] = i;
  }

  std::vector<float> downsampled(4);
  DownsampleImage(image.data(), 4, 4, 2, 2, downsampled.data());

  BOOST_CHECK_CLOSE(downsampled[0], 2.76810598, 1e-3);
  BOOST_CHECK_CLOSE(downsampled[1], 4.66086388, 1e-3);
  BOOST_CHECK_CLOSE(downsampled[2], 10.3391361, 1e-3);
  BOOST_CHECK_CLOSE(downsampled[3], 12.2318935, 1e-3);
}

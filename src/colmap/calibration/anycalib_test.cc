// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#include "colmap/calibration/anycalib.h"

#include "colmap/calibration/calibrator.h"
#include "colmap/math/random.h"

#include <cstdlib>

#include <gtest/gtest.h>

namespace colmap {
namespace {

void CreateSolidRgbImage(int width, int height, uint8_t value, Bitmap* bitmap) {
  *bitmap = Bitmap(width, height, /*as_rgb=*/true);
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      bitmap->SetPixel(c, r, BitmapColor<uint8_t>(value, value, value));
    }
  }
}

TEST(PrepareAnyCalibInputTest, LandscapeImage) {
  Bitmap bitmap;
  CreateSolidRgbImage(640, 480, 255, &bitmap);
  const AnyCalibInput input = PrepareAnyCalibInput(bitmap);
  ASSERT_EQ(input.data.size(), 3 * 322 * 322);
  // Center-crop to 480-square: shift_x = -(160 / 2), then scale by 322/480.
  const double s = 322.0 / 480;
  EXPECT_DOUBLE_EQ(input.scale_xy.x(), s);
  EXPECT_DOUBLE_EQ(input.scale_xy.y(), s);
  EXPECT_DOUBLE_EQ(input.shift_xy.x(), -80 * s);
  EXPECT_DOUBLE_EQ(input.shift_xy.y(), 0);
  for (const float v : input.data) {
    EXPECT_FLOAT_EQ(v, 1.0f);
  }
}

TEST(PrepareAnyCalibInputTest, PortraitImage) {
  Bitmap bitmap;
  CreateSolidRgbImage(100, 200, 0, &bitmap);
  const AnyCalibInput input = PrepareAnyCalibInput(bitmap);
  ASSERT_EQ(input.data.size(), 3 * 322 * 322);
  // Upscale by 3.22 to 322x644, then center-crop to 322-square.
  EXPECT_DOUBLE_EQ(input.scale_xy.x(), 3.22);
  EXPECT_DOUBLE_EQ(input.scale_xy.y(), 3.22);
  EXPECT_DOUBLE_EQ(input.shift_xy.x(), 0);
  EXPECT_DOUBLE_EQ(input.shift_xy.y(), -161);
  for (const float v : input.data) {
    EXPECT_FLOAT_EQ(v, 0.0f);
  }
}

TEST(PrepareAnyCalibInputTest, SmallLandscapeImage) {
  Bitmap bitmap;
  CreateSolidRgbImage(100, 80, 128, &bitmap);
  const AnyCalibInput input = PrepareAnyCalibInput(bitmap);
  ASSERT_EQ(input.data.size(), 3 * 322 * 322);
  // Upscale by max(322/80, 322/100) = 4.025 to 402x322, center-crop to
  // 322-square, no further rescaling.
  EXPECT_DOUBLE_EQ(input.scale_xy.x(), 402.0 / 100);
  EXPECT_DOUBLE_EQ(input.scale_xy.y(), 322.0 / 80);
  EXPECT_DOUBLE_EQ(input.shift_xy.x(), -40);
  EXPECT_DOUBLE_EQ(input.shift_xy.y(), 0);
  for (const float v : input.data) {
    EXPECT_FLOAT_EQ(v, 128 / 255.0f);
  }
}

TEST(CreateAnyCalibCalibratorTest, EmptyModelPathThrows) {
  CameraCalibrationOptions options;
  options.anycalib.model_path = "";
  EXPECT_THROW(CreateAnyCalibCalibrator(options), std::exception);
}

TEST(CreateAnyCalibCalibratorTest, MissingModelFileThrows) {
  CameraCalibrationOptions options;
  options.anycalib.model_path = "/nonexistent/anycalib_gen.onnx";
  EXPECT_THROW(CreateAnyCalibCalibrator(options), std::exception);
}

// Full-model smoke test, run only when the exported model is available:
// COLMAP_ANYCALIB_MODEL_PATH=/path/to/anycalib_gen.onnx ctest -R anycalib_test
TEST(AnyCalibCalibratorTest, SmokeTestWithModel) {
  const char* model_path = std::getenv("COLMAP_ANYCALIB_MODEL_PATH");
  if (model_path == nullptr) {
    GTEST_SKIP() << "Set COLMAP_ANYCALIB_MODEL_PATH to run this test";
  }
  Bitmap bitmap(64, 48, /*as_rgb=*/true);
  for (int r = 0; r < 48; ++r) {
    for (int c = 0; c < 64; ++c) {
      bitmap.SetPixel(c,
                      r,
                      BitmapColor<uint8_t>(RandomUniformInteger(0, 255),
                                           RandomUniformInteger(0, 255),
                                           RandomUniformInteger(0, 255)));
    }
  }
  CameraCalibrationOptions options;
  options.use_gpu = false;
  options.anycalib.model_path = model_path;
  auto calibrator = CameraCalibrator::Create(options);
  Camera camera;
  // Random noise is not expected to calibrate; the test only checks that
  // inference runs end to end without crashing.
  const bool success = calibrator->Calibrate(bitmap, &camera);
  if (success) {
    EXPECT_TRUE(camera.VerifyParams());
    EXPECT_EQ(camera.width, 64);
    EXPECT_EQ(camera.height, 48);
    EXPECT_TRUE(camera.has_prior_focal_length);
  }
}

}  // namespace
}  // namespace colmap

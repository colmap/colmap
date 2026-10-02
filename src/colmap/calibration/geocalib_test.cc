// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/calibration/geocalib.h"

#include "colmap/math/random.h"

#include <algorithm>

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

TEST(PrepareGeoCalibInputTest, LandscapeImagePreservesAspectRatio) {
  Bitmap bitmap;
  CreateSolidRgbImage(640, 480, 255, &bitmap);
  const GeoCalibInput input =
      PrepareGeoCalibInput(bitmap, /*image_size=*/320, /*force_square=*/false);
  // Short side 480 -> 320, long side 640 -> round(640 * 320 / 480) = 427,
  // cropped to multiple of 32: 416 x 320.
  EXPECT_EQ(input.width, 416);
  EXPECT_EQ(input.height, 320);
  EXPECT_EQ(input.data.size(), 3 * 416 * 320);
  EXPECT_DOUBLE_EQ(input.scale_xy.x(), 427.0 / 640.0);
  EXPECT_DOUBLE_EQ(input.scale_xy.y(), 320.0 / 480.0);
  EXPECT_DOUBLE_EQ(input.shift_xy.x(), -((427 - 416) / 2));
  EXPECT_DOUBLE_EQ(input.shift_xy.y(), 0.0);

  const Eigen::Vector2d orig_pt(250.0, 180.0);
  const Eigen::Vector2d net_pt =
      orig_pt.cwiseProduct(input.scale_xy) + input.shift_xy;
  EXPECT_TRUE(input.ImgToOrig(net_pt).isApprox(orig_pt, 1e-12));
}

TEST(PrepareGeoCalibInputTest, ForceSquareCrop) {
  Bitmap bitmap;
  CreateSolidRgbImage(640, 480, 128, &bitmap);
  const GeoCalibInput input =
      PrepareGeoCalibInput(bitmap, /*image_size=*/320, /*force_square=*/true);
  EXPECT_EQ(input.width, 320);
  EXPECT_EQ(input.height, 320);
  EXPECT_EQ(input.data.size(), 3 * 320 * 320);
}

TEST(GeoCalibTest, MissingModelFileThrows) {
  GeoCalibOptions options;
  options.model_path = "/nonexistent/geocalib.onnx";
  options.use_gpu = false;
  EXPECT_THROW(GeoCalib::Create(options), std::exception);
}

TEST(GeoCalibTest, SmokeTestWithModel) {
  Bitmap bitmap(96, 64, /*as_rgb=*/true);
  for (int r = 0; r < 64; ++r) {
    for (int c = 0; c < 96; ++c) {
      bitmap.SetPixel(c,
                      r,
                      BitmapColor<uint8_t>(RandomUniformInteger(0, 255),
                                           RandomUniformInteger(0, 255),
                                           RandomUniformInteger(0, 255)));
    }
  }
  GeoCalibOptions options;
  options.image_size = 64;
  options.use_gpu = false;
  auto geocalib = GeoCalib::Create(options);

  const PerspectiveField field = geocalib->PredictPerspectiveField(bitmap);
  EXPECT_TRUE(field.Check());
  EXPECT_EQ(field.width, 96);
  EXPECT_EQ(field.height, 64);

  Camera camera =
      Camera::CreateFromModelName(1, "SIMPLE_PINHOLE", 80.0, 96, 64);
  const FittedPerspectiveFields fitted =
      geocalib->Calibrate(bitmap, &camera, /*refine_camera=*/true);
  if (fitted.success) {
    EXPECT_NEAR(fitted.gravity_in_rig.norm(), 1.0, 1e-6);
    EXPECT_TRUE(camera.VerifyParams());
  }
}

}  // namespace
}  // namespace colmap

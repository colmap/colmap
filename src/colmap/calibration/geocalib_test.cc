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
  const GeoCalibInput input = PrepareGeoCalibInput(
      bitmap, /*image_size=*/320, /*force_square=*/false, PosePrior());
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
  const GeoCalibInput input = PrepareGeoCalibInput(
      bitmap, /*image_size=*/320, /*force_square=*/true, PosePrior());
  EXPECT_EQ(input.width, 320);
  EXPECT_EQ(input.height, 320);
  EXPECT_EQ(input.data.size(), 3 * 320 * 320);
}

TEST(PrepareGeoCalibInputTest, GravityRotationMapsBackToOriginal) {
  const std::vector<Eigen::Vector3d> gravities = {
      Eigen::Vector3d(0, 1, 0),
      Eigen::Vector3d(-1, 0, 0),
      Eigen::Vector3d(0, -1, 0),
      Eigen::Vector3d(1, 0, 0),
  };
  const Eigen::Vector2d original_point(100.0, 200.0);
  const Eigen::Vector2d original_up = Eigen::Vector2d(0.6, -0.8).normalized();
  for (int rot90 = 0; rot90 < 4; ++rot90) {
    Bitmap bitmap;
    CreateSolidRgbImage(640, 480, 255, &bitmap);
    PosePrior pose_prior;
    pose_prior.gravity = gravities[rot90];
    const GeoCalibInput input = PrepareGeoCalibInput(
        bitmap, /*image_size=*/320, /*force_square=*/false, pose_prior);
    EXPECT_EQ(input.image_rot90, rot90) << "rot90=" << rot90;
    const bool swap_dims = (rot90 == 1 || rot90 == 3);
    EXPECT_EQ(input.upright_width, swap_dims ? 480 : 640);
    EXPECT_EQ(input.upright_height, swap_dims ? 640 : 480);

    Eigen::Vector2d upright_point;
    Eigen::Vector2d upright_up;
    switch (rot90) {
      case 0:
        upright_point = original_point;
        upright_up = original_up;
        break;
      case 1:
        upright_point =
            Eigen::Vector2d(original_point.y(), 640 - original_point.x());
        upright_up = Eigen::Vector2d(original_up.y(), -original_up.x());
        break;
      case 2:
        upright_point =
            Eigen::Vector2d(640 - original_point.x(), 480 - original_point.y());
        upright_up = Eigen::Vector2d(-original_up.x(), -original_up.y());
        break;
      case 3:
        upright_point =
            Eigen::Vector2d(480 - original_point.y(), original_point.x());
        upright_up = Eigen::Vector2d(-original_up.y(), original_up.x());
        break;
      default:
        FAIL() << "Invalid rotation: " << rot90;
        break;
    }
    const Eigen::Vector2d network_point =
        upright_point.cwiseProduct(input.scale_xy) + input.shift_xy;
    const Eigen::Vector2d network_up =
        upright_up.cwiseProduct(input.scale_xy).normalized();
    EXPECT_TRUE(input.ImgToOrig(network_point).isApprox(original_point, 1e-12))
        << "rot90=" << rot90;
    EXPECT_TRUE(input.UpToOrig(network_up).isApprox(original_up, 1e-12))
        << "rot90=" << rot90;
  }
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

  const PerspectiveField field =
      geocalib->PredictPerspectiveField(bitmap, PosePrior());
  EXPECT_TRUE(field.Check());
  EXPECT_EQ(field.width, 96);
  EXPECT_EQ(field.height, 64);

  Camera camera =
      Camera::CreateFromModelName(1, "SIMPLE_PINHOLE", 80.0, 96, 64);
  const FittedPerspectiveFields fitted =
      geocalib->Calibrate(bitmap, &camera, /*refine_camera=*/true, PosePrior());
  if (fitted.success) {
    EXPECT_NEAR(fitted.gravity_in_rig.norm(), 1.0, 1e-6);
    EXPECT_TRUE(camera.VerifyParams());
  }
}

}  // namespace
}  // namespace colmap

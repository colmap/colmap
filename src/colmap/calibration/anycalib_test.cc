// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/calibration/anycalib.h"

#include "colmap/calibration/calibrator.h"
#include "colmap/math/random.h"

#include <algorithm>
#include <limits>
#include <vector>

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

void ExpectAllTensorValues(const AnyCalibInput& input, float expected) {
  ASSERT_FALSE(input.data.empty());
  const auto [min_it, max_it] =
      std::minmax_element(input.data.begin(), input.data.end());
  EXPECT_FLOAT_EQ(*min_it, expected);
  EXPECT_FLOAT_EQ(*max_it, expected);
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
  ExpectAllTensorValues(input, 1.0f);
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
  ExpectAllTensorValues(input, 0.0f);
}

TEST(PrepareAnyCalibInputTest, SmallLandscapeImage) {
  Bitmap bitmap;
  CreateSolidRgbImage(100, 80, 128, &bitmap);
  const AnyCalibInput input = PrepareAnyCalibInput(bitmap);
  ASSERT_EQ(input.data.size(), 3 * 322 * 322);
  // Upscale by max(322/80, 322/100) = 4.025 to 403x322 (rounded), center-crop
  // to 322-square, no further rescaling.
  EXPECT_DOUBLE_EQ(input.scale_xy.x(), 403.0 / 100);
  EXPECT_DOUBLE_EQ(input.scale_xy.y(), 322.0 / 80);
  EXPECT_DOUBLE_EQ(input.shift_xy.x(), -40);
  EXPECT_DOUBLE_EQ(input.shift_xy.y(), 0);
  ExpectAllTensorValues(input, 128 / 255.0f);
}

TEST(PrepareAnyCalibInputTest, RejectsEmptyBitmap) {
  Bitmap bitmap;
  EXPECT_THROW(PrepareAnyCalibInput(bitmap), std::exception);
}

TEST(PrepareAnyCalibInputTest, GravityRotationMapsBackToOriginal) {
  // Gravity direction per upright rotation, see `ComputeRot90FromGravity`.
  const std::vector<Eigen::Vector3d> gravities = {
      Eigen::Vector3d(0, 1, 0),
      Eigen::Vector3d(-1, 0, 0),
      Eigen::Vector3d(0, -1, 0),
      Eigen::Vector3d(1, 0, 0),
  };
  const Eigen::Vector2d original_point(100, 200);
  const Eigen::Vector3d original_ray(1, 2, 3);
  for (int rot90 = 0; rot90 < 4; ++rot90) {
    Bitmap bitmap;
    CreateSolidRgbImage(640, 480, 255, &bitmap);
    PosePrior pose_prior;
    pose_prior.gravity = gravities[rot90];
    const AnyCalibInput input = PrepareAnyCalibInput(bitmap, pose_prior);
    EXPECT_EQ(input.image_rot90, rot90) << "rot90=" << rot90;
    const bool swap_dims = (rot90 == 1 || rot90 == 3);
    EXPECT_EQ(input.upright_width, swap_dims ? 480 : 640);
    EXPECT_EQ(input.upright_height, swap_dims ? 640 : 480);

    // Forward map of the original point/ray to the upright image, inverse of
    // `ImgToOrig` / `CamToOrig` up to scale and shift.
    Eigen::Vector2d upright_point;
    Eigen::Vector3d upright_ray;
    switch (rot90) {
      case 0:
        upright_point = original_point;
        upright_ray = original_ray;
        break;
      case 1:
        upright_point =
            Eigen::Vector2d(original_point.y(), 640 - original_point.x());
        upright_ray = Eigen::Vector3d(
            original_ray.y(), -original_ray.x(), original_ray.z());
        break;
      case 2:
        upright_point =
            Eigen::Vector2d(640 - original_point.x(), 480 - original_point.y());
        upright_ray = Eigen::Vector3d(
            -original_ray.x(), -original_ray.y(), original_ray.z());
        break;
      case 3:
        upright_point =
            Eigen::Vector2d(480 - original_point.y(), original_point.x());
        upright_ray = Eigen::Vector3d(
            -original_ray.y(), original_ray.x(), original_ray.z());
        break;
      default:
        FAIL() << "Invalid rotation: " << rot90;
        break;
    }
    const Eigen::Vector2d network_point =
        upright_point.cwiseProduct(input.scale_xy) + input.shift_xy;
    EXPECT_TRUE(input.ImgToOrig(network_point).isApprox(original_point, 1e-12))
        << "rot90=" << rot90;
    EXPECT_TRUE(input.CamToOrig(upright_ray).isApprox(original_ray, 1e-12))
        << "rot90=" << rot90;
  }
}

TEST(PrepareAnyCalibInputTest, DegenerateGravityDoesNotCrash) {
  Bitmap bitmap;
  CreateSolidRgbImage(64, 48, 255, &bitmap);
  PosePrior zero_gravity;
  zero_gravity.gravity = Eigen::Vector3d::Zero();
  const AnyCalibInput input = PrepareAnyCalibInput(bitmap, zero_gravity);
  EXPECT_GE(input.image_rot90, 0);
  EXPECT_LT(input.image_rot90, 4);
  EXPECT_EQ(input.data.size(), 3 * 322 * 322);

  // Non-finite gravity counts as absent.
  PosePrior nan_gravity;
  nan_gravity.gravity =
      Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 0, 0);
  EXPECT_EQ(PrepareAnyCalibInput(bitmap, nan_gravity).image_rot90, 0);
}

TEST(CreateAnyCalibCalibratorTest, EmptyModelPathThrows) {
  MonocularCalibrationOptions options(MonocularCalibratorType::ANYCALIB);
  options.anycalib->model_path = "";
  // Route through the public factory, like production callers.
  EXPECT_THROW(MonocularCalibrator::Create(options), std::exception);
}

TEST(CreateAnyCalibCalibratorTest, NullAnyCalibOptionsThrows) {
  MonocularCalibrationOptions options;
  options.anycalib = nullptr;
  EXPECT_THROW(CreateAnyCalibCalibrator(options), std::exception);
}

TEST(CreateAnyCalibCalibratorTest, MissingModelFileThrows) {
  MonocularCalibrationOptions options;
  options.anycalib->model_path = "/nonexistent/anycalib_gen.onnx";
  EXPECT_THROW(CreateAnyCalibCalibrator(options), std::exception);
}

// Full-model smoke test with real inference. Like the ALIKED/LoMa tests, this
// downloads the default model on first run (hash-verified cache).
TEST(AnyCalibCalibratorTest, SmokeTestWithModel) {
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
  MonocularCalibrationOptions options(MonocularCalibratorType::ANYCALIB);
  options.use_gpu = false;
  auto calibrator = MonocularCalibrator::Create(options);
  // A valid model exercises inference end to end; the empty default target
  // model preserves it.
  Camera camera;
  camera.model_id = CameraModelId::kSimpleRadial;
  camera.width = 64;
  camera.height = 48;
  camera.params = {50, 32, 24, 0};
  // Random noise is not expected to calibrate; the test only checks that
  // inference runs end to end without crashing.
  PosePrior pose_prior;
  const bool success = calibrator->Calibrate(bitmap, &camera, &pose_prior);
  if (success) {
    EXPECT_EQ(camera.model_id, CameraModelId::kSimpleRadial);
    EXPECT_TRUE(camera.VerifyParams());
    EXPECT_EQ(camera.width, 64);
    EXPECT_EQ(camera.height, 48);
    EXPECT_TRUE(camera.has_prior_focal_length);
  }
  EXPECT_FALSE(pose_prior.HasPosition());
  EXPECT_FALSE(pose_prior.HasGravity());
}

// Grayscale input is converted for inference, and the pose prior is populated
// from EXIF independently of the intrinsics fit. Reuses the cached model from
// the smoke test.
TEST(AnyCalibCalibratorTest, GreyInputAndPosePrior) {
  Bitmap bitmap(64, 48, /*as_rgb=*/false);
  for (int r = 0; r < 48; ++r) {
    for (int c = 0; c < 64; ++c) {
      bitmap.SetPixel(c, r, BitmapColor<uint8_t>(RandomUniformInteger(0, 255)));
    }
  }
  float latitude[3] = {47.3769f, 0, 0};
  bitmap.SetMetaData("GPS:Latitude", "point", latitude);
  bitmap.SetMetaData("GPS:LatitudeRef", "N");
  float longitude[3] = {8.5417f, 0, 0};
  bitmap.SetMetaData("GPS:Longitude", "point", longitude);
  bitmap.SetMetaData("GPS:LongitudeRef", "E");
  float altitude = 400.0f;
  bitmap.SetMetaData("GPS:Altitude", "float", &altitude);
  bitmap.SetMetaData("GPS:AltitudeRef", "0");
  int orientation = 1;
  bitmap.SetMetaData("Orientation", "int", &orientation);
  MonocularCalibrationOptions options(MonocularCalibratorType::ANYCALIB);
  options.use_gpu = false;
  auto calibrator = MonocularCalibrator::Create(options);
  Camera camera;
  camera.model_id = CameraModelId::kSimpleRadial;
  camera.width = 64;
  camera.height = 48;
  camera.params = {50, 32, 24, 0};
  PosePrior pose_prior;
  calibrator->Calibrate(bitmap, &camera, &pose_prior);
  ASSERT_TRUE(pose_prior.HasPosition());
  EXPECT_DOUBLE_EQ(pose_prior.position.x(), 47.3769f);
  EXPECT_DOUBLE_EQ(pose_prior.position.y(), 8.5417f);
  EXPECT_DOUBLE_EQ(pose_prior.position.z(), 400.0f);
  EXPECT_EQ(pose_prior.coordinate_system, PosePrior::CoordinateSystem::WGS84);
  ASSERT_TRUE(pose_prior.HasGravity());
  EXPECT_EQ(pose_prior.gravity, Eigen::Vector3d(0, 1, 0));
}

// A camera without a valid model fails gracefully instead of throwing inside
// the camera-model switches. Reuses the cached model from the smoke test.
TEST(AnyCalibCalibratorTest, InvalidCameraModelReturnsFalse) {
  Bitmap bitmap;
  CreateSolidRgbImage(64, 48, 255, &bitmap);
  MonocularCalibrationOptions options(MonocularCalibratorType::ANYCALIB);
  options.use_gpu = false;
  auto calibrator = MonocularCalibrator::Create(options);
  Camera camera;
  PosePrior pose_prior;
  EXPECT_FALSE(calibrator->Calibrate(bitmap, &camera, &pose_prior));
  EXPECT_EQ(camera.model_id, CameraModelId::kInvalid);
}

}  // namespace
}  // namespace colmap

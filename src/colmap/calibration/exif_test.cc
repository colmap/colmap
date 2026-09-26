// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/calibration/exif.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

MonocularCalibrationOptions ExifOptions() {
  return MonocularCalibrationOptions(MonocularCalibratorType::EXIF);
}

// Bare camera as written by the image reader: default focal length, centered
// principal point, no distortion, no focal length prior.
Camera CreateBareCamera() {
  Camera camera = Camera::CreateFromModelName(
      /*camera_id=*/1, "SIMPLE_RADIAL", /*focal_length=*/76.8, 64, 48);
  camera.has_prior_focal_length = false;
  return camera;
}

void SetExifFocalLength35mm(Bitmap& bitmap, float focal_length_35mm) {
  bitmap.SetMetaData("Exif:FocalLengthIn35mmFilm", "float", &focal_length_35mm);
}

void SetGpsExifTags(Bitmap& bitmap,
                    float latitude_deg,
                    float longitude_deg,
                    float altitude) {
  float latitude[3] = {latitude_deg, 0, 0};
  bitmap.SetMetaData("GPS:Latitude", "point", latitude);
  bitmap.SetMetaData("GPS:LatitudeRef", "N");
  float longitude[3] = {longitude_deg, 0, 0};
  bitmap.SetMetaData("GPS:Longitude", "point", longitude);
  bitmap.SetMetaData("GPS:LongitudeRef", "E");
  bitmap.SetMetaData("GPS:Altitude", "float", &altitude);
  bitmap.SetMetaData("GPS:AltitudeRef", "0");
}

TEST(ExifCalibratorTest, SetsExifFocalLength) {
  Bitmap bitmap(64, 48, /*as_rgb=*/true);
  // 43.27mm 35mm-equivalent on a 64x48 image (diagonal 80px) is 80px.
  SetExifFocalLength35mm(bitmap, 43.27f);
  Camera camera = CreateBareCamera();
  PosePrior pose_prior;
  const auto calibrator = MonocularCalibrator::Create(ExifOptions());
  ASSERT_TRUE(calibrator->Calibrate(bitmap, &camera, &pose_prior));
  // The 35mm-equivalent tag is stored as float, hence the tolerance.
  EXPECT_NEAR(camera.FocalLength(), 80.0, 1e-4);
  EXPECT_TRUE(camera.has_prior_focal_length);
  EXPECT_EQ(camera.ModelName(), "SIMPLE_RADIAL");
  EXPECT_FALSE(pose_prior.HasPosition());
  EXPECT_FALSE(pose_prior.HasGravity());
}

TEST(ExifCalibratorTest, PopulatesPosePrior) {
  Bitmap bitmap(64, 48, /*as_rgb=*/true);
  SetGpsExifTags(bitmap, 47.3769f, 8.5417f, 400.0f);
  int orientation = 6;
  bitmap.SetMetaData("Orientation", "int", &orientation);
  // No EXIF focal length: intrinsics calibration fails, but the pose prior
  // is populated independently.
  Camera camera = CreateBareCamera();
  PosePrior pose_prior;
  const auto calibrator = MonocularCalibrator::Create(ExifOptions());
  EXPECT_FALSE(calibrator->Calibrate(bitmap, &camera, &pose_prior));
  EXPECT_EQ(camera.params, CreateBareCamera().params);
  ASSERT_TRUE(pose_prior.HasPosition());
  EXPECT_DOUBLE_EQ(pose_prior.position.x(), 47.3769f);
  EXPECT_DOUBLE_EQ(pose_prior.position.y(), 8.5417f);
  EXPECT_DOUBLE_EQ(pose_prior.position.z(), 400.0f);
  EXPECT_EQ(pose_prior.coordinate_system, PosePrior::CoordinateSystem::WGS84);
  ASSERT_TRUE(pose_prior.HasGravity());
  EXPECT_EQ(pose_prior.gravity, Eigen::Vector3d(1, 0, 0));
}

TEST(ExifCalibratorTest, MissingExifKeepsCameraUnmodified) {
  const Bitmap bitmap(64, 48, /*as_rgb=*/true);
  Camera camera = CreateBareCamera();
  const Camera expected = camera;
  PosePrior pose_prior;
  const auto calibrator = MonocularCalibrator::Create(ExifOptions());
  EXPECT_FALSE(calibrator->Calibrate(bitmap, &camera, &pose_prior));
  EXPECT_EQ(camera.params, expected.params);
  EXPECT_EQ(camera.has_prior_focal_length, expected.has_prior_focal_length);
  EXPECT_FALSE(pose_prior.HasPosition());
  EXPECT_FALSE(pose_prior.HasGravity());
}

TEST(ExifCalibratorTest, RejectsBogusExifFocalLength) {
  Bitmap bitmap(64, 48, /*as_rgb=*/true);
  // 500mm 35mm-equivalent is ~924px on this image, far above the default
  // maximum focal length ratio of 10.
  SetExifFocalLength35mm(bitmap, 500.0f);
  Camera camera = CreateBareCamera();
  const Camera expected = camera;
  PosePrior pose_prior;
  const auto calibrator = MonocularCalibrator::Create(ExifOptions());
  EXPECT_FALSE(calibrator->Calibrate(bitmap, &camera, &pose_prior));
  EXPECT_EQ(camera.params, expected.params);
  EXPECT_EQ(camera.has_prior_focal_length, expected.has_prior_focal_length);
}

TEST(ExifCalibratorTest, RejectsModelConversion) {
  Bitmap bitmap(64, 48, /*as_rgb=*/true);
  SetExifFocalLength35mm(bitmap, 43.27f);
  Camera camera = CreateBareCamera();
  PosePrior pose_prior;
  auto options = ExifOptions();
  options.camera_model = "PINHOLE";
  const auto calibrator = MonocularCalibrator::Create(options);
  EXPECT_FALSE(calibrator->Calibrate(bitmap, &camera, &pose_prior));
}

TEST(ExifCalibratorTest, RejectsDimensionMismatch) {
  Bitmap bitmap(32, 24, /*as_rgb=*/true);
  SetExifFocalLength35mm(bitmap, 43.27f);
  Camera camera = CreateBareCamera();
  PosePrior pose_prior;
  const auto calibrator = MonocularCalibrator::Create(ExifOptions());
  EXPECT_FALSE(calibrator->Calibrate(bitmap, &camera, &pose_prior));
}

}  // namespace
}  // namespace colmap

// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/calibration/calibrator.h"

#include "colmap/calibration/anycalib.h"

#include <limits>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(MonocularCalibratorTypeTest, StringRoundTrip) {
  EXPECT_EQ(MonocularCalibratorTypeToString(MonocularCalibratorType::ANYCALIB),
            "ANYCALIB");
  EXPECT_EQ(MonocularCalibratorTypeFromString("ANYCALIB"),
            MonocularCalibratorType::ANYCALIB);
  EXPECT_EQ(MonocularCalibratorTypeToString(MonocularCalibratorType::EXIF),
            "EXIF");
  EXPECT_EQ(MonocularCalibratorTypeFromString("EXIF"),
            MonocularCalibratorType::EXIF);
}

TEST(MonocularCalibrationOptionsTest, CopyDeepCopiesTypeOptions) {
  MonocularCalibrationOptions options;
  ASSERT_NE(options.anycalib, nullptr);
  options.anycalib->model_path = "original";

  MonocularCalibrationOptions copy = options;
  ASSERT_NE(copy.anycalib, nullptr);
  EXPECT_NE(copy.anycalib, options.anycalib);
  EXPECT_EQ(copy.anycalib->model_path, "original");
  copy.anycalib->model_path = "modified";
  EXPECT_EQ(options.anycalib->model_path, "original");

  MonocularCalibrationOptions assigned;
  assigned = options;
  EXPECT_NE(assigned.anycalib, options.anycalib);
  EXPECT_EQ(assigned.anycalib->model_path, "original");
}

TEST(MonocularCalibrationOptionsTest, CheckValidatesAllFields) {
  MonocularCalibrationOptions options;
  EXPECT_TRUE(options.Check());

  options = MonocularCalibrationOptions();
  options.camera_model = "DOES_NOT_EXIST";
  EXPECT_FALSE(options.Check());

  options = MonocularCalibrationOptions();
  options.camera_model = "EQUIRECTANGULAR";
  EXPECT_FALSE(options.Check());

  options = MonocularCalibrationOptions();
  options.num_threads = -2;
  EXPECT_FALSE(options.Check());

  options = MonocularCalibrationOptions();
  options.min_focal_length_ratio = 0;
  EXPECT_FALSE(options.Check());

  options = MonocularCalibrationOptions();
  options.max_focal_length_ratio = options.min_focal_length_ratio;
  EXPECT_FALSE(options.Check());

  options = MonocularCalibrationOptions();
  options.max_extra_param = 0;
  EXPECT_FALSE(options.Check());

  // Backend-specific settings are only validated for the selected backend.
  options = MonocularCalibrationOptions();
  options.anycalib = nullptr;
  EXPECT_TRUE(options.Check());

  options = MonocularCalibrationOptions();
  options.type = MonocularCalibratorType::ANYCALIB;
  options.anycalib = nullptr;
  EXPECT_FALSE(options.Check());

  options = MonocularCalibrationOptions();
  options.type = MonocularCalibratorType::ANYCALIB;
  options.anycalib->fitting.max_num_points = 0;
  EXPECT_FALSE(options.Check());

  options = MonocularCalibrationOptions();
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  options.type = static_cast<MonocularCalibratorType>(-1);
  EXPECT_FALSE(options.Check());
}

TEST(MonocularCalibrationOptionsTest, MovePreservesTypeOptions) {
  MonocularCalibrationOptions options;
  options.anycalib->model_path = "original";

  MonocularCalibrationOptions moved = std::move(options);
  ASSERT_NE(moved.anycalib, nullptr);
  EXPECT_EQ(moved.anycalib->model_path, "original");

  MonocularCalibrationOptions assigned;
  assigned = std::move(moved);
  ASSERT_NE(assigned.anycalib, nullptr);
  EXPECT_EQ(assigned.anycalib->model_path, "original");
}

// The plausibility bounds match the defaults of the incremental mapper, so
// that intrinsics accepted here are not rejected during mapping.
TEST(MonocularCalibrationOptionsTest, DefaultsRejectImplausibleIntrinsics) {
  const MonocularCalibrationOptions options;
  Camera camera = Camera::CreateFromModelId(
      /*camera_id=*/1, CameraModelId::kSimpleRadial, 500, 640, 480);

  const auto has_bogus_params = [&options, &camera]() {
    return camera.HasBogusParams(options.min_focal_length_ratio,
                                 options.max_focal_length_ratio,
                                 options.max_extra_param);
  };

  EXPECT_FALSE(has_bogus_params());
  // Focal length far too short and far too long.
  camera.params[0] = 50;
  EXPECT_TRUE(has_bogus_params());
  camera.params[0] = 10000;
  EXPECT_TRUE(has_bogus_params());
  // Excessive distortion.
  camera.params[0] = 500;
  camera.params[3] = 5;
  EXPECT_TRUE(has_bogus_params());
}

TEST(IsValidCalibrationTest, RejectsInvalidCameras) {
  Camera camera = Camera::CreateFromModelId(
      /*camera_id=*/1, CameraModelId::kSimpleRadial, 500, 640, 480);
  EXPECT_TRUE(IsValidCalibration(camera));

  Camera zero_dims = camera;
  zero_dims.width = 0;
  EXPECT_FALSE(IsValidCalibration(zero_dims));
  zero_dims.width = 640;
  zero_dims.height = 0;
  EXPECT_FALSE(IsValidCalibration(zero_dims));

  Camera non_finite = camera;
  non_finite.params[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(IsValidCalibration(non_finite));

  Camera negative_focal = camera;
  negative_focal.params[0] = -500;
  EXPECT_FALSE(IsValidCalibration(negative_focal));

  // Diverging distortion fails the projection round-trip: the undistortion
  // iteration does not converge back to the input pixels.
  Camera diverging = Camera::CreateFromModelId(
      /*camera_id=*/1, CameraModelId::kOpenCV, 500, 640, 480);
  diverging.params = {500, 500, 320, 240, 2.5, -7.5, 0, 0};
  EXPECT_FALSE(IsValidCalibration(diverging));
}

Camera CreateSimpleRadialCamera() {
  return Camera::CreateFromModelId(
      /*camera_id=*/1, CameraModelId::kSimpleRadial, 500, 640, 480);
}

TEST(AggregateMonocularCalibrationsTest, EmptyList) {
  Camera camera = CreateSimpleRadialCamera();
  EXPECT_FALSE(AggregateMonocularCalibrations(
      CameraModelId::kSimpleRadial, {}, &camera));
  EXPECT_EQ(camera.params, std::vector<double>({500, 320, 240, 0}));
  EXPECT_FALSE(camera.has_prior_focal_length);
}

TEST(AggregateMonocularCalibrationsTest, OddMedian) {
  Camera camera = CreateSimpleRadialCamera();
  EXPECT_TRUE(AggregateMonocularCalibrations(
      CameraModelId::kSimpleRadial,
      {{600, 315, 235, 0.05}, {400, 325, 245, 0.15}, {500, 320, 240, 0.10}},
      &camera));
  EXPECT_EQ(camera.params, std::vector<double>({500, 320, 240, 0.10}));
  EXPECT_TRUE(camera.has_prior_focal_length);
}

TEST(AggregateMonocularCalibrationsTest, EvenMedian) {
  Camera camera = CreateSimpleRadialCamera();
  EXPECT_TRUE(AggregateMonocularCalibrations(
      CameraModelId::kSimpleRadial,
      {{600, 315, 235, 0.05}, {400, 325, 245, 0.15}},
      &camera));
  ASSERT_EQ(camera.params.size(), 4);
  EXPECT_DOUBLE_EQ(camera.params[0], 500);
  EXPECT_DOUBLE_EQ(camera.params[1], 320);
  EXPECT_DOUBLE_EQ(camera.params[2], 240);
  EXPECT_DOUBLE_EQ(camera.params[3], 0.10);
  EXPECT_TRUE(camera.has_prior_focal_length);
}

TEST(AggregateMonocularCalibrationsTest, SetsTargetCameraModel) {
  Camera camera = CreateSimpleRadialCamera();
  EXPECT_TRUE(AggregateMonocularCalibrations(
      CameraModelId::kPinhole, {{500, 500, 320, 240}}, &camera));
  EXPECT_EQ(camera.model_id, CameraModelId::kPinhole);
  EXPECT_EQ(camera.params, std::vector<double>({500, 500, 320, 240}));
}

TEST(AggregateMonocularCalibrationsTest, InvalidMedianFallsBackToClosest) {
  // Two individually well-behaved OPENCV calibrations whose coefficient-wise
  // median is not projection-stable, because the median breaks the correlation
  // between focal length and the radial coefficients.
  const std::vector<double> params1 = {200, 200, 320, 240, -3, 5, 0, 0};
  const std::vector<double> params2 = {800, 800, 320, 240, 8, -20, 0, 0};
  const std::vector<double> median = {500, 500, 320, 240, 2.5, -7.5, 0, 0};
  const CameraModelId model_id = CameraModelId::kOpenCV;

  Camera camera = Camera::CreateFromModelId(
      /*camera_id=*/1, model_id, 500, 640, 480);
  for (const std::vector<double>& params : {params1, params2, median}) {
    Camera single = camera;
    EXPECT_EQ(AggregateMonocularCalibrations(model_id, {params}, &single),
              params != median);
  }

  ASSERT_TRUE(
      AggregateMonocularCalibrations(model_id, {params1, params2}, &camera));
  // The two candidates are symmetric around the median, so their distances
  // tie exactly and `min_element` deterministically picks the first.
  EXPECT_EQ(camera.params, params1);
  EXPECT_TRUE(camera.has_prior_focal_length);
}

TEST(AggregateMonocularCalibrationsTest, RejectsRaggedParams) {
  Camera camera = CreateSimpleRadialCamera();
  EXPECT_THROW(AggregateMonocularCalibrations(CameraModelId::kSimpleRadial,
                                              {{500, 320, 240, 0}, {500, 320}},
                                              &camera),
               std::exception);
}

TEST(AggregateMonocularCalibrationsTest, RejectsInvalidParams) {
  const std::vector<double> original_params = {500, 320, 240, 0};
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const std::vector<double>& params :
       {std::vector<double>{nan, 320, 240, 0},
        std::vector<double>{-500, 320, 240, 0}}) {
    Camera camera = CreateSimpleRadialCamera();
    EXPECT_FALSE(AggregateMonocularCalibrations(
        CameraModelId::kSimpleRadial, {params}, &camera));
    EXPECT_EQ(camera.params, original_params);
    EXPECT_EQ(camera.model_id, CameraModelId::kSimpleRadial);
    EXPECT_FALSE(camera.has_prior_focal_length);
  }
}

TEST(MonocularCalibratorTest, MissingModelThrows) {
  MonocularCalibrationOptions options(MonocularCalibratorType::ANYCALIB);
  options.anycalib->model_path = "/nonexistent/anycalib_gen.onnx";
  EXPECT_THROW(MonocularCalibrator::Create(options), std::exception);
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

TEST(SetPosePriorFromExifTest, SetsPositionAndGravity) {
  Bitmap bitmap(64, 48, /*as_rgb=*/true);
  SetGpsExifTags(bitmap, 47.3769f, 8.5417f, 400.0f);
  int orientation = 1;
  bitmap.SetMetaData("Orientation", "int", &orientation);
  PosePrior pose_prior;
  SetPosePriorFromExif(bitmap, &pose_prior);
  ASSERT_TRUE(pose_prior.HasPosition());
  EXPECT_DOUBLE_EQ(pose_prior.position.x(), 47.3769f);
  EXPECT_DOUBLE_EQ(pose_prior.position.y(), 8.5417f);
  EXPECT_DOUBLE_EQ(pose_prior.position.z(), 400.0f);
  EXPECT_EQ(pose_prior.coordinate_system, PosePrior::CoordinateSystem::WGS84);
  ASSERT_TRUE(pose_prior.HasGravity());
  EXPECT_EQ(pose_prior.gravity, Eigen::Vector3d(0, 1, 0));
}

TEST(SetPosePriorFromExifTest, MissingTagsLeavePriorUntouched) {
  const Bitmap bitmap(64, 48, /*as_rgb=*/true);
  PosePrior pose_prior;
  SetPosePriorFromExif(bitmap, &pose_prior);
  EXPECT_FALSE(pose_prior.HasPosition());
  EXPECT_FALSE(pose_prior.HasGravity());
}

TEST(SetPosePriorFromExifTest, PartialGpsLeavesPositionUntouched) {
  Bitmap bitmap(64, 48, /*as_rgb=*/true);
  float latitude[3] = {47.3769f, 0, 0};
  bitmap.SetMetaData("GPS:Latitude", "point", latitude);
  bitmap.SetMetaData("GPS:LatitudeRef", "N");
  PosePrior pose_prior;
  SetPosePriorFromExif(bitmap, &pose_prior);
  EXPECT_FALSE(pose_prior.HasPosition());
  EXPECT_FALSE(pose_prior.HasGravity());
}

TEST(SetPosePriorFromExifTest, PresentValuesAreNeverOverwritten) {
  Bitmap bitmap(64, 48, /*as_rgb=*/true);
  SetGpsExifTags(bitmap, 47.3769f, 8.5417f, 400.0f);
  int orientation = 1;
  bitmap.SetMetaData("Orientation", "int", &orientation);
  PosePrior pose_prior;
  pose_prior.position = Eigen::Vector3d(1, 2, 3);
  pose_prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
  pose_prior.gravity = Eigen::Vector3d(0, 0, 1);
  SetPosePriorFromExif(bitmap, &pose_prior);
  EXPECT_EQ(pose_prior.position, Eigen::Vector3d(1, 2, 3));
  EXPECT_EQ(pose_prior.coordinate_system,
            PosePrior::CoordinateSystem::CARTESIAN);
  EXPECT_EQ(pose_prior.gravity, Eigen::Vector3d(0, 0, 1));
}

}  // namespace
}  // namespace colmap

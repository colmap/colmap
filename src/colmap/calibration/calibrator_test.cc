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

#include "colmap/calibration/calibrator.h"

#include "colmap/calibration/anycalib.h"

#include <limits>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(CameraCalibratorTypeTest, StringRoundTrip) {
  EXPECT_EQ(CameraCalibratorTypeToString(CameraCalibratorType::ANYCALIB),
            "ANYCALIB");
  EXPECT_EQ(CameraCalibratorTypeFromString("ANYCALIB"),
            CameraCalibratorType::ANYCALIB);
}

TEST(CameraCalibrationOptionsTest, CopyDeepCopiesTypeOptions) {
  CameraCalibrationOptions options;
  ASSERT_NE(options.anycalib, nullptr);
  options.anycalib->model_path = "original";

  CameraCalibrationOptions copy = options;
  ASSERT_NE(copy.anycalib, nullptr);
  EXPECT_NE(copy.anycalib, options.anycalib);
  EXPECT_EQ(copy.anycalib->model_path, "original");
  copy.anycalib->model_path = "modified";
  EXPECT_EQ(options.anycalib->model_path, "original");

  CameraCalibrationOptions assigned;
  assigned = options;
  EXPECT_NE(assigned.anycalib, options.anycalib);
  EXPECT_EQ(assigned.anycalib->model_path, "original");
}

TEST(CameraCalibrationOptionsTest, ChecksPlausibilityBounds) {
  CameraCalibrationOptions options;
  EXPECT_TRUE(options.Check());

  options = CameraCalibrationOptions();
  options.min_focal_length_ratio = 0;
  EXPECT_FALSE(options.Check());

  options = CameraCalibrationOptions();
  options.max_focal_length_ratio = options.min_focal_length_ratio;
  EXPECT_FALSE(options.Check());

  options = CameraCalibrationOptions();
  options.max_extra_param = 0;
  EXPECT_FALSE(options.Check());
}

// The plausibility bounds match the defaults of the incremental mapper, so
// that intrinsics accepted here are not rejected during mapping.
TEST(CameraCalibrationOptionsTest, DefaultsRejectImplausibleIntrinsics) {
  const CameraCalibrationOptions options;
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

Camera CreateSimpleRadialCamera() {
  return Camera::CreateFromModelId(
      /*camera_id=*/1, CameraModelId::kSimpleRadial, 500, 640, 480);
}

TEST(AggregateCameraCalibrationsTest, EmptyList) {
  Camera camera = CreateSimpleRadialCamera();
  EXPECT_FALSE(
      AggregateCameraCalibrations(CameraModelId::kSimpleRadial, {}, &camera));
  EXPECT_EQ(camera.params, std::vector<double>({500, 320, 240, 0}));
  EXPECT_FALSE(camera.has_prior_focal_length);
}

TEST(AggregateCameraCalibrationsTest, OddMedian) {
  Camera camera = CreateSimpleRadialCamera();
  EXPECT_TRUE(AggregateCameraCalibrations(
      CameraModelId::kSimpleRadial,
      {{600, 315, 235, 0.05}, {400, 325, 245, 0.15}, {500, 320, 240, 0.10}},
      &camera));
  EXPECT_EQ(camera.params, std::vector<double>({500, 320, 240, 0.10}));
  EXPECT_TRUE(camera.has_prior_focal_length);
}

TEST(AggregateCameraCalibrationsTest, EvenMedian) {
  Camera camera = CreateSimpleRadialCamera();
  EXPECT_TRUE(AggregateCameraCalibrations(
      CameraModelId::kSimpleRadial,
      {{600, 315, 235, 0.05}, {400, 325, 245, 0.15}},
      &camera));
  EXPECT_EQ(camera.params, std::vector<double>({500, 320, 240, 0.10}));
  EXPECT_TRUE(camera.has_prior_focal_length);
}

TEST(AggregateCameraCalibrationsTest, SetsTargetCameraModel) {
  Camera camera = CreateSimpleRadialCamera();
  EXPECT_TRUE(AggregateCameraCalibrations(
      CameraModelId::kPinhole, {{500, 500, 320, 240}}, &camera));
  EXPECT_EQ(camera.model_id, CameraModelId::kPinhole);
  EXPECT_EQ(camera.params, std::vector<double>({500, 500, 320, 240}));
}

TEST(AggregateCameraCalibrationsTest, InvalidMedianFallsBackToClosest) {
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
    EXPECT_EQ(AggregateCameraCalibrations(model_id, {params}, &single),
              params != median);
  }

  ASSERT_TRUE(
      AggregateCameraCalibrations(model_id, {params1, params2}, &camera));
  EXPECT_TRUE(camera.params == params1 || camera.params == params2);
  EXPECT_TRUE(camera.has_prior_focal_length);
}

TEST(AggregateCameraCalibrationsTest, RejectsInvalidParams) {
  const std::vector<double> original_params = {500, 320, 240, 0};
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const std::vector<double>& params :
       {std::vector<double>{nan, 320, 240, 0},
        std::vector<double>{-500, 320, 240, 0}}) {
    Camera camera = CreateSimpleRadialCamera();
    EXPECT_FALSE(AggregateCameraCalibrations(
        CameraModelId::kSimpleRadial, {params}, &camera));
    EXPECT_EQ(camera.params, original_params);
    EXPECT_EQ(camera.model_id, CameraModelId::kSimpleRadial);
    EXPECT_FALSE(camera.has_prior_focal_length);
  }
}

}  // namespace
}  // namespace colmap

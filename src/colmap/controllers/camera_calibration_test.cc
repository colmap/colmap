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

#include "colmap/controllers/camera_calibration.h"

#include "colmap/calibration/anycalib.h"
#include "colmap/math/random.h"
#include "colmap/scene/database.h"
#include "colmap/util/testing.h"

#include <cstdlib>

#include <gtest/gtest.h>

namespace colmap {
namespace {

// NOTE: The controller creates its calibrator lazily when it runs, mirroring
// the feature extraction and matching controllers, so that the network is not
// held in device memory during preceding pipeline stages. A missing model
// therefore surfaces when creating the calibrator, not the controller.
TEST(CameraCalibratorTest, MissingModelThrows) {
  CameraCalibrationOptions options;
  options.anycalib->model_path = "/nonexistent/anycalib_gen.onnx";
  EXPECT_THROW(CameraCalibrator::Create(options), std::exception);
}

TEST(CreateCameraCalibrationControllerTest, MissingModelDoesNotThrow) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  Database::Open(database_path);
  CameraCalibrationOptions options;
  options.anycalib->model_path = "/nonexistent/anycalib_gen.onnx";
  std::unique_ptr<Thread> controller;
  EXPECT_NO_THROW(controller = CreateCameraCalibrationController(
                      database_path, test_dir, options));
  // The calibrator is created in the worker thread, where an escaping
  // exception would terminate the process instead of failing the stage.
  ASSERT_NE(controller, nullptr);
  controller->Start();
  EXPECT_NO_THROW(controller->Wait());
}

// Full-controller integration test, run only when the exported model is
// available: COLMAP_ANYCALIB_MODEL_PATH=/path/to/anycalib_gen.onnx ctest -R
// camera_calibration_test. Uses random-noise images, which are not expected
// to calibrate; the test checks graceful handling end to end.
TEST(CameraCalibrationControllerTest, IntegrationTestWithModel) {
  const char* model_path = std::getenv("COLMAP_ANYCALIB_MODEL_PATH");
  if (model_path == nullptr) {
    GTEST_SKIP() << "Set COLMAP_ANYCALIB_MODEL_PATH to run this test";
  }

  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const std::vector<double> initial_params = {50, 32, 24, 0};
  {
    auto database = Database::Open(database_path);
    Camera camera;
    camera.model_id = CameraModelId::kSimpleRadial;
    camera.width = 64;
    camera.height = 48;
    camera.params = initial_params;
    camera.has_prior_focal_length = true;
    const camera_t camera_id = database->WriteCamera(camera);
    for (int i = 0; i < 2; ++i) {
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
      const std::string name = "image" + std::to_string(i) + ".png";
      ASSERT_TRUE(bitmap.Write(test_dir / name));
      Image image;
      image.SetName(name);
      image.SetCameraId(camera_id);
      database->WriteImage(image);
    }
  }

  CameraCalibrationOptions options;
  options.num_threads = 1;
  options.use_gpu = false;
  options.anycalib->model_path = model_path;
  Bitmap selected_bitmap;
  ASSERT_TRUE(selected_bitmap.Read(test_dir / "image0.png", /*as_rgb=*/true));
  Camera expected_camera;
  expected_camera.model_id = CameraModelId::kSimpleRadial;
  expected_camera.width = 64;
  expected_camera.height = 48;
  expected_camera.params = initial_params;
  expected_camera.has_prior_focal_length = true;
  const bool expected_success = CameraCalibrator::Create(options)->Calibrate(
      selected_bitmap, &expected_camera);

  auto controller = CreateCameraCalibrationController(
      database_path, test_dir, options, {"image0.png"});
  controller->Start();
  controller->Wait();

  auto database = Database::Open(database_path);
  const std::vector<Camera> cameras = database->ReadAllCameras();
  ASSERT_EQ(cameras.size(), 1);
  EXPECT_EQ(cameras[0].width, 64);
  EXPECT_EQ(cameras[0].height, 48);
  EXPECT_EQ(cameras[0].params,
            expected_success ? expected_camera.params : initial_params);
}

}  // namespace
}  // namespace colmap

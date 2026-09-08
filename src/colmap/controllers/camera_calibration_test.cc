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

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <vector>

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
  const std::vector<double> initial_params = {50, 32, 24, 0};
  {
    auto database = Database::Open(database_path);
    Camera camera;
    camera.model_id = CameraModelId::kSimpleRadial;
    camera.width = 64;
    camera.height = 48;
    camera.params = initial_params;
    const camera_t camera_id = database->WriteCamera(camera);
    Image image;
    image.SetName("image0.png");
    image.SetCameraId(camera_id);
    database->WriteImage(image);
  }
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

  // The failed stage must not have updated any camera.
  auto database = Database::Open(database_path);
  const std::vector<Camera> cameras = database->ReadAllCameras();
  ASSERT_EQ(cameras.size(), 1);
  EXPECT_EQ(cameras[0].params, initial_params);
}

void WriteSolidImage(const std::filesystem::path& path, int width, int height) {
  Bitmap bitmap(width, height, /*as_rgb=*/true);
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      bitmap.SetPixel(c, r, BitmapColor<uint8_t>(128, 128, 128));
    }
  }
  ASSERT_TRUE(bitmap.Write(path));
}

// Deterministic rendezvous to stop the controller while it is blocked inside
// `Calibrate`: the worker signals entry, the test stops the controller, then
// releases the worker.
struct BlockingState {
  std::promise<void> entered;
  std::promise<void> release;
  std::atomic<bool> entered_once = false;
};

class FakeCalibrator : public CameraCalibrator {
 public:
  struct Config {
    // Per-call parameters; the last entry repeats once exhausted.
    std::vector<std::vector<double>> params_sequence;
    bool success = true;
    bool throw_error = false;
    std::shared_ptr<BlockingState> block;
  };

  FakeCalibrator(Config config, CameraModelId model_id)
      : config_(std::move(config)), model_id_(model_id) {}

  bool Calibrate(const Bitmap& bitmap,
                 Camera* camera,
                 const PosePrior& /*pose_prior*/) const override {
    if (config_.block && !config_.block->entered_once.exchange(true)) {
      config_.block->entered.set_value();
      config_.block->release.get_future().wait();
    }
    if (config_.throw_error) {
      throw std::runtime_error("fake calibration failure");
    }
    if (!config_.success) {
      return false;
    }
    camera->model_id = model_id_;
    camera->width = bitmap.Width();
    camera->height = bitmap.Height();
    const size_t idx =
        std::min(num_calls_++, config_.params_sequence.size() - 1);
    camera->params = config_.params_sequence[idx];
    camera->has_prior_focal_length = true;
    return true;
  }

 private:
  Config config_;
  CameraModelId model_id_;
  mutable std::atomic<size_t> num_calls_ = 0;
};

CameraCalibratorFactory FakeCalibratorFactory(FakeCalibrator::Config config) {
  return [config = std::move(config)](const CameraCalibrationOptions& options) {
    return std::make_unique<FakeCalibrator>(
        config, CameraModelNameToId(options.camera_model));
  };
}

struct FakeCalibrationScene {
  std::filesystem::path test_dir;
  std::filesystem::path database_path;
  std::vector<double> initial_params;
};

// Two 64x48 images sharing one SIMPLE_RADIAL camera.
FakeCalibrationScene CreateFakeCalibrationScene() {
  FakeCalibrationScene scene;
  scene.test_dir = CreateTestDir();
  scene.database_path = scene.test_dir / "database.db";
  scene.initial_params = {50, 32, 24, 0};
  auto database = Database::Open(scene.database_path);
  Camera camera;
  camera.model_id = CameraModelId::kSimpleRadial;
  camera.width = 64;
  camera.height = 48;
  camera.params = scene.initial_params;
  camera.has_prior_focal_length = true;
  const camera_t camera_id = database->WriteCamera(camera);
  for (int i = 0; i < 2; ++i) {
    const std::string name = "image" + std::to_string(i) + ".png";
    WriteSolidImage(scene.test_dir / name, 64, 48);
    Image image;
    image.SetName(name);
    image.SetCameraId(camera_id);
    database->WriteImage(image);
  }
  return scene;
}

Camera ReadSingleCamera(const std::filesystem::path& database_path) {
  auto database = Database::Open(database_path);
  const std::vector<Camera> cameras = database->ReadAllCameras();
  if (cameras.size() != 1) {
    ADD_FAILURE() << "Expected exactly one camera, got " << cameras.size();
    return Camera();
  }
  return cameras[0];
}

void ExpectParamsNear(const std::vector<double>& actual,
                      const std::vector<double>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_DOUBLE_EQ(actual[i], expected[i]) << "param " << i;
  }
}

TEST(CameraCalibrationControllerTest, FakeCalibratorUpdatesDatabase) {
  const FakeCalibrationScene scene = CreateFakeCalibrationScene();
  FakeCalibrator::Config config;
  config.params_sequence = {{600, 32, 24, 0.05}, {400, 32, 24, 0.15}};

  CameraCalibrationOptions options;
  auto controller =
      CreateCameraCalibrationController(scene.database_path,
                                        scene.test_dir,
                                        options,
                                        {},
                                        FakeCalibratorFactory(config));
  controller->Start();
  controller->Wait();

  // The per-image fits are aggregated by coefficient-wise median.
  const Camera camera = ReadSingleCamera(scene.database_path);
  ExpectParamsNear(camera.params, {500, 32, 24, 0.10});
  EXPECT_TRUE(camera.has_prior_focal_length);
}

TEST(CameraCalibrationControllerTest, FailingCalibratorKeepsDatabase) {
  const FakeCalibrationScene scene = CreateFakeCalibrationScene();
  FakeCalibrator::Config config;
  config.params_sequence = {{600, 32, 24, 0.05}};
  config.success = false;

  CameraCalibrationOptions options;
  auto controller =
      CreateCameraCalibrationController(scene.database_path,
                                        scene.test_dir,
                                        options,
                                        {},
                                        FakeCalibratorFactory(config));
  controller->Start();
  controller->Wait();

  const Camera camera = ReadSingleCamera(scene.database_path);
  EXPECT_EQ(camera.params, scene.initial_params);
}

TEST(CameraCalibrationControllerTest, ThrowingCalibratorIsTolerated) {
  const FakeCalibrationScene scene = CreateFakeCalibrationScene();
  FakeCalibrator::Config config;
  config.params_sequence = {{600, 32, 24, 0.05}};
  config.throw_error = true;

  CameraCalibrationOptions options;
  auto controller =
      CreateCameraCalibrationController(scene.database_path,
                                        scene.test_dir,
                                        options,
                                        {},
                                        FakeCalibratorFactory(config));
  controller->Start();
  EXPECT_NO_THROW(controller->Wait());

  const Camera camera = ReadSingleCamera(scene.database_path);
  EXPECT_EQ(camera.params, scene.initial_params);
}

TEST(CameraCalibrationControllerTest, DimensionMismatchSkipsImage) {
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
    const camera_t camera_id = database->WriteCamera(camera);
    // The file is smaller than the database camera; its fit would live in the
    // wrong pixel frame and must be skipped.
    WriteSolidImage(test_dir / "image0.png", 32, 32);
    Image image;
    image.SetName("image0.png");
    image.SetCameraId(camera_id);
    database->WriteImage(image);
  }

  FakeCalibrator::Config config;
  config.params_sequence = {{600, 32, 24, 0.05}};
  CameraCalibrationOptions options;
  auto controller = CreateCameraCalibrationController(
      database_path, test_dir, options, {}, FakeCalibratorFactory(config));
  controller->Start();
  controller->Wait();

  const Camera camera = ReadSingleCamera(database_path);
  EXPECT_EQ(camera.params, initial_params);
}

TEST(CameraCalibrationControllerTest, UnknownImageNamesAreIgnored) {
  const FakeCalibrationScene scene = CreateFakeCalibrationScene();
  FakeCalibrator::Config config;
  config.params_sequence = {{600, 32, 24, 0.05}};

  CameraCalibrationOptions options;
  auto controller =
      CreateCameraCalibrationController(scene.database_path,
                                        scene.test_dir,
                                        options,
                                        {"image0.png", "missing.png"},
                                        FakeCalibratorFactory(config));
  controller->Start();
  controller->Wait();

  const Camera camera = ReadSingleCamera(scene.database_path);
  ExpectParamsNear(camera.params, {600, 32, 24, 0.05});
}

TEST(CameraCalibrationControllerTest, StopWhileCalibrating) {
  const FakeCalibrationScene scene = CreateFakeCalibrationScene();
  auto block = std::make_shared<BlockingState>();
  FakeCalibrator::Config config;
  config.params_sequence = {{600, 32, 24, 0.05}};
  config.block = block;

  CameraCalibrationOptions options;
  auto controller =
      CreateCameraCalibrationController(scene.database_path,
                                        scene.test_dir,
                                        options,
                                        {},
                                        FakeCalibratorFactory(config));
  controller->Start();
  // Wait until the worker is blocked inside `Calibrate`, stop it, then release
  // the worker so the run terminates deterministically.
  block->entered.get_future().wait();
  controller->Stop();
  block->release.set_value();
  EXPECT_NO_THROW(controller->Wait());

  // The interrupted run must not have updated any camera.
  const Camera camera = ReadSingleCamera(scene.database_path);
  EXPECT_EQ(camera.params, scene.initial_params);
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

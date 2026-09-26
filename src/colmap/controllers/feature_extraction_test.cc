// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/feature_extraction.h"

#include "colmap/calibration/anycalib.h"
#include "colmap/scene/database.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <functional>

#include <gtest/gtest.h>

namespace colmap {
namespace {

Bitmap CreateTestBitmap() {
  Bitmap bitmap(100, 100, /*as_rgb=*/false);
  bitmap.Fill(BitmapColor<uint8_t>(0));
  for (int y = 30; y < 70; ++y) {
    for (int x = 30; x < 70; ++x) {
      bitmap.SetPixel(x, y, BitmapColor<uint8_t>(255));
    }
  }
  return bitmap;
}

class FakeCalibrator : public MonocularCalibrator {
 public:
  struct Config {
    // Per-call parameters; the last entry repeats once exhausted.
    std::vector<std::vector<double>> params_sequence;
    bool success = true;
    bool throw_error = false;
    std::function<void()> on_calibrate;
  };

  FakeCalibrator(Config config, CameraModelId model_id)
      : config_(std::move(config)), model_id_(model_id) {}

  bool Calibrate(const Bitmap& bitmap,
                 Camera* camera,
                 PosePrior* /*pose_prior*/) const override {
    if (config_.throw_error) {
      throw std::runtime_error("fake calibration failure");
    }
    if (!config_.success) {
      return false;
    }
    // kInvalid preserves the input camera's model, mirroring the production
    // backends with an empty target model.
    if (model_id_ != CameraModelId::kInvalid) {
      camera->model_id = model_id_;
    }
    camera->width = bitmap.Width();
    camera->height = bitmap.Height();
    const size_t idx =
        std::min(num_calls_++, config_.params_sequence.size() - 1);
    camera->params = config_.params_sequence[idx];
    camera->has_prior_focal_length = true;
    if (config_.on_calibrate) {
      config_.on_calibrate();
    }
    return true;
  }

 private:
  Config config_;
  CameraModelId model_id_;
  mutable std::atomic<size_t> num_calls_ = 0;
};

MonocularCalibratorFactory FakeCalibratorFactory(
    FakeCalibrator::Config config) {
  return
      [config = std::move(config)](const MonocularCalibrationOptions& options) {
        return std::make_unique<FakeCalibrator>(
            config,
            options.camera_model.empty()
                ? CameraModelId::kInvalid
                : CameraModelNameToId(options.camera_model));
      };
}

void ExpectParamsNear(const std::vector<double>& actual,
                      const std::vector<double>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_DOUBLE_EQ(actual[i], expected[i]) << "param " << i;
  }
}

TEST(CreateFeatureExtractorController, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(image_path);

  // Create test images
  const int kNumImages = 2;
  const Bitmap test_bitmap = CreateTestBitmap();
  for (int i = 0; i < kNumImages; ++i) {
    test_bitmap.Write(image_path / (std::to_string(i) + ".png"));
  }

  // Set up options
  ImageReaderOptions reader_options;
  reader_options.image_path = image_path;

  FeatureExtractionOptions extraction_options;
  extraction_options.use_gpu = false;
  extraction_options.num_threads = kNumImages;

  // Create and run the controller
  auto controller =
      CreateFeatureExtractorController(database_path,
                                       reader_options,
                                       extraction_options,
                                       MonocularCalibrationOptions());
  ASSERT_NE(controller, nullptr);
  controller->Start();
  controller->Wait();

  // Verify results in database
  auto database = Database::Open(database_path);
  const std::vector<Image> images = database->ReadAllImages();
  EXPECT_EQ(images.size(), kNumImages);

  // The trailing EXIF calibration keeps the default intrinsics for the
  // EXIF-less test images.
  for (const Camera& camera : database->ReadAllCameras()) {
    EXPECT_DOUBLE_EQ(camera.FocalLength(), 1.2 * 100);
    EXPECT_FALSE(camera.has_prior_focal_length);
  }

  for (const auto& image : images) {
    EXPECT_TRUE(database->ExistsKeypoints(image.ImageId()));
    EXPECT_TRUE(database->ExistsDescriptors(image.ImageId()));

    const FeatureKeypoints keypoints = database->ReadKeypoints(image.ImageId());
    const FeatureDescriptors descriptors =
        database->ReadDescriptors(image.ImageId());

    // Check that features were extracted
    EXPECT_GT(keypoints.size(), 0);
    EXPECT_EQ(keypoints.size(), descriptors.data.rows());
    EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
    EXPECT_EQ(descriptors.data.cols(), 128);
  }
}

TEST(CreateFeatureExtractorController, WithCameraMask) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";
  const auto mask_path = test_dir / "mask.png";
  CreateDirIfNotExists(image_path);

  // Create test image with features
  const Bitmap test_bitmap = CreateTestBitmap();
  test_bitmap.Write(image_path / "test.png");

  // Create a mask that only allows the center region (white = keep, black =
  // mask) The test bitmap has a white square from (30,30) to (70,70) We'll
  // create a mask that only keeps a smaller region
  Bitmap mask_bitmap(100, 100, /*as_rgb=*/false);
  mask_bitmap.Fill(BitmapColor<uint8_t>(0));  // Start with all black (masked)

  // Only keep center region (40,40) to (60,60)
  for (int y = 40; y < 60; ++y) {
    for (int x = 40; x < 60; ++x) {
      mask_bitmap.SetPixel(x, y, BitmapColor<uint8_t>(255));  // White = keep
    }
  }
  mask_bitmap.Write(mask_path);

  // Extract features without mask first to get baseline
  ImageReaderOptions reader_options_no_mask;
  reader_options_no_mask.image_path = image_path;

  FeatureExtractionOptions extraction_options;
  extraction_options.use_gpu = false;
  extraction_options.num_threads = 1;

  auto controller =
      CreateFeatureExtractorController(database_path,
                                       reader_options_no_mask,
                                       extraction_options,
                                       MonocularCalibrationOptions());
  ASSERT_NE(controller, nullptr);
  controller->Start();
  controller->Wait();

  auto database = Database::Open(database_path);
  std::vector<Image> images = database->ReadAllImages();
  ASSERT_EQ(images.size(), 1);

  const size_t num_features_no_mask =
      database->ReadKeypoints(images[0].ImageId()).size();
  EXPECT_GT(num_features_no_mask, 0);

  // Now extract with mask
  const auto database_path_masked = test_dir / "database_masked.db";
  ImageReaderOptions reader_options_masked;
  reader_options_masked.image_path = image_path;
  reader_options_masked.camera_mask_path = mask_path;

  controller = CreateFeatureExtractorController(database_path_masked,
                                                reader_options_masked,
                                                extraction_options,
                                                MonocularCalibrationOptions());
  ASSERT_NE(controller, nullptr);
  controller->Start();
  controller->Wait();

  auto database_masked = Database::Open(database_path_masked);
  images = database_masked->ReadAllImages();
  ASSERT_EQ(images.size(), 1);

  const FeatureKeypoints keypoints_masked =
      database_masked->ReadKeypoints(images[0].ImageId());
  const FeatureDescriptors descriptors_masked =
      database_masked->ReadDescriptors(images[0].ImageId());
  const size_t num_features_masked = keypoints_masked.size();

  // With mask, should have fewer features
  EXPECT_LT(num_features_masked, num_features_no_mask);
  EXPECT_GT(num_features_masked, 0);  // But should still have some features

  // All remaining keypoints should be within the unmasked region (40-60, 40-60)
  for (const auto& kp : keypoints_masked) {
    EXPECT_GE(kp.x, 40.0f);
    EXPECT_LT(kp.x, 60.0f);
    EXPECT_GE(kp.y, 40.0f);
    EXPECT_LT(kp.y, 60.0f);
  }

  // Descriptors should match keypoints count
  EXPECT_EQ(descriptors_masked.data.rows(), keypoints_masked.size());
  EXPECT_EQ(descriptors_masked.type, FeatureExtractorType::SIFT);
  EXPECT_EQ(descriptors_masked.data.cols(), 128);
}

TEST(CreateFeatureImporterController, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";
  const auto import_path = test_dir / "features";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(import_path);

  const int kNumImages = 2;
  const int kNumFeatures = 3;

  // Create test images
  const Bitmap test_bitmap = CreateTestBitmap();
  for (int i = 0; i < kNumImages; ++i) {
    test_bitmap.Write(image_path / (std::to_string(i) + ".png"));
  }

  // Create feature text files for each image
  for (int i = 0; i < kNumImages; ++i) {
    const auto feature_file = import_path / (std::to_string(i) + ".png.txt");
    std::ofstream file(feature_file);
    ASSERT_TRUE(file.is_open());

    // Write header: num_features dimension
    const int kDimension = 128;
    file << kNumFeatures << " " << kDimension << "\n";

    // Write features: x y scale orientation descriptor[0..127]
    for (int j = 0; j < kNumFeatures; ++j) {
      // Keypoint data
      file << (10.0f + j * 5.0f) << " "  // x
           << (20.0f + j * 5.0f) << " "  // y
           << (1.5f + j * 0.1f) << " "   // scale
           << (0.5f + j * 0.2f);         // orientation

      // Descriptor data (128 values)
      for (int k = 0; k < kDimension; ++k) {
        file << " " << ((j * kDimension + k) % 256);
      }
      file << "\n";
    }
  }

  // Set up options
  ImageReaderOptions reader_options;
  reader_options.image_path = image_path;

  // Create and run the controller
  auto controller =
      CreateFeatureImporterController(database_path,
                                      reader_options,
                                      import_path,
                                      MonocularCalibrationOptions());
  ASSERT_NE(controller, nullptr);
  controller->Start();
  controller->Wait();

  // Verify results in database
  auto database = Database::Open(database_path);
  const std::vector<Image> images = database->ReadAllImages();
  EXPECT_EQ(images.size(), kNumImages);

  // The EXIF-less test images keep the default intrinsics.
  for (const Camera& camera : database->ReadAllCameras()) {
    EXPECT_DOUBLE_EQ(camera.FocalLength(), 1.2 * 100);
    EXPECT_FALSE(camera.has_prior_focal_length);
  }

  for (const auto& image : images) {
    EXPECT_TRUE(database->ExistsKeypoints(image.ImageId()));
    EXPECT_TRUE(database->ExistsDescriptors(image.ImageId()));

    const FeatureKeypoints keypoints = database->ReadKeypoints(image.ImageId());
    const FeatureDescriptors descriptors =
        database->ReadDescriptors(image.ImageId());

    // Check that features were imported correctly
    EXPECT_EQ(keypoints.size(), kNumFeatures);
    EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
    EXPECT_EQ(descriptors.data.rows(), kNumFeatures);
    EXPECT_EQ(descriptors.data.cols(), 128);

    // Verify some keypoint values
    EXPECT_FLOAT_EQ(keypoints[0].x, 10.0f);
    EXPECT_FLOAT_EQ(keypoints[0].y, 20.0f);
  }
}

struct FakeCalibrationScene {
  std::filesystem::path test_dir;
  std::filesystem::path database_path;
  std::filesystem::path image_path;
};

// Two 100x100 images sharing one camera (single_camera=true).
FakeCalibrationScene CreateFakeCalibrationScene() {
  FakeCalibrationScene scene;
  scene.test_dir = CreateTestDir();
  scene.database_path = scene.test_dir / "database.db";
  scene.image_path = scene.test_dir / "images";
  CreateDirIfNotExists(scene.image_path);
  const Bitmap test_bitmap = CreateTestBitmap();
  test_bitmap.Write(scene.image_path / "0.png");
  test_bitmap.Write(scene.image_path / "1.png");
  return scene;
}

TEST(CreateFeatureImporterController, StopPersistsCalibration) {
  const FakeCalibrationScene scene = CreateFakeCalibrationScene();
  const auto import_path = scene.test_dir / "features";
  CreateDirIfNotExists(import_path);

  ImageReaderOptions reader_options;
  reader_options.image_path = scene.image_path;
  reader_options.single_camera = true;

  FakeCalibrator::Config config;
  config.params_sequence = {{500, 50, 50, 0.1}};
  Thread* controller_ptr = nullptr;
  config.on_calibrate = [&controller_ptr]() { controller_ptr->Stop(); };
  MonocularCalibrationOptions calibration_options(
      MonocularCalibratorType::ANYCALIB);

  auto controller =
      CreateFeatureImporterController(scene.database_path,
                                      reader_options,
                                      import_path,
                                      calibration_options,
                                      FakeCalibratorFactory(std::move(config)));
  controller_ptr = controller.get();
  controller->Start();
  controller->Wait();

  auto database = Database::Open(scene.database_path);
  const std::vector<Camera> cameras = database->ReadAllCameras();
  ASSERT_EQ(cameras.size(), 1);
  ExpectParamsNear(cameras[0].params, {500, 50, 50, 0.1});
  EXPECT_TRUE(cameras[0].has_prior_focal_length);
}

TEST(CreateFeatureExtractorController, StopPersistsCalibration) {
  const FakeCalibrationScene scene = CreateFakeCalibrationScene();

  ImageReaderOptions reader_options;
  reader_options.image_path = scene.image_path;
  reader_options.single_camera = true;

  FeatureExtractionOptions extraction_options;
  extraction_options.use_gpu = false;
  extraction_options.num_threads = 1;

  FakeCalibrator::Config config;
  config.params_sequence = {{500, 50, 50, 0.1}};
  Thread* controller_ptr = nullptr;
  config.on_calibrate = [&controller_ptr]() { controller_ptr->Stop(); };
  MonocularCalibrationOptions calibration_options(
      MonocularCalibratorType::ANYCALIB);

  auto controller = CreateFeatureExtractorController(
      scene.database_path,
      reader_options,
      extraction_options,
      calibration_options,
      FakeCalibratorFactory(std::move(config)));
  controller_ptr = controller.get();
  controller->Start();
  controller->Wait();

  auto database = Database::Open(scene.database_path);
  const std::vector<Camera> cameras = database->ReadAllCameras();
  ASSERT_EQ(cameras.size(), 1);
  ExpectParamsNear(cameras[0].params, {500, 50, 50, 0.1});
  EXPECT_TRUE(cameras[0].has_prior_focal_length);
}

TEST(CreateFeatureExtractorController, FakeBackendAggregatesMedian) {
  const FakeCalibrationScene scene = CreateFakeCalibrationScene();

  ImageReaderOptions reader_options;
  reader_options.image_path = scene.image_path;
  reader_options.single_camera = true;

  FeatureExtractionOptions extraction_options;
  extraction_options.use_gpu = false;
  extraction_options.num_threads = 1;

  FakeCalibrator::Config config;
  config.params_sequence = {{600, 50, 50, 0.05}, {400, 50, 50, 0.15}};
  MonocularCalibrationOptions calibration_options(
      MonocularCalibratorType::ANYCALIB);

  auto controller =
      CreateFeatureExtractorController(scene.database_path,
                                       reader_options,
                                       extraction_options,
                                       calibration_options,
                                       FakeCalibratorFactory(config));
  controller->Start();
  controller->Wait();

  // The per-image fits are aggregated by coefficient-wise median.
  auto database = Database::Open(scene.database_path);
  const std::vector<Camera> cameras = database->ReadAllCameras();
  ASSERT_EQ(cameras.size(), 1);
  ExpectParamsNear(cameras[0].params, {500, 50, 50, 0.10});
  EXPECT_TRUE(cameras[0].has_prior_focal_length);
}

TEST(CreateFeatureExtractorController, FakeBackendConvertsModel) {
  const FakeCalibrationScene scene = CreateFakeCalibrationScene();

  ImageReaderOptions reader_options;
  reader_options.image_path = scene.image_path;
  reader_options.single_camera = true;
  reader_options.camera_model = "OPENCV";

  FeatureExtractionOptions extraction_options;
  extraction_options.use_gpu = false;
  extraction_options.num_threads = 1;

  FakeCalibrator::Config config;
  config.params_sequence = {{600, 50, 50, 0.05}, {400, 50, 50, 0.15}};
  MonocularCalibrationOptions calibration_options(
      MonocularCalibratorType::ANYCALIB);
  calibration_options.camera_model = "SIMPLE_RADIAL";

  auto controller =
      CreateFeatureExtractorController(scene.database_path,
                                       reader_options,
                                       extraction_options,
                                       calibration_options,
                                       FakeCalibratorFactory(config));
  controller->Start();
  controller->Wait();

  auto database = Database::Open(scene.database_path);
  const std::vector<Camera> cameras = database->ReadAllCameras();
  ASSERT_EQ(cameras.size(), 1);
  EXPECT_EQ(cameras[0].model_id, CameraModelId::kSimpleRadial);
  ExpectParamsNear(cameras[0].params, {500, 50, 50, 0.10});
}

TEST(CreateFeatureExtractorController, FailingBackendKeepsDefaults) {
  const FakeCalibrationScene scene = CreateFakeCalibrationScene();

  ImageReaderOptions reader_options;
  reader_options.image_path = scene.image_path;
  reader_options.single_camera = true;

  FeatureExtractionOptions extraction_options;
  extraction_options.use_gpu = false;
  extraction_options.num_threads = 1;

  FakeCalibrator::Config config;
  config.params_sequence = {{600, 50, 50, 0.05}};
  config.success = false;
  MonocularCalibrationOptions calibration_options(
      MonocularCalibratorType::ANYCALIB);

  auto controller =
      CreateFeatureExtractorController(scene.database_path,
                                       reader_options,
                                       extraction_options,
                                       calibration_options,
                                       FakeCalibratorFactory(config));
  controller->Start();
  controller->Wait();

  auto database = Database::Open(scene.database_path);
  const std::vector<Camera> cameras = database->ReadAllCameras();
  ASSERT_EQ(cameras.size(), 1);
  EXPECT_DOUBLE_EQ(cameras[0].FocalLength(), 1.2 * 100);
  EXPECT_FALSE(cameras[0].has_prior_focal_length);
}

TEST(CreateFeatureExtractorController, MissingModelKeepsDefaults) {
  const FakeCalibrationScene scene = CreateFakeCalibrationScene();

  ImageReaderOptions reader_options;
  reader_options.image_path = scene.image_path;
  reader_options.single_camera = true;

  FeatureExtractionOptions extraction_options;
  extraction_options.use_gpu = false;
  extraction_options.num_threads = 1;

  MonocularCalibrationOptions calibration_options(
      MonocularCalibratorType::ANYCALIB);
  calibration_options.anycalib->model_path = "/nonexistent/anycalib_gen.onnx";

  auto controller = CreateFeatureExtractorController(scene.database_path,
                                                     reader_options,
                                                     extraction_options,
                                                     calibration_options);
  EXPECT_NO_THROW({
    controller->Start();
    controller->Wait();
  });

  // Extraction completes and the cameras keep their default intrinsics.
  auto database = Database::Open(scene.database_path);
  EXPECT_EQ(database->ReadAllImages().size(), 2);
  const std::vector<Camera> cameras = database->ReadAllCameras();
  ASSERT_EQ(cameras.size(), 1);
  EXPECT_DOUBLE_EQ(cameras[0].FocalLength(), 1.2 * 100);
  EXPECT_FALSE(cameras[0].has_prior_focal_length);
}

TEST(CreateFeatureExtractorController, ExplicitParamsSkipBackend) {
  const FakeCalibrationScene scene = CreateFakeCalibrationScene();

  ImageReaderOptions reader_options;
  reader_options.image_path = scene.image_path;
  reader_options.single_camera = true;
  reader_options.camera_model = "PINHOLE";
  reader_options.camera_params = "500.0, 500.0, 50.0, 50.0";

  FeatureExtractionOptions extraction_options;
  extraction_options.use_gpu = false;
  extraction_options.num_threads = 1;

  // The backend must never be created when explicit parameters are provided.
  bool factory_called = false;
  MonocularCalibratorFactory factory =
      [&factory_called](const MonocularCalibrationOptions& options) {
        factory_called = true;
        return MonocularCalibrator::Create(options);
      };

  auto controller =
      CreateFeatureExtractorController(scene.database_path,
                                       reader_options,
                                       extraction_options,
                                       MonocularCalibrationOptions(),
                                       std::move(factory));
  controller->Start();
  controller->Wait();

  EXPECT_FALSE(factory_called);
  auto database = Database::Open(scene.database_path);
  const std::vector<Camera> cameras = database->ReadAllCameras();
  ASSERT_EQ(cameras.size(), 1);
  EXPECT_EQ(cameras[0].params, std::vector<double>({500.0, 500.0, 50.0, 50.0}));
}

}  // namespace
}  // namespace colmap

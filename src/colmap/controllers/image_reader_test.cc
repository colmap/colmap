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

#include "colmap/controllers/image_reader.h"

#include "colmap/scene/database_sqlite.h"
#include "colmap/sensor/models.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <fstream>
#include <tuple>

#include <gtest/gtest.h>

namespace colmap {
namespace {

Bitmap CreateTestBitmap(bool as_rgb) {
  Bitmap bitmap(1, 3, as_rgb);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(1));
  bitmap.SetPixel(1, 0, BitmapColor<uint8_t>(2));
  bitmap.SetPixel(2, 0, BitmapColor<uint8_t>(3));
  return bitmap;
}

class ParameterizedImageReaderTests
    : public ::testing::TestWithParam<std::tuple</*num_images=*/int,
                                                 /*with_masks=*/bool,
                                                 /*with_existing_images=*/bool,
                                                 /*as_rgb=*/bool,
                                                 /*extension=*/std::string>> {};

TEST_P(ParameterizedImageReaderTests, Nominal) {
  const auto [kNumImages, kWithMasks, kWithExistingImages, kAsRGB, kExtension] =
      GetParam();

  auto database = Database::Open(kInMemorySqliteDatabasePath);

  const auto test_dir = CreateTestDir();
  ImageReaderOptions options;
  options.image_path = test_dir / "images";
  options.as_rgb = kAsRGB;
  CreateDirIfNotExists(options.image_path);
  if (kWithMasks) {
    options.mask_path = test_dir / "masks";
    CreateDirIfNotExists(options.mask_path);
  }
  const Bitmap test_bitmap = CreateTestBitmap(kAsRGB);
  for (int i = 0; i < kNumImages; ++i) {
    const std::string stem = std::to_string(i);
    const std::string image_name = stem + kExtension;
    test_bitmap.Write(options.image_path / image_name);
    if (kWithMasks) {
      if (i == 0) {
        // append .png to image_name
        test_bitmap.Write(options.mask_path / (image_name + ".png"));
      } else {
        // replace mask extension by .png
        test_bitmap.Write(options.mask_path / (stem + ".png"));
      }
    }
    if (kWithExistingImages) {
      Image image;
      image.SetName(image_name);
      image.SetCameraId(database->WriteCamera(
          Camera::CreateFromModelName(i + 1,
                                      options.camera_model,
                                      /*focal_length=*/1,
                                      test_bitmap.Width(),
                                      test_bitmap.Height())));
      image.SetImageId(database->WriteImage(image));
      database->WriteKeypoints(image.ImageId(), FeatureKeypoints());
      database->WriteDescriptors(image.ImageId(), FeatureDescriptors());
      Rig rig;
      rig.AddRefSensor(sensor_t(SensorType::CAMERA, image.CameraId()));
      database->WriteRig(rig);
    }
  }

  ImageReader image_reader(options, database.get());
  EXPECT_EQ(image_reader.NumImages(), kNumImages);

  Rig rig;
  Camera camera;
  Image image;
  PosePrior pose_prior;
  Bitmap bitmap;
  Bitmap mask;
  for (int i = 0; i < kNumImages; ++i) {
    EXPECT_EQ(image_reader.NextIndex(), i);
    const auto status =
        image_reader.Next(&rig, &camera, &image, &pose_prior, &bitmap, &mask);
    if (kWithExistingImages) {
      EXPECT_EQ(status, ImageReader::Status::IMAGE_EXISTS);
      continue;
    }
    ASSERT_EQ(status, ImageReader::Status::SUCCESS);
    EXPECT_EQ(rig.RigId(), i + 1);
    EXPECT_EQ(camera.camera_id, i + 1);
    EXPECT_EQ(camera.ModelName(), options.camera_model);
    EXPECT_EQ(camera.width, test_bitmap.Width());
    EXPECT_EQ(camera.height, test_bitmap.Height());
    EXPECT_EQ(image.Name(), std::to_string(i) + kExtension);
    EXPECT_EQ(bitmap.IsRGB(), kAsRGB);
    EXPECT_EQ(bitmap.RowMajorData(), test_bitmap.RowMajorData());
    if (kWithExistingImages) {
      EXPECT_EQ(database->NumRigs(), kNumImages);
      EXPECT_EQ(database->NumCameras(), kNumImages);
    } else {
      EXPECT_EQ(database->NumRigs(), i + 1);
      EXPECT_EQ(database->NumCameras(), i + 1);
    }
  }

  EXPECT_THROW(
      image_reader.Next(&rig, &camera, &image, &pose_prior, &bitmap, &mask),
      std::invalid_argument);
  EXPECT_EQ(database->NumRigs(), kNumImages);
  EXPECT_EQ(database->NumCameras(), kNumImages);
}

INSTANTIATE_TEST_SUITE_P(
    ImageReaderTests,
    ParameterizedImageReaderTests,
    ::testing::Values(std::make_tuple(/*num_images=*/0,
                                      /*with_masks=*/false,
                                      /*with_existing_images=*/true,
                                      /*as_rgb=*/true,
                                      /*extension=*/".png"),
                      std::make_tuple(/*num_images=*/5,
                                      /*with_masks=*/false,
                                      /*with_existing_images=*/false,
                                      /*as_rgb=*/true,
                                      /*extension=*/".png"),
                      std::make_tuple(/*num_images=*/5,
                                      /*with_masks=*/true,
                                      /*with_existing_images=*/false,
                                      /*as_rgb=*/true,
                                      /*extension=*/".png"),
                      std::make_tuple(/*num_images=*/5,
                                      /*with_masks=*/true,
                                      /*with_existing_images=*/false,
                                      /*as_rgb=*/true,
                                      /*extension=*/".bmp"),
                      std::make_tuple(/*num_images=*/5,
                                      /*with_masks=*/true,
                                      /*with_existing_images=*/false,
                                      /*as_rgb=*/false,
                                      /*extension=*/".png"),
                      std::make_tuple(/*num_images=*/5,
                                      /*with_masks=*/false,
                                      /*with_existing_images=*/true,
                                      /*as_rgb=*/true,
                                      /*extension=*/".png")));

TEST(ImageReaderTest, SingleCamera) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  const auto test_dir = CreateTestDir();

  ImageReaderOptions options;
  options.image_path = test_dir / "images";
  options.single_camera = true;
  CreateDirIfNotExists(options.image_path);

  // Create 3 test images with same dimensions
  Bitmap test_bitmap(10, 20, true);
  test_bitmap.Write(options.image_path / "0.png");
  test_bitmap.Write(options.image_path / "1.png");
  test_bitmap.Write(options.image_path / "2.png");

  ImageReader image_reader(options, database.get());
  EXPECT_EQ(image_reader.NumImages(), 3);

  Rig rig;
  Camera camera;
  Image image;
  PosePrior pose_prior;
  Bitmap bitmap;
  Bitmap mask;

  for (int i = 0; i < 3; ++i) {
    const auto status =
        image_reader.Next(&rig, &camera, &image, &pose_prior, &bitmap, &mask);
    ASSERT_EQ(status, ImageReader::Status::SUCCESS);
  }

  EXPECT_EQ(database->NumRigs(), 1);
  EXPECT_EQ(database->NumCameras(), 1);
}

TEST(ImageReaderTest, SingleCameraDimensionError) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  const auto test_dir = CreateTestDir();

  ImageReaderOptions options;
  options.image_path = test_dir / "images";
  options.single_camera = true;
  CreateDirIfNotExists(options.image_path);

  // Create images with different dimensions
  Bitmap bitmap1(10, 20, true);
  Bitmap bitmap2(30, 40, true);
  bitmap1.Write(options.image_path / "0.png");
  bitmap2.Write(options.image_path / "1.png");

  ImageReader image_reader(options, database.get());

  Rig rig;
  Camera camera;
  Image image;
  PosePrior pose_prior;
  Bitmap bitmap;
  Bitmap mask;

  // First image succeeds
  auto status =
      image_reader.Next(&rig, &camera, &image, &pose_prior, &bitmap, &mask);
  ASSERT_EQ(status, ImageReader::Status::SUCCESS);

  // Second image fails due to dimension mismatch
  status =
      image_reader.Next(&rig, &camera, &image, &pose_prior, &bitmap, &mask);
  EXPECT_EQ(status, ImageReader::Status::CAMERA_SINGLE_DIM_ERROR);
}

TEST(ImageReaderTest, SingleCameraPerFolder) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  const auto test_dir = CreateTestDir();

  ImageReaderOptions options;
  options.image_path = test_dir / "images";
  options.single_camera_per_folder = true;
  CreateDirIfNotExists(options.image_path);
  CreateDirIfNotExists(options.image_path / "folder1");
  CreateDirIfNotExists(options.image_path / "folder2");

  // Create 2 images in each folder
  Bitmap test_bitmap(10, 20, true);
  test_bitmap.Write(options.image_path / "folder1" / "0.png");
  test_bitmap.Write(options.image_path / "folder1" / "1.png");
  test_bitmap.Write(options.image_path / "folder2" / "0.png");
  test_bitmap.Write(options.image_path / "folder2" / "1.png");

  ImageReader image_reader(options, database.get());
  EXPECT_EQ(image_reader.NumImages(), 4);

  Rig rig;
  Camera camera;
  Image image;
  PosePrior pose_prior;
  Bitmap bitmap;
  Bitmap mask;

  std::unordered_map<std::string, camera_t> folder_cameras;
  for (int i = 0; i < 4; ++i) {
    const auto status =
        image_reader.Next(&rig, &camera, &image, &pose_prior, &bitmap, &mask);
    ASSERT_EQ(status, ImageReader::Status::SUCCESS);
    const std::string folder = GetParentDir(image.Name()).string();
    if (folder_cameras.count(folder) == 0) {
      folder_cameras[folder] = camera.camera_id;
    } else {
      EXPECT_EQ(camera.camera_id, folder_cameras[folder]);
    }
  }

  // Should have 2 cameras (one per folder)
  EXPECT_EQ(database->NumCameras(), 2);
}

TEST(ImageReaderTest, SingleCameraPerImage) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  const auto test_dir = CreateTestDir();

  ImageReaderOptions options;
  options.image_path = test_dir / "images";
  options.single_camera_per_image = true;
  CreateDirIfNotExists(options.image_path);

  // Create 3 images with same dimensions
  Bitmap test_bitmap(10, 20, true);
  test_bitmap.Write(options.image_path / "0.png");
  test_bitmap.Write(options.image_path / "1.png");
  test_bitmap.Write(options.image_path / "2.png");

  ImageReader image_reader(options, database.get());

  Rig rig;
  Camera camera;
  Image image;
  PosePrior pose_prior;
  Bitmap bitmap;
  Bitmap mask;

  for (int i = 0; i < 3; ++i) {
    const auto status =
        image_reader.Next(&rig, &camera, &image, &pose_prior, &bitmap, &mask);
    ASSERT_EQ(status, ImageReader::Status::SUCCESS);
    EXPECT_EQ(camera.camera_id, i + 1);  // Each image gets its own camera
  }

  // Should have 3 cameras (one per image)
  EXPECT_EQ(database->NumCameras(), 3);
}

TEST(ImageReaderTest, ExistingCameraId) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  const auto test_dir = CreateTestDir();

  // Create an existing camera in the database
  Camera existing_camera;
  existing_camera.model_id = CameraModelNameToId("SIMPLE_RADIAL");
  existing_camera.width = 10;
  existing_camera.height = 20;
  existing_camera.params = {1.0, 5.0, 10.0, 0.0};
  existing_camera.camera_id = database->WriteCamera(existing_camera);
  Rig existing_rig;
  existing_rig.AddRefSensor(existing_camera.SensorId());
  database->WriteRig(existing_rig);

  ImageReaderOptions options;
  options.image_path = test_dir / "images";
  options.existing_camera_id = existing_camera.camera_id;
  CreateDirIfNotExists(options.image_path);

  Bitmap test_bitmap(10, 20, true);
  test_bitmap.Write(options.image_path / "0.png");
  test_bitmap.Write(options.image_path / "1.png");

  ImageReader image_reader(options, database.get());

  Rig rig;
  Camera camera;
  Image image;
  PosePrior pose_prior;
  Bitmap bitmap;
  Bitmap mask;

  for (int i = 0; i < 2; ++i) {
    const auto status =
        image_reader.Next(&rig, &camera, &image, &pose_prior, &bitmap, &mask);
    ASSERT_EQ(status, ImageReader::Status::SUCCESS);
    EXPECT_EQ(camera.camera_id, existing_camera.camera_id);
    EXPECT_EQ(camera.params, existing_camera.params);
  }

  // No new cameras created
  EXPECT_EQ(database->NumCameras(), 1);
}

TEST(ImageReaderTest, ManualCameraParams) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  const auto test_dir = CreateTestDir();

  ImageReaderOptions options;
  options.image_path = test_dir / "images";
  options.camera_model = "PINHOLE";
  options.camera_params = "500.0, 500.0, 320.0, 240.0";
  CreateDirIfNotExists(options.image_path);

  Bitmap test_bitmap(640, 480, true);
  test_bitmap.Write(options.image_path / "test.png");

  ImageReader image_reader(options, database.get());

  Rig rig;
  Camera camera;
  Image image;
  PosePrior pose_prior;
  Bitmap bitmap;
  Bitmap mask;

  const auto status =
      image_reader.Next(&rig, &camera, &image, &pose_prior, &bitmap, &mask);
  ASSERT_EQ(status, ImageReader::Status::SUCCESS);
  EXPECT_EQ(camera.model_id, PinholeCameraModel::model_id);
  EXPECT_EQ(camera.params[0], 500.0);
  EXPECT_EQ(camera.params[1], 500.0);
  EXPECT_EQ(camera.params[2], 320.0);
  EXPECT_EQ(camera.params[3], 240.0);
  EXPECT_TRUE(camera.has_prior_focal_length);
}

TEST(ImageReaderTest, ExplicitImageNames) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  const auto test_dir = CreateTestDir();

  ImageReaderOptions options;
  options.image_path = test_dir / "images";
  CreateDirIfNotExists(options.image_path);

  // Create 5 images
  Bitmap test_bitmap(10, 20, true);
  for (int i = 0; i < 5; ++i) {
    test_bitmap.Write(options.image_path / (std::to_string(i) + ".png"));
  }

  // Only select a subset of images
  options.image_names = {"1.png", "3.png"};

  ImageReader image_reader(options, database.get());
  EXPECT_EQ(image_reader.NumImages(), 2);

  Rig rig;
  Camera camera;
  Image image;
  PosePrior pose_prior;
  Bitmap bitmap;
  Bitmap mask;

  auto status =
      image_reader.Next(&rig, &camera, &image, &pose_prior, &bitmap, &mask);
  ASSERT_EQ(status, ImageReader::Status::SUCCESS);
  EXPECT_EQ(image.Name(), "1.png");

  status =
      image_reader.Next(&rig, &camera, &image, &pose_prior, &bitmap, &mask);
  ASSERT_EQ(status, ImageReader::Status::SUCCESS);
  EXPECT_EQ(image.Name(), "3.png");
}

TEST(ImageReaderTest, BitmapError) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  const auto test_dir = CreateTestDir();

  ImageReaderOptions options;
  options.image_path = test_dir / "images";
  CreateDirIfNotExists(options.image_path);

  // Create a file that is not a valid image
  std::ofstream file(options.image_path / "invalid.png");
  file << "not an image";
  file.close();

  ImageReader image_reader(options, database.get());

  Rig rig;
  Camera camera;
  Image image;
  PosePrior pose_prior;
  Bitmap bitmap;
  Bitmap mask;

  const auto status =
      image_reader.Next(&rig, &camera, &image, &pose_prior, &bitmap, &mask);
  EXPECT_EQ(status, ImageReader::Status::BITMAP_ERROR);
}

TEST(ImageReaderTest, MaskErrorMissing) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  const auto test_dir = CreateTestDir();

  ImageReaderOptions options;
  options.image_path = test_dir / "images";
  options.mask_path = test_dir / "masks";
  CreateDirIfNotExists(options.image_path);
  CreateDirIfNotExists(options.mask_path);

  Bitmap test_bitmap(10, 20, true);
  test_bitmap.Write(options.image_path / "test.png");
  // Don't create mask file

  ImageReader image_reader(options, database.get());

  Rig rig;
  Camera camera;
  Image image;
  PosePrior pose_prior;
  Bitmap bitmap;
  Bitmap mask;

  const auto status =
      image_reader.Next(&rig, &camera, &image, &pose_prior, &bitmap, &mask);
  EXPECT_EQ(status, ImageReader::Status::MASK_ERROR);
}

TEST(ImageReaderTest, MaskErrorInvalid) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  const auto test_dir = CreateTestDir();

  ImageReaderOptions options;
  options.image_path = test_dir / "images";
  options.mask_path = test_dir / "masks";
  CreateDirIfNotExists(options.image_path);
  CreateDirIfNotExists(options.mask_path);

  Bitmap test_bitmap(10, 20, true);
  test_bitmap.Write(options.image_path / "test.png");

  // Create invalid mask file
  std::ofstream file(options.mask_path / "test.png.png");
  file << "not an image";
  file.close();

  ImageReader image_reader(options, database.get());

  Rig rig;
  Camera camera;
  Image image;
  PosePrior pose_prior;
  Bitmap bitmap;
  Bitmap mask;

  const auto status =
      image_reader.Next(&rig, &camera, &image, &pose_prior, &bitmap, &mask);
  EXPECT_EQ(status, ImageReader::Status::MASK_ERROR);
}

TEST(ImageReaderTest, ImageExistsWithKeypoints) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  const auto test_dir = CreateTestDir();

  ImageReaderOptions options;
  options.image_path = test_dir / "images";
  CreateDirIfNotExists(options.image_path);

  Bitmap test_bitmap(10, 20, true);
  test_bitmap.Write(options.image_path / "test.png");

  // Add existing image with keypoints and descriptors
  Camera existing_camera;
  existing_camera.model_id = CameraModelNameToId("SIMPLE_RADIAL");
  existing_camera.width = 10;
  existing_camera.height = 20;
  existing_camera.params = {1.0, 5.0, 10.0, 0.0};
  existing_camera.camera_id = database->WriteCamera(existing_camera);
  Rig existing_rig;
  existing_rig.AddRefSensor(existing_camera.SensorId());
  database->WriteRig(existing_rig);

  Image existing_image;
  existing_image.SetName("test.png");
  existing_image.SetCameraId(existing_camera.camera_id);
  existing_image.SetImageId(database->WriteImage(existing_image));
  database->WriteKeypoints(existing_image.ImageId(), FeatureKeypoints());
  database->WriteDescriptors(existing_image.ImageId(), FeatureDescriptors());

  ImageReader image_reader(options, database.get());

  Rig rig;
  Camera camera;
  Image image;
  PosePrior pose_prior;
  Bitmap bitmap;
  Bitmap mask;

  const auto status =
      image_reader.Next(&rig, &camera, &image, &pose_prior, &bitmap, &mask);
  EXPECT_EQ(status, ImageReader::Status::IMAGE_EXISTS);
}

}  // namespace
}  // namespace colmap

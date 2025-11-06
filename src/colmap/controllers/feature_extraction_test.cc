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

#include "colmap/controllers/feature_extraction.h"

#include "colmap/scene/database.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <fstream>

#include <gtest/gtest.h>

namespace colmap {
namespace {

Bitmap CreateTestBitmap() {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, /*as_rgb=*/false);
  bitmap.Fill(BitmapColor<uint8_t>(0));
  for (int y = 30; y < 70; ++y) {
    for (int x = 30; x < 70; ++x) {
      bitmap.SetPixel(x, y, BitmapColor<uint8_t>(255));
    }
  }
  return bitmap;
}

TEST(CreateFeatureExtractorController, Nominal) {
  const std::string test_dir = CreateTestDir();
  const std::string database_path = test_dir + "/database.db";
  const std::string image_path = test_dir + "/images";
  CreateDirIfNotExists(image_path);

  // Create test images
  const int kNumImages = 2;
  const Bitmap test_bitmap = CreateTestBitmap();
  for (int i = 0; i < kNumImages; ++i) {
    test_bitmap.Write(
        std::string(image_path).append("/").append(std::to_string(i) + ".png"));
  }

  // Set up options
  ImageReaderOptions reader_options;
  reader_options.image_path = image_path;

  FeatureExtractionOptions extraction_options;
  extraction_options.use_gpu = false;
  extraction_options.num_threads = kNumImages;

  // Create and run the controller
  auto controller = CreateFeatureExtractorController(
      database_path, reader_options, extraction_options);
  ASSERT_NE(controller, nullptr);
  controller->Start();
  controller->Wait();

  // Verify results in database
  auto database = Database::Open(database_path);
  const std::vector<Image> images = database->ReadAllImages();
  EXPECT_EQ(images.size(), kNumImages);

  for (const auto& image : images) {
    EXPECT_TRUE(database->ExistsKeypoints(image.ImageId()));
    EXPECT_TRUE(database->ExistsDescriptors(image.ImageId()));

    const FeatureKeypoints keypoints = database->ReadKeypoints(image.ImageId());
    const FeatureDescriptors descriptors =
        database->ReadDescriptors(image.ImageId());

    // Check that features were extracted
    EXPECT_GT(keypoints.size(), 0);
    EXPECT_EQ(keypoints.size(), descriptors.rows());
  }
}

TEST(CreateFeatureExtractorController, WithCameraMask) {
  const std::string test_dir = CreateTestDir();
  const std::string database_path = test_dir + "/database.db";
  const std::string image_path = test_dir + "/images";
  const std::string mask_path = test_dir + "/mask.png";
  CreateDirIfNotExists(image_path);

  // Create test image with features
  const Bitmap test_bitmap = CreateTestBitmap();
  test_bitmap.Write(image_path + "/test.png");

  // Create a mask that only allows the center region (white = keep, black =
  // mask) The test bitmap has a white square from (30,30) to (70,70) We'll
  // create a mask that only keeps a smaller region
  Bitmap mask_bitmap;
  mask_bitmap.Allocate(100, 100, /*as_rgb=*/false);
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

  auto controller = CreateFeatureExtractorController(
      database_path, reader_options_no_mask, extraction_options);
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
  const std::string database_path_masked = test_dir + "/database_masked.db";
  ImageReaderOptions reader_options_masked;
  reader_options_masked.image_path = image_path;
  reader_options_masked.camera_mask_path = mask_path;

  controller = CreateFeatureExtractorController(
      database_path_masked, reader_options_masked, extraction_options);
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
  EXPECT_EQ(descriptors_masked.rows(), keypoints_masked.size());
}

TEST(CreateFeatureImporterController, Nominal) {
  const std::string test_dir = CreateTestDir();
  const std::string database_path = test_dir + "/database.db";
  const std::string image_path = test_dir + "/images";
  const std::string import_path = test_dir + "/features";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(import_path);

  const int kNumImages = 2;
  const int kNumFeatures = 3;

  // Create test images
  const Bitmap test_bitmap = CreateTestBitmap();
  for (int i = 0; i < kNumImages; ++i) {
    test_bitmap.Write(
        std::string(image_path).append("/").append(std::to_string(i) + ".png"));
  }

  // Create feature text files for each image
  for (int i = 0; i < kNumImages; ++i) {
    const std::string feature_file =
        import_path + "/" + std::to_string(i) + ".png.txt";
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
  auto controller = CreateFeatureImporterController(
      database_path, reader_options, import_path);
  ASSERT_NE(controller, nullptr);
  controller->Start();
  controller->Wait();

  // Verify results in database
  auto database = Database::Open(database_path);
  const std::vector<Image> images = database->ReadAllImages();
  EXPECT_EQ(images.size(), kNumImages);

  for (const auto& image : images) {
    EXPECT_TRUE(database->ExistsKeypoints(image.ImageId()));
    EXPECT_TRUE(database->ExistsDescriptors(image.ImageId()));

    const FeatureKeypoints keypoints = database->ReadKeypoints(image.ImageId());
    const FeatureDescriptors descriptors =
        database->ReadDescriptors(image.ImageId());

    // Check that features were imported correctly
    EXPECT_EQ(keypoints.size(), kNumFeatures);
    EXPECT_EQ(descriptors.rows(), kNumFeatures);
    EXPECT_EQ(descriptors.cols(), 128);

    // Verify some keypoint values
    EXPECT_FLOAT_EQ(keypoints[0].x, 10.0f);
    EXPECT_FLOAT_EQ(keypoints[0].y, 20.0f);
  }
}

}  // namespace
}  // namespace colmap

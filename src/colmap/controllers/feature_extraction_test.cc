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
  Bitmap bitmap(100, 100, /*as_rgb=*/false);
  bitmap.Fill(BitmapColor<uint8_t>(0));
  for (int y = 30; y < 70; ++y) {
    for (int x = 30; x < 70; ++x) {
      bitmap.SetPixel(x, y, BitmapColor<uint8_t>(255));
    }
  }
  return bitmap;
}

Bitmap CreateLargeTestBitmap(int width, int height) {
  Bitmap bitmap(width, height, /*as_rgb=*/false);
  bitmap.Fill(BitmapColor<uint8_t>(128));
  // Add a feature-rich pattern with gradients and edges.
  for (int y = 0; y < height; y += 20) {
    for (int x = 0; x < width; x += 20) {
      const int block_w = std::min(10, width - x);
      const int block_h = std::min(10, height - y);
      for (int dy = 0; dy < block_h; ++dy) {
        for (int dx = 0; dx < block_w; ++dx) {
          bitmap.SetPixel(x + dx, y + dy, BitmapColor<uint8_t>(255));
        }
      }
    }
  }
  return bitmap;
}

void WriteFeatureFile(const std::filesystem::path& path,
                      int num_features,
                      int dimension) {
  std::ofstream file(path);
  file << num_features << " " << dimension << "\n";
  for (int j = 0; j < num_features; ++j) {
    file << (10.0f + j * 5.0f) << " "
         << (20.0f + j * 5.0f) << " "
         << (1.5f + j * 0.1f) << " "
         << (0.5f + j * 0.2f);
    for (int k = 0; k < dimension; ++k) {
      file << " " << ((j * dimension + k) % 256);
    }
    file << "\n";
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
  const auto database_path_masked = test_dir / "database_masked.db";
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
  EXPECT_EQ(descriptors_masked.data.rows(), keypoints_masked.size());
  EXPECT_EQ(descriptors_masked.type, FeatureExtractorType::SIFT);
  EXPECT_EQ(descriptors_masked.data.cols(), 128);
}

// Covers ImageResizerThread downscaling path when image exceeds max_image_size.
TEST(CreateFeatureExtractorController, WithMaxImageSize) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(image_path);

  // Create a 200x150 image, then set max_image_size=80 to force downscaling.
  const Bitmap large_bitmap = CreateLargeTestBitmap(200, 150);
  large_bitmap.Write(image_path / "large.png");

  ImageReaderOptions reader_options;
  reader_options.image_path = image_path;

  FeatureExtractionOptions extraction_options;
  extraction_options.use_gpu = false;
  extraction_options.num_threads = 1;
  extraction_options.max_image_size = 80;

  auto controller = CreateFeatureExtractorController(
      database_path, reader_options, extraction_options);
  ASSERT_NE(controller, nullptr);
  controller->Start();
  controller->Wait();

  auto database = Database::Open(database_path);
  const std::vector<Image> images = database->ReadAllImages();
  ASSERT_EQ(images.size(), 1);

  // The camera dimensions in the database reflect the original image size,
  // not the downscaled size. Features should still be extracted successfully.
  EXPECT_TRUE(database->ExistsKeypoints(images[0].ImageId()));
  EXPECT_TRUE(database->ExistsDescriptors(images[0].ImageId()));

  const FeatureKeypoints keypoints =
      database->ReadKeypoints(images[0].ImageId());
  const FeatureDescriptors descriptors =
      database->ReadDescriptors(images[0].ImageId());

  EXPECT_GT(keypoints.size(), 0);
  EXPECT_EQ(keypoints.size(), descriptors.data.rows());
  EXPECT_EQ(descriptors.data.cols(), 128);

  // Keypoints should be scaled back to original image coordinates.
  // The original image is 200x150, so keypoints should be within those bounds.
  for (const auto& kp : keypoints) {
    EXPECT_GE(kp.x, 0.0f);
    EXPECT_LE(kp.x, 200.0f);
    EXPECT_GE(kp.y, 0.0f);
    EXPECT_LE(kp.y, 150.0f);
  }
}

// Covers per-image mask path (mask_path option), which uses a separate mask
// per image as opposed to the camera_mask_path which applies one mask to all.
TEST(CreateFeatureExtractorController, WithPerImageMask) {
  const auto test_dir = CreateTestDir();
  const auto database_path_unmasked = test_dir / "database_unmasked.db";
  const auto database_path_masked = test_dir / "database_masked.db";
  const auto image_path = test_dir / "images";
  const auto mask_dir = test_dir / "masks";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(mask_dir);

  const Bitmap test_bitmap = CreateTestBitmap();
  test_bitmap.Write(image_path / "img.png");

  // Per-image mask: mask_dir/img.png.png (image name + ".png")
  Bitmap mask_bitmap(100, 100, /*as_rgb=*/false);
  mask_bitmap.Fill(BitmapColor<uint8_t>(0));
  for (int y = 40; y < 60; ++y) {
    for (int x = 40; x < 60; ++x) {
      mask_bitmap.SetPixel(x, y, BitmapColor<uint8_t>(255));
    }
  }
  mask_bitmap.Write(mask_dir / "img.png.png");

  FeatureExtractionOptions extraction_options;
  extraction_options.use_gpu = false;
  extraction_options.num_threads = 1;

  // Extract without mask for baseline.
  {
    ImageReaderOptions reader_options;
    reader_options.image_path = image_path;
    auto controller = CreateFeatureExtractorController(
        database_path_unmasked, reader_options, extraction_options);
    ASSERT_NE(controller, nullptr);
    controller->Start();
    controller->Wait();
  }

  auto db_unmasked = Database::Open(database_path_unmasked);
  auto images_unmasked = db_unmasked->ReadAllImages();
  ASSERT_EQ(images_unmasked.size(), 1);
  const size_t num_unmasked =
      db_unmasked->ReadKeypoints(images_unmasked[0].ImageId()).size();
  EXPECT_GT(num_unmasked, 0);

  // Extract with per-image mask.
  {
    ImageReaderOptions reader_options;
    reader_options.image_path = image_path;
    reader_options.mask_path = mask_dir;
    auto controller = CreateFeatureExtractorController(
        database_path_masked, reader_options, extraction_options);
    ASSERT_NE(controller, nullptr);
    controller->Start();
    controller->Wait();
  }

  auto db_masked = Database::Open(database_path_masked);
  auto images_masked = db_masked->ReadAllImages();
  ASSERT_EQ(images_masked.size(), 1);

  const FeatureKeypoints kps_masked =
      db_masked->ReadKeypoints(images_masked[0].ImageId());
  EXPECT_GT(kps_masked.size(), 0);
  EXPECT_LT(kps_masked.size(), num_unmasked);

  for (const auto& kp : kps_masked) {
    EXPECT_GE(kp.x, 40.0f);
    EXPECT_LT(kp.x, 60.0f);
    EXPECT_GE(kp.y, 40.0f);
    EXPECT_LT(kp.y, 60.0f);
  }
}

// Re-running extraction on the same database should skip images that already
// have keypoints and descriptors (IMAGE_EXISTS path), leaving data unchanged.
TEST(CreateFeatureExtractorController, ExistingImageSkipsRewrite) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(image_path);

  const Bitmap test_bitmap = CreateTestBitmap();
  test_bitmap.Write(image_path / "img.png");

  ImageReaderOptions reader_options;
  reader_options.image_path = image_path;

  FeatureExtractionOptions extraction_options;
  extraction_options.use_gpu = false;
  extraction_options.num_threads = 1;

  // First run: extract features normally.
  auto controller = CreateFeatureExtractorController(
      database_path, reader_options, extraction_options);
  ASSERT_NE(controller, nullptr);
  controller->Start();
  controller->Wait();

  auto database = Database::Open(database_path);
  auto images = database->ReadAllImages();
  ASSERT_EQ(images.size(), 1);
  const FeatureKeypoints kps_first =
      database->ReadKeypoints(images[0].ImageId());
  const FeatureDescriptors desc_first =
      database->ReadDescriptors(images[0].ImageId());
  ASSERT_GT(kps_first.size(), 0);

  // Second run: same database, same images. Should hit IMAGE_EXISTS and not
  // overwrite.
  database.reset();
  auto controller2 = CreateFeatureExtractorController(
      database_path, reader_options, extraction_options);
  ASSERT_NE(controller2, nullptr);
  controller2->Start();
  controller2->Wait();

  database = Database::Open(database_path);
  images = database->ReadAllImages();
  ASSERT_EQ(images.size(), 1);

  const FeatureKeypoints kps_second =
      database->ReadKeypoints(images[0].ImageId());
  const FeatureDescriptors desc_second =
      database->ReadDescriptors(images[0].ImageId());

  // Data should be identical since the second run skipped re-extraction.
  EXPECT_EQ(kps_first.size(), kps_second.size());
  EXPECT_EQ(desc_first.data.rows(), desc_second.data.rows());
  EXPECT_EQ(desc_first.data.cols(), desc_second.data.cols());
  for (size_t i = 0; i < kps_first.size(); ++i) {
    EXPECT_EQ(kps_first[i], kps_second[i]);
  }
}

// Covers the nonexistent camera_mask_path error path in the constructor:
// a camera_mask_path that does not point to an existing file triggers a
// LOG(ERROR) but extraction still proceeds without masking.
TEST(CreateFeatureExtractorController, NonExistentCameraMaskPath) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(image_path);

  const Bitmap test_bitmap = CreateTestBitmap();
  test_bitmap.Write(image_path / "img.png");

  ImageReaderOptions reader_options;
  reader_options.image_path = image_path;
  reader_options.camera_mask_path = test_dir / "nonexistent_mask.png";

  FeatureExtractionOptions extraction_options;
  extraction_options.use_gpu = false;
  extraction_options.num_threads = 1;

  // Should not crash; proceeds without mask.
  auto controller = CreateFeatureExtractorController(
      database_path, reader_options, extraction_options);
  ASSERT_NE(controller, nullptr);
  controller->Start();
  controller->Wait();

  auto database = Database::Open(database_path);
  auto images = database->ReadAllImages();
  ASSERT_EQ(images.size(), 1);

  const FeatureKeypoints keypoints =
      database->ReadKeypoints(images[0].ImageId());
  EXPECT_GT(keypoints.size(), 0);
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
    WriteFeatureFile(feature_file, kNumFeatures, 128);
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
    EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
    EXPECT_EQ(descriptors.data.rows(), kNumFeatures);
    EXPECT_EQ(descriptors.data.cols(), 128);

    // Verify some keypoint values
    EXPECT_FLOAT_EQ(keypoints[0].x, 10.0f);
    EXPECT_FLOAT_EQ(keypoints[0].y, 20.0f);
  }
}

// Covers the "SKIP: No features found" path when a feature text file is
// missing for one image but present for another.
TEST(CreateFeatureImporterController, MissingFeatureFile) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";
  const auto import_path = test_dir / "features";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(import_path);

  const Bitmap test_bitmap = CreateTestBitmap();
  test_bitmap.Write(image_path / "has_features.png");
  test_bitmap.Write(image_path / "no_features.png");

  // Only create feature file for the first image.
  WriteFeatureFile(import_path / "has_features.png.txt", 3, 128);
  // Deliberately do NOT create no_features.png.txt.

  ImageReaderOptions reader_options;
  reader_options.image_path = image_path;

  auto controller = CreateFeatureImporterController(
      database_path, reader_options, import_path);
  ASSERT_NE(controller, nullptr);
  controller->Start();
  controller->Wait();

  auto database = Database::Open(database_path);
  const auto images = database->ReadAllImages();

  // The image with features should be fully imported.
  // The image without features should still be in the database (from the
  // image reader) but should have no keypoints/descriptors.
  bool found_with_features = false;
  bool found_without_features = false;
  for (const auto& image : images) {
    if (image.Name() == "has_features.png") {
      found_with_features = true;
      EXPECT_TRUE(database->ExistsKeypoints(image.ImageId()));
      EXPECT_TRUE(database->ExistsDescriptors(image.ImageId()));
      EXPECT_EQ(database->ReadKeypoints(image.ImageId()).size(), 3);
    } else if (image.Name() == "no_features.png") {
      found_without_features = true;
      // Image was read but no features were written.
      EXPECT_FALSE(database->ExistsKeypoints(image.ImageId()));
      EXPECT_FALSE(database->ExistsDescriptors(image.ImageId()));
    }
  }
  EXPECT_TRUE(found_with_features);
  EXPECT_TRUE(found_without_features);
}

// Covers the early-return path when the import directory does not exist.
TEST(CreateFeatureImporterController, NonExistentImportDir) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(image_path);

  const Bitmap test_bitmap = CreateTestBitmap();
  test_bitmap.Write(image_path / "img.png");

  ImageReaderOptions reader_options;
  reader_options.image_path = image_path;

  const auto bogus_import_path = test_dir / "does_not_exist";

  auto controller = CreateFeatureImporterController(
      database_path, reader_options, bogus_import_path);
  ASSERT_NE(controller, nullptr);
  controller->Start();
  controller->Wait();

  // The controller should return early without crashing.
  // No images should be in the database since the import loop was never entered.
  auto database = Database::Open(database_path);
  EXPECT_EQ(database->ReadAllImages().size(), 0);
}

// Re-running the importer on the same database should not overwrite existing
// keypoints/descriptors (ExistsKeypoints / ExistsDescriptors guard paths).
TEST(CreateFeatureImporterController, ExistingFeaturesNotOverwritten) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";
  const auto import_path = test_dir / "features";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(import_path);

  const Bitmap test_bitmap = CreateTestBitmap();
  test_bitmap.Write(image_path / "img.png");

  WriteFeatureFile(import_path / "img.png.txt", 3, 128);

  ImageReaderOptions reader_options;
  reader_options.image_path = image_path;

  // First import.
  {
    auto controller = CreateFeatureImporterController(
        database_path, reader_options, import_path);
    ASSERT_NE(controller, nullptr);
    controller->Start();
    controller->Wait();
  }

  auto database = Database::Open(database_path);
  auto images = database->ReadAllImages();
  ASSERT_EQ(images.size(), 1);
  const image_t image_id = images[0].ImageId();

  const FeatureKeypoints kps_first = database->ReadKeypoints(image_id);
  ASSERT_EQ(kps_first.size(), 3);

  // Now write a different feature file with more features.
  WriteFeatureFile(import_path / "img.png.txt", 5, 128);
  database.reset();

  // Second import on the same database.
  {
    auto controller = CreateFeatureImporterController(
        database_path, reader_options, import_path);
    ASSERT_NE(controller, nullptr);
    controller->Start();
    controller->Wait();
  }

  database = Database::Open(database_path);
  images = database->ReadAllImages();
  ASSERT_EQ(images.size(), 1);

  // Features should still be 3 from the first import, not 5.
  const FeatureKeypoints kps_second = database->ReadKeypoints(image_id);
  EXPECT_EQ(kps_second.size(), 3);
}

}  // namespace
}  // namespace colmap

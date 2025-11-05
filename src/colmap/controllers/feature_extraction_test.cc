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

}  // namespace
}  // namespace colmap

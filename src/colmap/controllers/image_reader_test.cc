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
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

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

  const std::string test_dir = CreateTestDir();
  ImageReaderOptions options;
  options.image_path = test_dir + "/images";
  options.as_rgb = kAsRGB;
  CreateDirIfNotExists(options.image_path);
  if (kWithMasks) {
    options.mask_path = test_dir + "/masks";
    CreateDirIfNotExists(options.mask_path);
  }
  const Bitmap test_bitmap = CreateTestBitmap(kAsRGB);
  for (int i = 0; i < kNumImages; ++i) {
    const std::string stem = std::to_string(i);
    const std::string image_name = stem + kExtension;
    test_bitmap.Write(options.image_path + "/" + image_name);
    if (kWithMasks) {
      if (i == 0) {
        // append .png to image_name
        test_bitmap.Write(options.mask_path + "/" + image_name + ".png");
      } else {
        // replace mask extension by .png
        test_bitmap.Write(options.mask_path + "/" + stem + ".png");
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

}  // namespace
}  // namespace colmap

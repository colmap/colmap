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

#include "colmap/feature/loma.h"

#include "colmap/feature/matcher.h"
#include "colmap/math/random.h"
#include "colmap/scene/camera.h"
#include "colmap/sensor/bitmap.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

void CreateRandomRgbImage(int width, int height, Bitmap* bitmap) {
  SetPRNGSeed(42);
  *bitmap = Bitmap(width, height, /*as_rgb=*/true);
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      bitmap->SetPixel(c,
                       r,
                       BitmapColor<uint8_t>(RandomUniformInteger(0, 255),
                                            RandomUniformInteger(0, 255),
                                            RandomUniformInteger(0, 255)));
    }
  }
}

TEST(LomaTest, Nominal) {
  Bitmap image;
  CreateRandomRgbImage(512, 512, &image);

  FeatureExtractionOptions extraction_options(FeatureExtractorType::LOMA_B);
  extraction_options.use_gpu = false;
  extraction_options.loma->min_score = 0.0;
  auto extractor = CreateLomaFeatureExtractor(extraction_options);
  auto keypoints = std::make_shared<FeatureKeypoints>();
  auto descriptors = std::make_shared<FeatureDescriptors>();
  ASSERT_TRUE(extractor->Extract(image, keypoints.get(), descriptors.get()));

  EXPECT_GT(keypoints->size(), 0);
  EXPECT_EQ(keypoints->size(), descriptors->data.rows());
  EXPECT_EQ(descriptors->type, FeatureExtractorType::LOMA_B);

  for (const auto& keypoint : *keypoints) {
    EXPECT_GE(keypoint.x, 0);
    EXPECT_GE(keypoint.y, 0);
    EXPECT_LE(keypoint.x, image.Width());
    EXPECT_LE(keypoint.y, image.Height());
  }

  Camera camera;
  camera.width = image.Width();
  camera.height = image.Height();

  FeatureMatchingOptions matching_options(FeatureMatcherType::LOMA_B);
  matching_options.use_gpu = false;
  matching_options.loma->min_score = 0.0;
  auto matcher = CreateLomaFeatureMatcher(matching_options);

  FeatureMatches matches;
  const FeatureMatcher::Image image1{1, &camera, keypoints, descriptors};
  const FeatureMatcher::Image image2{2, &camera, keypoints, descriptors};
  matcher->Match(image1, image2, &matches);

  ASSERT_GT(matches.size(), 0);
  int num_self_matches = 0;
  for (const auto& match : matches) {
    EXPECT_GE(match.point2D_idx1, 0);
    EXPECT_LT(match.point2D_idx1, keypoints->size());
    EXPECT_GE(match.point2D_idx2, 0);
    EXPECT_LT(match.point2D_idx2, keypoints->size());
    if (match.point2D_idx1 == match.point2D_idx2) ++num_self_matches;
  }
  EXPECT_GT(num_self_matches, 0.5 * matches.size());
}

TEST(LomaTest, MinScore) {
  Bitmap image;
  CreateRandomRgbImage(512, 512, &image);

  FeatureExtractionOptions options_low(FeatureExtractorType::LOMA_B);
  options_low.use_gpu = false;
  options_low.loma->min_score = 0.0;
  auto extractor_low = CreateLomaFeatureExtractor(options_low);
  FeatureKeypoints keypoints_low;
  FeatureDescriptors descriptors_low;
  ASSERT_TRUE(extractor_low->Extract(image, &keypoints_low, &descriptors_low));

  FeatureExtractionOptions options_high(FeatureExtractorType::LOMA_B);
  options_high.use_gpu = false;
  options_high.loma->min_score = 0.9;
  auto extractor_high = CreateLomaFeatureExtractor(options_high);
  FeatureKeypoints keypoints_high;
  FeatureDescriptors descriptors_high;
  ASSERT_TRUE(
      extractor_high->Extract(image, &keypoints_high, &descriptors_high));

  EXPECT_LE(keypoints_high.size(), keypoints_low.size());
}

TEST(LomaTest, DynamicNumKeypoints) {
  Bitmap image;
  CreateRandomRgbImage(512, 512, &image);

  FeatureExtractionOptions options_512(FeatureExtractorType::LOMA_B);
  options_512.use_gpu = false;
  options_512.loma->min_score = 0.0;
  options_512.loma->max_num_features = 512;
  auto extractor_512 = CreateLomaFeatureExtractor(options_512);
  FeatureKeypoints keypoints_512;
  FeatureDescriptors descriptors_512;
  ASSERT_TRUE(extractor_512->Extract(image, &keypoints_512, &descriptors_512));

  FeatureExtractionOptions options_2048(FeatureExtractorType::LOMA_B);
  options_2048.use_gpu = false;
  options_2048.loma->min_score = 0.0;
  options_2048.loma->max_num_features = 2048;
  auto extractor_2048 = CreateLomaFeatureExtractor(options_2048);
  FeatureKeypoints keypoints_2048;
  FeatureDescriptors descriptors_2048;
  ASSERT_TRUE(
      extractor_2048->Extract(image, &keypoints_2048, &descriptors_2048));

  // With min_score=0 nothing is filtered, so the exact requested count should
  // come back -- proves num_keypoints is a real runtime input to the detector
  // graph (see make_detector_dynamic_k() in deployment/export_onnx.py),
  // not just accepted-but-ignored.
  EXPECT_EQ(keypoints_512.size(), 512);
  EXPECT_EQ(keypoints_2048.size(), 2048);
}

}  // namespace
}  // namespace colmap

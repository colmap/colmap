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

#include "colmap/feature/aliked.h"

#include "colmap/feature/matcher.h"
#include "colmap/feature/resources.h"
#include "colmap/math/random.h"
#include "colmap/sensor/bitmap.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

void CreateRandomRgbImage(const int width, const int height, Bitmap* bitmap) {
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

class ParameterizedAlikedTests : public testing::TestWithParam<std::string> {};

TEST_P(ParameterizedAlikedTests, Nominal) {
  Bitmap image;
  CreateRandomRgbImage(200, 100, &image);

  FeatureExtractionOptions extraction_options(FeatureExtractorType::ALIKED);
  extraction_options.use_gpu = false;
  extraction_options.aliked->model_path = GetParam();
  auto extractor = CreateAlikedFeatureExtractor(extraction_options);
  auto keypoints = std::make_shared<FeatureKeypoints>();
  auto descriptors = std::make_shared<FeatureDescriptors>();
  ASSERT_TRUE(extractor->Extract(image, keypoints.get(), descriptors.get()));

  // Check keypoint count is reasonable.
  EXPECT_GT(keypoints->size(), 0);
  EXPECT_LE(keypoints->size(), extraction_options.aliked->max_num_features);
  EXPECT_EQ(keypoints->size(), descriptors->rows());

  // Descriptor dimension should be a multiple of sizeof(float).
  EXPECT_EQ(descriptors->cols() % sizeof(float), 0);
  const int descriptor_dim = descriptors->cols() / sizeof(float);
  EXPECT_GT(descriptor_dim, 0);

  // Keypoints should be within image bounds (with small tolerance for
  // sub-pixel refinement).
  for (const auto& keypoint : *keypoints) {
    EXPECT_GE(keypoint.x, -1);
    EXPECT_GE(keypoint.y, -1);
    EXPECT_LE(keypoint.x, image.Width() + 1);
    EXPECT_LE(keypoint.y, image.Height() + 1);
  }

  // Test brute-force matcher.
  FeatureMatchingOptions matching_options(
      FeatureMatcherType::ALIKED_BRUTEFORCE);
  matching_options.use_gpu = false;
  auto matcher = CreateAlikedFeatureMatcher(matching_options);
  FeatureMatches matches;
  FeatureMatcher::Image img1{/*image_id=*/1,
                             /*camera=*/nullptr,
                             keypoints,
                             descriptors};
  FeatureMatcher::Image img2{/*image_id=*/2,
                             /*camera=*/nullptr,
                             keypoints,
                             descriptors};
  matcher->Match(img1, img2, &matches);

  // Self-matching should produce many matches.
  EXPECT_NEAR(matches.size(), keypoints->size(), 0.1 * keypoints->size());
  for (const auto& match : matches) {
    EXPECT_GE(match.point2D_idx1, 0);
    EXPECT_GE(match.point2D_idx2, 0);
    EXPECT_LT(match.point2D_idx1, keypoints->size());
    EXPECT_LT(match.point2D_idx2, keypoints->size());
  }
}

TEST_P(ParameterizedAlikedTests, MaxNumFeatures) {
  Bitmap image;
  CreateRandomRgbImage(200, 100, &image);

  // Extract with default max_num_features.
  FeatureExtractionOptions options_default(FeatureExtractorType::ALIKED);
  options_default.use_gpu = false;
  options_default.aliked->model_path = GetParam();
  auto extractor_default = CreateAlikedFeatureExtractor(options_default);
  FeatureKeypoints keypoints_default;
  FeatureDescriptors descriptors_default;
  ASSERT_TRUE(extractor_default->Extract(
      image, &keypoints_default, &descriptors_default));

  // Extract with reduced max_num_features.
  FeatureExtractionOptions options_limited(FeatureExtractorType::ALIKED);
  options_limited.use_gpu = false;
  options_limited.aliked->model_path = GetParam();
  options_limited.aliked->max_num_features = 100;
  auto extractor_limited = CreateAlikedFeatureExtractor(options_limited);
  FeatureKeypoints keypoints_limited;
  FeatureDescriptors descriptors_limited;
  ASSERT_TRUE(extractor_limited->Extract(
      image, &keypoints_limited, &descriptors_limited));

  // Limited extraction should have fewer or equal keypoints.
  EXPECT_LE(keypoints_limited.size(), 100);
  EXPECT_LT(keypoints_limited.size(), keypoints_default.size());
}

TEST_P(ParameterizedAlikedTests, MinScore) {
  Bitmap image;
  CreateRandomRgbImage(200, 100, &image);

  // Extract with low min_score threshold.
  FeatureExtractionOptions options_low(FeatureExtractorType::ALIKED);
  options_low.use_gpu = false;
  options_low.aliked->model_path = GetParam();
  options_low.aliked->min_score = 0.0;
  auto extractor_low = CreateAlikedFeatureExtractor(options_low);
  FeatureKeypoints keypoints_low;
  FeatureDescriptors descriptors_low;
  ASSERT_TRUE(extractor_low->Extract(image, &keypoints_low, &descriptors_low));

  // Extract with high min_score threshold.
  FeatureExtractionOptions options_high(FeatureExtractorType::ALIKED);
  options_high.use_gpu = false;
  options_high.aliked->model_path = GetParam();
  options_high.aliked->min_score = 0.9;
  auto extractor_high = CreateAlikedFeatureExtractor(options_high);
  FeatureKeypoints keypoints_high;
  FeatureDescriptors descriptors_high;
  ASSERT_TRUE(
      extractor_high->Extract(image, &keypoints_high, &descriptors_high));

  // Different min_score values should produce different keypoint counts
  // (verifies the parameter is being passed to the model).
  EXPECT_NE(keypoints_high.size(), keypoints_low.size());
}

INSTANTIATE_TEST_SUITE_P(
    AlikedTests,
    ParameterizedAlikedTests,
    testing::Values(kDefaultALIKEDN16RotFeatureExtractorUri,
                    kDefaultALIKEDN32FeatureExtractorUri),
    [](const testing::TestParamInfo<std::string>& info) {
      if (info.param == kDefaultALIKEDN16RotFeatureExtractorUri) {
        return "N16Rot";
      } else if (info.param == kDefaultALIKEDN32FeatureExtractorUri) {
        return "N32";
      }
      return "Unknown";
    });

}  // namespace
}  // namespace colmap

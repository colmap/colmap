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
#include "colmap/feature/utils.h"
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

class ParameterizedAlikedTests
    : public testing::TestWithParam<FeatureExtractorType> {};

TEST_P(ParameterizedAlikedTests, Nominal) {
  Bitmap image;
  CreateRandomRgbImage(200, 100, &image);

  FeatureExtractionOptions extraction_options(GetParam());
  extraction_options.use_gpu = false;
  auto extractor = CreateAlikedFeatureExtractor(extraction_options);
  auto keypoints = std::make_shared<FeatureKeypoints>();
  auto descriptors = std::make_shared<FeatureDescriptors>();
  ASSERT_TRUE(extractor->Extract(image, keypoints.get(), descriptors.get()));

  // Check keypoint count is reasonable.
  EXPECT_GT(keypoints->size(), 0);
  EXPECT_LE(keypoints->size(), extraction_options.aliked->max_num_features);
  EXPECT_EQ(keypoints->size(), descriptors->data.rows());
  EXPECT_EQ(descriptors->type, GetParam());
  EXPECT_EQ(descriptors->data.cols(), 128 * sizeof(float));

  // Keypoints should be within image bounds.
  for (const auto& keypoint : *keypoints) {
    EXPECT_GE(keypoint.x, 0);
    EXPECT_GE(keypoint.y, 0);
    EXPECT_LE(keypoint.x, image.Width());
    EXPECT_LE(keypoint.y, image.Height());
  }

  // Test brute-force matcher.
  FeatureMatchingOptions matching_options(
      FeatureMatcherType::ALIKED_BRUTEFORCE);
  matching_options.use_gpu = false;
  // Disable ratio test for self-matching to get all matches.
  matching_options.aliked->max_ratio = 0;
  auto matcher = CreateAlikedFeatureMatcher(matching_options);
  FeatureMatches matches;
  FeatureMatcher::Image image1{/*image_id=*/1,
                               /*camera=*/nullptr,
                               keypoints,
                               descriptors};
  FeatureMatcher::Image image2{/*image_id=*/2,
                               /*camera=*/nullptr,
                               keypoints,
                               descriptors};
  matcher->Match(image1, image2, &matches);

  // Self-matching should produce a match for every keypoint.
  EXPECT_EQ(matches.size(), keypoints->size());
  for (const auto& match : matches) {
    EXPECT_EQ(match.point2D_idx1, match.point2D_idx2);
    EXPECT_GE(match.point2D_idx1, 0);
    EXPECT_LT(match.point2D_idx1, keypoints->size());
  }
}

TEST_P(ParameterizedAlikedTests, MaxNumFeatures) {
  Bitmap image;
  CreateRandomRgbImage(200, 100, &image);

  // Extract with default max_num_features.
  FeatureExtractionOptions options_default(GetParam());
  options_default.use_gpu = false;
  auto extractor_default = CreateAlikedFeatureExtractor(options_default);
  FeatureKeypoints keypoints_default;
  FeatureDescriptors descriptors_default;
  ASSERT_TRUE(extractor_default->Extract(
      image, &keypoints_default, &descriptors_default));

  // Extract with reduced max_num_features.
  FeatureExtractionOptions options_limited(GetParam());
  options_limited.use_gpu = false;
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
  FeatureExtractionOptions options_low(GetParam());
  options_low.use_gpu = false;
  options_low.aliked->min_score = 0.0;
  auto extractor_low = CreateAlikedFeatureExtractor(options_low);
  FeatureKeypoints keypoints_low;
  FeatureDescriptors descriptors_low;
  ASSERT_TRUE(extractor_low->Extract(image, &keypoints_low, &descriptors_low));

  // Extract with high min_score threshold.
  FeatureExtractionOptions options_high(GetParam());
  options_high.use_gpu = false;
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

INSTANTIATE_TEST_SUITE_P(AlikedTests,
                         ParameterizedAlikedTests,
                         testing::Values(FeatureExtractorType::ALIKED_N16ROT,
                                         FeatureExtractorType::ALIKED_N32));

// Standalone matcher test with synthesized random descriptors.
class AlikedMatcherTest : public testing::Test {
 protected:
  static constexpr int kDescriptorDim = 128;

  // Create L2-normalized random descriptors.
  static std::shared_ptr<FeatureDescriptors> CreateRandomDescriptors(
      int kNumDescriptors) {
    auto descriptors = std::make_shared<FeatureDescriptors>();
    descriptors->type = FeatureExtractorType::ALIKED_N16ROT;
    descriptors->data.resize(kNumDescriptors, kDescriptorDim * sizeof(float));

    if (kNumDescriptors == 0) {
      return descriptors;
    }

    // Generate random float descriptors and L2 normalize.
    FeatureDescriptorsFloatData float_data =
        FeatureDescriptorsFloatData::Random(kNumDescriptors, kDescriptorDim);
    L2NormalizeFeatureDescriptors(&float_data);

    // Copy float data into byte storage.
    std::memcpy(descriptors->data.data(),
                float_data.data(),
                kNumDescriptors * kDescriptorDim * sizeof(float));
    return descriptors;
  }

  // Create dummy keypoints (matcher doesn't use coordinates).
  static std::shared_ptr<FeatureKeypoints> CreateDummyKeypoints(int count) {
    auto keypoints = std::make_shared<FeatureKeypoints>(count);
    for (int i = 0; i < count; ++i) {
      (*keypoints)[i].x = static_cast<float>(i);
      (*keypoints)[i].y = static_cast<float>(i);
    }
    return keypoints;
  }
};

TEST_F(AlikedMatcherTest, SelfMatching) {
  SetPRNGSeed(42);
  constexpr int kNumDescriptors = 20;

  auto keypoints = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors = CreateRandomDescriptors(kNumDescriptors);

  FeatureMatchingOptions options(FeatureMatcherType::ALIKED_BRUTEFORCE);
  options.use_gpu = false;
  options.aliked->max_ratio = 0;  // Disable ratio test.
  auto matcher = CreateAlikedFeatureMatcher(options);

  FeatureMatcher::Image image1{1, nullptr, keypoints, descriptors};
  FeatureMatcher::Image image2{2, nullptr, keypoints, descriptors};

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);

  // Self-matching should match every descriptor to itself.
  EXPECT_EQ(matches.size(), kNumDescriptors);
  for (const auto& match : matches) {
    EXPECT_EQ(match.point2D_idx1, match.point2D_idx2);
  }
}

TEST_F(AlikedMatcherTest, RatioTest) {
  SetPRNGSeed(42);
  constexpr int kNumDescriptors = 20;

  auto keypoints1 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors1 = CreateRandomDescriptors(kNumDescriptors);
  auto keypoints2 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors2 = CreateRandomDescriptors(kNumDescriptors);

  FeatureMatchingOptions options_no_ratio(
      FeatureMatcherType::ALIKED_BRUTEFORCE);
  options_no_ratio.use_gpu = false;
  options_no_ratio.aliked->max_ratio = 0;  // Disable ratio test.
  options_no_ratio.aliked->cross_check = false;
  options_no_ratio.aliked->min_cossim = -1;  // Accept all.
  auto matcher_no_ratio = CreateAlikedFeatureMatcher(options_no_ratio);

  FeatureMatchingOptions options_ratio(FeatureMatcherType::ALIKED_BRUTEFORCE);
  options_ratio.use_gpu = false;
  options_ratio.aliked->max_ratio = 0.8;  // Enable ratio test.
  options_ratio.aliked->cross_check = false;
  options_ratio.aliked->min_cossim = -1;  // Accept all.
  auto matcher_ratio = CreateAlikedFeatureMatcher(options_ratio);

  FeatureMatcher::Image image1{1, nullptr, keypoints1, descriptors1};
  FeatureMatcher::Image image2{2, nullptr, keypoints2, descriptors2};

  FeatureMatches matches_no_ratio, matches_ratio;
  matcher_no_ratio->Match(image1, image2, &matches_no_ratio);
  matcher_ratio->Match(image1, image2, &matches_ratio);

  // Ratio test should filter some matches.
  EXPECT_GT(matches_no_ratio.size(), 0);
  EXPECT_LT(matches_ratio.size(), matches_no_ratio.size());
}

TEST_F(AlikedMatcherTest, CrossCheck) {
  SetPRNGSeed(42);
  constexpr int kNumDescriptors = 20;

  auto keypoints1 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors1 = CreateRandomDescriptors(kNumDescriptors);
  auto keypoints2 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors2 = CreateRandomDescriptors(kNumDescriptors);

  FeatureMatchingOptions options_no_cross(
      FeatureMatcherType::ALIKED_BRUTEFORCE);
  options_no_cross.use_gpu = false;
  options_no_cross.aliked->max_ratio = 0;
  options_no_cross.aliked->cross_check = false;
  options_no_cross.aliked->min_cossim = -1;
  auto matcher_no_cross = CreateAlikedFeatureMatcher(options_no_cross);

  FeatureMatchingOptions options_cross(FeatureMatcherType::ALIKED_BRUTEFORCE);
  options_cross.use_gpu = false;
  options_cross.aliked->max_ratio = 0;
  options_cross.aliked->cross_check = true;
  options_cross.aliked->min_cossim = -1;
  auto matcher_cross = CreateAlikedFeatureMatcher(options_cross);

  FeatureMatcher::Image image1{1, nullptr, keypoints1, descriptors1};
  FeatureMatcher::Image image2{2, nullptr, keypoints2, descriptors2};

  FeatureMatches matches_no_cross, matches_cross;
  matcher_no_cross->Match(image1, image2, &matches_no_cross);
  matcher_cross->Match(image1, image2, &matches_cross);

  // Cross-check should filter some matches.
  EXPECT_GT(matches_no_cross.size(), 0);
  EXPECT_LT(matches_cross.size(), matches_no_cross.size());
}

TEST_F(AlikedMatcherTest, MinCossim) {
  SetPRNGSeed(42);
  constexpr int kNumDescriptors = 20;

  auto keypoints1 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors1 = CreateRandomDescriptors(kNumDescriptors);
  auto keypoints2 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors2 = CreateRandomDescriptors(kNumDescriptors);

  FeatureMatchingOptions options_low(FeatureMatcherType::ALIKED_BRUTEFORCE);
  options_low.use_gpu = false;
  options_low.aliked->max_ratio = 0;
  options_low.aliked->cross_check = false;
  options_low.aliked->min_cossim = -1;  // Accept all.
  auto matcher_low = CreateAlikedFeatureMatcher(options_low);

  FeatureMatchingOptions options_high(FeatureMatcherType::ALIKED_BRUTEFORCE);
  options_high.use_gpu = false;
  options_high.aliked->max_ratio = 0;
  options_high.aliked->cross_check = false;
  options_high.aliked->min_cossim = 0.9;  // Very strict.
  auto matcher_high = CreateAlikedFeatureMatcher(options_high);

  FeatureMatcher::Image image1{1, nullptr, keypoints1, descriptors1};
  FeatureMatcher::Image image2{2, nullptr, keypoints2, descriptors2};

  FeatureMatches matches_low, matches_high;
  matcher_low->Match(image1, image2, &matches_low);
  matcher_high->Match(image1, image2, &matches_high);

  // Higher min_cossim should filter more matches.
  EXPECT_GT(matches_low.size(), 0);
  EXPECT_LT(matches_high.size(), matches_low.size());
}

TEST_F(AlikedMatcherTest, TooFewDescriptors) {
  SetPRNGSeed(42);

  // Model requires at least 2 descriptors for ratio test.
  auto keypoints1 = CreateDummyKeypoints(1);
  auto descriptors1 = CreateRandomDescriptors(1);
  auto keypoints2 = CreateDummyKeypoints(10);
  auto descriptors2 = CreateRandomDescriptors(10);

  FeatureMatchingOptions options(FeatureMatcherType::ALIKED_BRUTEFORCE);
  options.use_gpu = false;
  auto matcher = CreateAlikedFeatureMatcher(options);

  FeatureMatcher::Image image1{1, nullptr, keypoints1, descriptors1};
  FeatureMatcher::Image image2{2, nullptr, keypoints2, descriptors2};

  FeatureMatches matches12;
  matcher->Match(image1, image2, &matches12);
  EXPECT_EQ(matches12.size(), 0);

  FeatureMatches matches21;
  matcher->Match(image2, image1, &matches21);
  EXPECT_EQ(matches21.size(), 0);
}

TEST_F(AlikedMatcherTest, EmptyDescriptors) {
  auto keypoints1 = CreateDummyKeypoints(0);
  auto descriptors1 = CreateRandomDescriptors(0);
  auto keypoints2 = CreateDummyKeypoints(10);
  auto descriptors2 = CreateRandomDescriptors(10);

  FeatureMatchingOptions options(FeatureMatcherType::ALIKED_BRUTEFORCE);
  options.use_gpu = false;
  auto matcher = CreateAlikedFeatureMatcher(options);

  FeatureMatcher::Image image1{1, nullptr, keypoints1, descriptors1};
  FeatureMatcher::Image image2{2, nullptr, keypoints2, descriptors2};

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);

  // Should return empty matches without crashing.
  EXPECT_EQ(matches.size(), 0);
}

}  // namespace
}  // namespace colmap

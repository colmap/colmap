// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/aliked.h"

#include "colmap/feature/matcher.h"
#include "colmap/math/random.h"
#include "colmap/sensor/bitmap.h"

#include <map>
#include <memory>

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

void CreateRandomRgbImage(const int width, const int height, Bitmap* bitmap) {
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

struct SharedDefaultExtraction {
  Bitmap image;
  std::shared_ptr<FeatureKeypoints> keypoints;
  std::shared_ptr<FeatureDescriptors> descriptors;
  int max_num_features = 0;
};

// The default extraction, shared across the TEST_Ps below to avoid redundant
// inference runs. Computed lazily per variant, since a parameterized suite
// cannot precompute per-variant data in SetUpTestSuite. Only plain image
// and feature data is cached (the extractor itself is destroyed before
// returning), so destruction at process exit is safe.
const SharedDefaultExtraction& SharedDefaultExtractionFor(
    FeatureExtractorType type) {
  static std::map<FeatureExtractorType, SharedDefaultExtraction> cache;
  const auto it = cache.find(type);
  if (it != cache.end()) {
    return it->second;
  }
  SharedDefaultExtraction extraction;
  extraction.keypoints = std::make_shared<FeatureKeypoints>();
  extraction.descriptors = std::make_shared<FeatureDescriptors>();
  CreateRandomRgbImage(200, 100, &extraction.image);
  FeatureExtractionOptions options(type);
  options.use_gpu = false;
  options.aliked->min_score = 0.0;
  auto extractor = CreateAlikedFeatureExtractor(options);
  THROW_CHECK(extractor->Extract(extraction.image,
                                 extraction.keypoints.get(),
                                 extraction.descriptors.get()));
  extraction.max_num_features = options.aliked->max_num_features;
  return cache.emplace(type, std::move(extraction)).first->second;
}

TEST_P(ParameterizedAlikedTests, Nominal) {
  const SharedDefaultExtraction& extraction =
      SharedDefaultExtractionFor(GetParam());
  const auto& keypoints = extraction.keypoints;
  const auto& descriptors = extraction.descriptors;

  // Check keypoint count is reasonable.
  EXPECT_GT(keypoints->size(), 0);
  EXPECT_LE(keypoints->size(), extraction.max_num_features);
  EXPECT_EQ(keypoints->size(), descriptors->data.rows());
  EXPECT_EQ(descriptors->type, GetParam());
  EXPECT_EQ(descriptors->data.cols(), 128 * sizeof(float));

  // Keypoints should be within image bounds.
  for (const auto& keypoint : *keypoints) {
    EXPECT_GE(keypoint.x, 0);
    EXPECT_GE(keypoint.y, 0);
    EXPECT_LE(keypoint.x, extraction.image.Width());
    EXPECT_LE(keypoint.y, extraction.image.Height());
  }

  // Test brute-force matcher.
  FeatureMatchingOptions matching_options(
      FeatureMatcherType::ALIKED_BRUTEFORCE);
  matching_options.use_gpu = false;
  // Disable ratio test for self-matching to get all matches.
  matching_options.aliked->brute_force.max_ratio = 0;
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
  const SharedDefaultExtraction& default_extraction =
      SharedDefaultExtractionFor(GetParam());

  // Extract with reduced max_num_features. The threshold is a model input,
  // so it needs its own inference run.
  FeatureExtractionOptions options_limited(GetParam());
  options_limited.use_gpu = false;
  options_limited.aliked->min_score = 0.0;
  options_limited.aliked->max_num_features = 100;
  auto extractor_limited = CreateAlikedFeatureExtractor(options_limited);
  FeatureKeypoints keypoints_limited;
  FeatureDescriptors descriptors_limited;
  ASSERT_TRUE(extractor_limited->Extract(
      default_extraction.image, &keypoints_limited, &descriptors_limited));

  // Limited extraction should have fewer or equal keypoints.
  EXPECT_LE(keypoints_limited.size(), 100);
  EXPECT_LT(keypoints_limited.size(), default_extraction.keypoints->size());
}

TEST_P(ParameterizedAlikedTests, MinScore) {
  const SharedDefaultExtraction& low_extraction =
      SharedDefaultExtractionFor(GetParam());

  // Extract with high min_score threshold. The threshold is a model input,
  // so it needs its own inference run.
  FeatureExtractionOptions options_high(GetParam());
  options_high.use_gpu = false;
  options_high.aliked->min_score = 0.9;
  auto extractor_high = CreateAlikedFeatureExtractor(options_high);
  FeatureKeypoints keypoints_high;
  FeatureDescriptors descriptors_high;
  ASSERT_TRUE(extractor_high->Extract(
      low_extraction.image, &keypoints_high, &descriptors_high));

  // Higher min_score should produce strictly fewer keypoints.
  EXPECT_GT(low_extraction.keypoints->size(), 0);
  EXPECT_LT(keypoints_high.size(), low_extraction.keypoints->size());

  // Keypoints from the high-score extraction should be a subset of the
  // low-score extraction, since raising the threshold only removes keypoints.
  for (const auto& kp_high : keypoints_high) {
    bool found = false;
    for (const auto& kp_low : *low_extraction.keypoints) {
      if (kp_high.x == kp_low.x && kp_high.y == kp_low.y) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Keypoint (" << kp_high.x << ", " << kp_high.y
                       << ") from high-score extraction not found in "
                          "low-score extraction";
  }
}

INSTANTIATE_TEST_SUITE_P(AlikedTests,
                         ParameterizedAlikedTests,
                         testing::Values(FeatureExtractorType::ALIKED_N16ROT,
                                         FeatureExtractorType::ALIKED_N32));

}  // namespace
}  // namespace colmap

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

#include "colmap/feature/onnx_matchers.h"

#include "colmap/feature/matcher.h"
#include "colmap/feature/resources.h"
#include "colmap/feature/utils.h"
#include "colmap/math/random.h"
#include "colmap/scene/camera.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

class BruteForceONNXMatcherTest : public testing::Test {
 protected:
  static constexpr int kDescriptorDim = 128;

  void SetUp() override { SetPRNGSeed(42); }

  static FeatureMatchingOptions CreateFeatureMatcherOptions() {
    FeatureMatchingOptions options(FeatureMatcherType::ALIKED_BRUTEFORCE);
    options.use_gpu = false;
    return options;
  }

  static BruteForceONNXMatchingOptions CreateBruteForceONNXMatchingOptions() {
    BruteForceONNXMatchingOptions options;
    options.model_path = kDefaultBruteForceONNXMatcherUri;
    return options;
  }

  static std::shared_ptr<FeatureDescriptors> CreateRandomDescriptors(
      int num_descriptors) {
    const auto descriptors = std::make_shared<FeatureDescriptors>();
    descriptors->type = FeatureExtractorType::ALIKED_N16ROT;
    descriptors->data.resize(num_descriptors, kDescriptorDim * sizeof(float));

    if (num_descriptors == 0) {
      return descriptors;
    }

    FeatureDescriptorsFloatData float_data =
        FeatureDescriptorsFloatData::Random(num_descriptors, kDescriptorDim);
    L2NormalizeFeatureDescriptors(&float_data);

    std::memcpy(descriptors->data.data(),
                float_data.data(),
                num_descriptors * kDescriptorDim * sizeof(float));
    return descriptors;
  }

  static std::shared_ptr<FeatureKeypoints> CreateDummyKeypoints(int count) {
    const auto keypoints = std::make_shared<FeatureKeypoints>(count);
    for (int i = 0; i < count; ++i) {
      (*keypoints)[i].x = static_cast<float>(i);
      (*keypoints)[i].y = static_cast<float>(i);
    }
    return keypoints;
  }
};

TEST_F(BruteForceONNXMatcherTest, SelfMatching) {
  constexpr int kNumDescriptors = 20;

  auto keypoints = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors = CreateRandomDescriptors(kNumDescriptors);

  BruteForceONNXMatchingOptions bf_options =
      CreateBruteForceONNXMatchingOptions();
  bf_options.max_ratio = 0;  // Disable ratio test.
  auto matcher = CreateBruteForceONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), bf_options);

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

TEST_F(BruteForceONNXMatcherTest, RatioTest) {
  constexpr int kNumDescriptors = 20;

  auto keypoints1 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors1 = CreateRandomDescriptors(kNumDescriptors);
  auto keypoints2 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors2 = CreateRandomDescriptors(kNumDescriptors);

  BruteForceONNXMatchingOptions bf_options_no_ratio =
      CreateBruteForceONNXMatchingOptions();
  bf_options_no_ratio.max_ratio = 0;  // Disable ratio test.
  bf_options_no_ratio.cross_check = false;
  bf_options_no_ratio.min_cossim = -1;  // Accept all.
  auto matcher_no_ratio = CreateBruteForceONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), bf_options_no_ratio);

  BruteForceONNXMatchingOptions bf_options_ratio =
      CreateBruteForceONNXMatchingOptions();
  bf_options_ratio.max_ratio = 0.8;  // Enable ratio test.
  bf_options_ratio.cross_check = false;
  bf_options_ratio.min_cossim = -1;  // Accept all.
  auto matcher_ratio = CreateBruteForceONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), bf_options_ratio);

  FeatureMatcher::Image image1{1, nullptr, keypoints1, descriptors1};
  FeatureMatcher::Image image2{2, nullptr, keypoints2, descriptors2};

  FeatureMatches matches_no_ratio, matches_ratio;
  matcher_no_ratio->Match(image1, image2, &matches_no_ratio);
  matcher_ratio->Match(image1, image2, &matches_ratio);

  // Ratio test should filter some matches.
  EXPECT_GT(matches_no_ratio.size(), 0);
  EXPECT_LT(matches_ratio.size(), matches_no_ratio.size());
}

TEST_F(BruteForceONNXMatcherTest, CrossCheck) {
  constexpr int kNumDescriptors = 20;

  auto keypoints1 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors1 = CreateRandomDescriptors(kNumDescriptors);
  auto keypoints2 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors2 = CreateRandomDescriptors(kNumDescriptors);

  BruteForceONNXMatchingOptions bf_options_no_cross =
      CreateBruteForceONNXMatchingOptions();
  bf_options_no_cross.max_ratio = 0;
  bf_options_no_cross.cross_check = false;
  bf_options_no_cross.min_cossim = -1;
  auto matcher_no_cross = CreateBruteForceONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), bf_options_no_cross);

  BruteForceONNXMatchingOptions bf_options_cross =
      CreateBruteForceONNXMatchingOptions();
  bf_options_cross.max_ratio = 0;
  bf_options_cross.cross_check = true;
  bf_options_cross.min_cossim = -1;
  auto matcher_cross = CreateBruteForceONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), bf_options_cross);

  FeatureMatcher::Image image1{1, nullptr, keypoints1, descriptors1};
  FeatureMatcher::Image image2{2, nullptr, keypoints2, descriptors2};

  FeatureMatches matches_no_cross, matches_cross;
  matcher_no_cross->Match(image1, image2, &matches_no_cross);
  matcher_cross->Match(image1, image2, &matches_cross);

  // Cross-check should filter some matches.
  EXPECT_GT(matches_no_cross.size(), 0);
  EXPECT_LT(matches_cross.size(), matches_no_cross.size());
}

TEST_F(BruteForceONNXMatcherTest, MinCossim) {
  constexpr int kNumDescriptors = 20;

  auto keypoints1 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors1 = CreateRandomDescriptors(kNumDescriptors);
  auto keypoints2 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors2 = CreateRandomDescriptors(kNumDescriptors);

  BruteForceONNXMatchingOptions bf_options_low =
      CreateBruteForceONNXMatchingOptions();
  bf_options_low.max_ratio = 0;
  bf_options_low.cross_check = false;
  bf_options_low.min_cossim = -1;  // Accept all.
  auto matcher_low = CreateBruteForceONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), bf_options_low);

  BruteForceONNXMatchingOptions bf_options_high =
      CreateBruteForceONNXMatchingOptions();
  bf_options_high.max_ratio = 0;
  bf_options_high.cross_check = false;
  bf_options_high.min_cossim = 0.9;  // Very strict.
  auto matcher_high = CreateBruteForceONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), bf_options_high);

  FeatureMatcher::Image image1{1, nullptr, keypoints1, descriptors1};
  FeatureMatcher::Image image2{2, nullptr, keypoints2, descriptors2};

  FeatureMatches matches_low, matches_high;
  matcher_low->Match(image1, image2, &matches_low);
  matcher_high->Match(image1, image2, &matches_high);

  // Higher min_cossim should filter more matches.
  EXPECT_GT(matches_low.size(), 0);
  EXPECT_LT(matches_high.size(), matches_low.size());
}

TEST_F(BruteForceONNXMatcherTest, TooFewDescriptors) {
  // Model requires at least 2 descriptors for ratio test.
  auto keypoints1 = CreateDummyKeypoints(1);
  auto descriptors1 = CreateRandomDescriptors(1);
  auto keypoints2 = CreateDummyKeypoints(10);
  auto descriptors2 = CreateRandomDescriptors(10);

  auto matcher = CreateBruteForceONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), CreateBruteForceONNXMatchingOptions());

  FeatureMatcher::Image image1{1, nullptr, keypoints1, descriptors1};
  FeatureMatcher::Image image2{2, nullptr, keypoints2, descriptors2};

  FeatureMatches matches12;
  matcher->Match(image1, image2, &matches12);
  EXPECT_EQ(matches12.size(), 0);

  FeatureMatches matches21;
  matcher->Match(image2, image1, &matches21);
  EXPECT_EQ(matches21.size(), 0);
}

TEST_F(BruteForceONNXMatcherTest, EmptyDescriptors) {
  auto keypoints1 = CreateDummyKeypoints(0);
  auto descriptors1 = CreateRandomDescriptors(0);
  auto keypoints2 = CreateDummyKeypoints(10);
  auto descriptors2 = CreateRandomDescriptors(10);

  auto matcher = CreateBruteForceONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), CreateBruteForceONNXMatchingOptions());

  FeatureMatcher::Image image1{1, nullptr, keypoints1, descriptors1};
  FeatureMatcher::Image image2{2, nullptr, keypoints2, descriptors2};

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);

  // Should return empty matches without crashing.
  EXPECT_EQ(matches.size(), 0);
}

TEST_F(BruteForceONNXMatcherTest, Caching) {
  constexpr int kNumDescriptors = 10;

  // Create three sets of descriptors for images A, B, C.
  auto keypointsA = CreateDummyKeypoints(kNumDescriptors);
  auto descriptorsA = CreateRandomDescriptors(kNumDescriptors);
  auto keypointsB = CreateDummyKeypoints(kNumDescriptors);
  auto descriptorsB = CreateRandomDescriptors(kNumDescriptors);
  auto keypointsC = CreateDummyKeypoints(kNumDescriptors);
  auto descriptorsC = CreateRandomDescriptors(kNumDescriptors);

  BruteForceONNXMatchingOptions bf_options =
      CreateBruteForceONNXMatchingOptions();
  bf_options.max_ratio = 0;
  bf_options.cross_check = false;
  bf_options.min_cossim = -1;
  auto matcher = CreateBruteForceONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), bf_options);

  FeatureMatcher::Image imageA{1, nullptr, keypointsA, descriptorsA};
  FeatureMatcher::Image imageB{2, nullptr, keypointsB, descriptorsB};
  FeatureMatcher::Image imageC{3, nullptr, keypointsC, descriptorsC};

  // Match (A, B).
  FeatureMatches matchesAB1;
  matcher->Match(imageA, imageB, &matchesAB1);
  EXPECT_GT(matchesAB1.size(), 0);

  // Match (A, B) again - should use cached features and produce same results.
  FeatureMatches matchesAB2;
  matcher->Match(imageA, imageB, &matchesAB2);
  ASSERT_EQ(matchesAB1.size(), matchesAB2.size());
  for (size_t i = 0; i < matchesAB1.size(); ++i) {
    EXPECT_EQ(matchesAB1[i].point2D_idx1, matchesAB2[i].point2D_idx1);
    EXPECT_EQ(matchesAB1[i].point2D_idx2, matchesAB2[i].point2D_idx2);
  }

  // Match (B, C) - B should be swapped from slot 2 to slot 1.
  FeatureMatches matchesBC;
  matcher->Match(imageB, imageC, &matchesBC);
  EXPECT_GT(matchesBC.size(), 0);

  // Match (A, B) again - verify correctness after swap.
  FeatureMatches matchesAB3;
  matcher->Match(imageA, imageB, &matchesAB3);
  ASSERT_EQ(matchesAB1.size(), matchesAB3.size());
  for (size_t i = 0; i < matchesAB1.size(); ++i) {
    EXPECT_EQ(matchesAB1[i].point2D_idx1, matchesAB3[i].point2D_idx1);
    EXPECT_EQ(matchesAB1[i].point2D_idx2, matchesAB3[i].point2D_idx2);
  }

  // Self-match A - verify self-matching still works.
  FeatureMatches matchesAA;
  matcher->Match(imageA, imageA, &matchesAA);
  EXPECT_EQ(matchesAA.size(), kNumDescriptors);
  for (const auto& match : matchesAA) {
    EXPECT_EQ(match.point2D_idx1, match.point2D_idx2);
  }
}

class AlikedLightGlueONNXMatcherTest : public testing::Test {
 protected:
  static constexpr int kWidth = 640;
  static constexpr int kHeight = 480;
  static constexpr int kDescriptorDim = 128;

  void SetUp() override { SetPRNGSeed(42); }

  static FeatureMatchingOptions CreateFeatureMatcherOptions() {
    FeatureMatchingOptions options(FeatureMatcherType::ALIKED_LIGHTGLUE);
    options.use_gpu = false;
    return options;
  }

  static LightGlueONNXMatchingOptions CreateLightGlueONNXMatchingOptions() {
    LightGlueONNXMatchingOptions options;
    options.model_path = kDefaultAlikedLightGlueFeatureMatcherUri;
    return options;
  }

  static Camera CreateCamera() {
    Camera camera;
    camera.width = kWidth;
    camera.height = kHeight;
    return camera;
  }

  static std::shared_ptr<FeatureDescriptors> CreateRandomDescriptors(
      int num_descriptors) {
    const auto descriptors = std::make_shared<FeatureDescriptors>();
    descriptors->type = FeatureExtractorType::ALIKED_N16ROT;
    descriptors->data.resize(num_descriptors, kDescriptorDim * sizeof(float));

    if (num_descriptors == 0) {
      return descriptors;
    }

    FeatureDescriptorsFloatData float_data =
        FeatureDescriptorsFloatData::Random(num_descriptors, kDescriptorDim);
    L2NormalizeFeatureDescriptors(&float_data);

    std::memcpy(descriptors->data.data(),
                float_data.data(),
                num_descriptors * kDescriptorDim * sizeof(float));
    return descriptors;
  }

  static std::shared_ptr<FeatureKeypoints> CreateRandomKeypoints(int count) {
    const auto keypoints = std::make_shared<FeatureKeypoints>(count);
    for (int i = 0; i < count; ++i) {
      (*keypoints)[i].x = 0.5f + RandomUniformReal(0.0f, kWidth - 1.0f);
      (*keypoints)[i].y = 0.5f + RandomUniformReal(0.0f, kHeight - 1.0f);
    }
    return keypoints;
  }
};

TEST_F(AlikedLightGlueONNXMatcherTest, SelfMatching) {
  constexpr int kNumKeypoints = 10;

  const auto keypoints = CreateRandomKeypoints(kNumKeypoints);
  const auto descriptors = CreateRandomDescriptors(kNumKeypoints);

  const Camera camera = CreateCamera();

  auto matcher = CreateLightGlueONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), CreateLightGlueONNXMatchingOptions());

  const FeatureMatcher::Image image1{1, &camera, keypoints, descriptors};
  const FeatureMatcher::Image image2{2, &camera, keypoints, descriptors};

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);

  // Self-matching should produce a high number of correct matches.
  // LightGlue is attention-based and may not match every keypoint,
  // but correct matches should map keypoints to themselves.
  EXPECT_GT(matches.size(), kNumKeypoints / 2);
  for (const auto& match : matches) {
    EXPECT_EQ(match.point2D_idx1, match.point2D_idx2);
  }

  PosePrior pose_prior;
  // Orientation 6: Rotate 90 CW. Gravity points to +X.
  pose_prior.gravity = Eigen::Vector3d(1, 0, 0);

  const FeatureMatcher::Image image1_rotated{
      1, &camera, keypoints, descriptors, &pose_prior};
  const FeatureMatcher::Image image2_rotated{
      2, &camera, keypoints, descriptors, &pose_prior};

  matcher->Match(image1_rotated, image2_rotated, &matches);

  // Self-matching with the same pose prior should still produce matches.
  EXPECT_GT(matches.size(), kNumKeypoints / 2);
  for (const auto& match : matches) {
    EXPECT_EQ(match.point2D_idx1, match.point2D_idx2);
  }
}

TEST_F(AlikedLightGlueONNXMatcherTest, MinScoreFiltering) {
  constexpr int kNumKeypoints = 10;

  const auto keypoints1 = CreateRandomKeypoints(kNumKeypoints);
  const auto descriptors1 = CreateRandomDescriptors(kNumKeypoints);
  const auto keypoints2 = CreateRandomKeypoints(kNumKeypoints);
  const auto descriptors2 = CreateRandomDescriptors(kNumKeypoints);

  const Camera camera = CreateCamera();

  LightGlueONNXMatchingOptions options_low =
      CreateLightGlueONNXMatchingOptions();
  options_low.min_score = 0.1;
  auto matcher_low = CreateLightGlueONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), options_low);

  LightGlueONNXMatchingOptions options_high =
      CreateLightGlueONNXMatchingOptions();
  options_high.min_score = 0.9;
  auto matcher_high = CreateLightGlueONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), options_high);

  const FeatureMatcher::Image image1{1, &camera, keypoints1, descriptors1};
  const FeatureMatcher::Image image2{2, &camera, keypoints2, descriptors2};

  FeatureMatches matches_low, matches_high;
  matcher_low->Match(image1, image2, &matches_low);
  matcher_high->Match(image1, image2, &matches_high);

  EXPECT_GT(matches_low.size(), matches_high.size());
}

TEST_F(AlikedLightGlueONNXMatcherTest, EmptyKeypoints) {
  const auto keypoints1 = std::make_shared<FeatureKeypoints>();
  const auto descriptors1 = CreateRandomDescriptors(0);
  const auto keypoints2 = CreateRandomKeypoints(10);
  const auto descriptors2 = CreateRandomDescriptors(10);
  const auto camera = CreateCamera();

  auto matcher = CreateLightGlueONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), CreateLightGlueONNXMatchingOptions());

  const FeatureMatcher::Image image1{1, &camera, keypoints1, descriptors1};
  const FeatureMatcher::Image image2{2, &camera, keypoints2, descriptors2};

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);

  EXPECT_EQ(matches.size(), 0);
}

TEST_F(AlikedLightGlueONNXMatcherTest, Caching) {
  constexpr int kNumKeypoints = 10;

  const auto keypointsA = CreateRandomKeypoints(kNumKeypoints);
  const auto descriptorsA = CreateRandomDescriptors(kNumKeypoints);
  const auto keypointsB = CreateRandomKeypoints(kNumKeypoints);
  const auto descriptorsB = CreateRandomDescriptors(kNumKeypoints);
  const auto keypointsC = CreateRandomKeypoints(kNumKeypoints);
  const auto descriptorsC = CreateRandomDescriptors(kNumKeypoints);

  const Camera camera = CreateCamera();

  auto matcher = CreateLightGlueONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), CreateLightGlueONNXMatchingOptions());

  const FeatureMatcher::Image imageA{1, &camera, keypointsA, descriptorsA};
  const FeatureMatcher::Image imageB{2, &camera, keypointsB, descriptorsB};
  const FeatureMatcher::Image imageC{3, &camera, keypointsC, descriptorsC};

  // Match (A, B).
  FeatureMatches matchesAB1;
  matcher->Match(imageA, imageB, &matchesAB1);

  // Match (A, B) again - should use cached features and produce same results.
  FeatureMatches matchesAB2;
  matcher->Match(imageA, imageB, &matchesAB2);
  ASSERT_EQ(matchesAB1.size(), matchesAB2.size());
  for (size_t i = 0; i < matchesAB1.size(); ++i) {
    EXPECT_EQ(matchesAB1[i].point2D_idx1, matchesAB2[i].point2D_idx1);
    EXPECT_EQ(matchesAB1[i].point2D_idx2, matchesAB2[i].point2D_idx2);
  }

  // Match (B, C) - B should be swapped from slot 2 to slot 1.
  FeatureMatches matchesBC;
  matcher->Match(imageB, imageC, &matchesBC);

  // Match (A, B) again - verify correctness after swap.
  FeatureMatches matchesAB3;
  matcher->Match(imageA, imageB, &matchesAB3);
  ASSERT_EQ(matchesAB1.size(), matchesAB3.size());
  for (size_t i = 0; i < matchesAB1.size(); ++i) {
    EXPECT_EQ(matchesAB1[i].point2D_idx1, matchesAB3[i].point2D_idx1);
    EXPECT_EQ(matchesAB1[i].point2D_idx2, matchesAB3[i].point2D_idx2);
  }
}

class SiftLightGlueONNXMatcherTest : public testing::Test {
 protected:
  static constexpr int kWidth = 640;
  static constexpr int kHeight = 480;
  static constexpr int kDescriptorDim = 128;

  void SetUp() override { SetPRNGSeed(42); }

  static FeatureMatchingOptions CreateFeatureMatcherOptions() {
    FeatureMatchingOptions options(FeatureMatcherType::SIFT_LIGHTGLUE);
    options.use_gpu = false;
    return options;
  }

  static LightGlueONNXMatchingOptions CreateLightGlueONNXMatchingOptions() {
    LightGlueONNXMatchingOptions options;
    options.model_path = kDefaultSiftLightGlueFeatureMatcherUri;
    return options;
  }

  static Camera CreateCamera() {
    Camera camera;
    camera.width = kWidth;
    camera.height = kHeight;
    return camera;
  }

  static std::shared_ptr<FeatureDescriptors> CreateRandomDescriptors(
      int num_descriptors) {
    const auto descriptors = std::make_shared<FeatureDescriptors>();
    descriptors->type = FeatureExtractorType::SIFT;
    descriptors->data.resize(num_descriptors, kDescriptorDim);

    if (num_descriptors == 0) {
      return descriptors;
    }

    for (int i = 0; i < num_descriptors; ++i) {
      for (int j = 0; j < kDescriptorDim; ++j) {
        descriptors->data(i, j) =
            static_cast<uint8_t>(RandomUniformInteger(0, 255));
      }
    }
    return descriptors;
  }

  static std::shared_ptr<FeatureKeypoints> CreateRandomKeypoints(int count) {
    const auto keypoints = std::make_shared<FeatureKeypoints>(count);
    for (int i = 0; i < count; ++i) {
      const float x = 0.5f + RandomUniformReal(0.0f, kWidth - 1.0f);
      const float y = 0.5f + RandomUniformReal(0.0f, kHeight - 1.0f);
      const float scale = 1.0f + RandomUniformReal(0.0f, 5.0f);
      const float orientation = RandomUniformReal(0.0f, 2.0f * 3.14159f);
      (*keypoints)[i] = FeatureKeypoint(x, y, scale, orientation);
    }
    return keypoints;
  }
};

TEST_F(SiftLightGlueONNXMatcherTest, SelfMatching) {
  constexpr int kNumKeypoints = 10;

  const auto keypoints = CreateRandomKeypoints(kNumKeypoints);
  const auto descriptors = CreateRandomDescriptors(kNumKeypoints);

  const Camera camera = CreateCamera();

  auto matcher = CreateLightGlueONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), CreateLightGlueONNXMatchingOptions());

  const FeatureMatcher::Image image1{1, &camera, keypoints, descriptors};
  const FeatureMatcher::Image image2{2, &camera, keypoints, descriptors};

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);

  // Self-matching should produce a high number of correct matches.
  EXPECT_GT(matches.size(), kNumKeypoints / 2);
  for (const auto& match : matches) {
    EXPECT_EQ(match.point2D_idx1, match.point2D_idx2);
  }

  PosePrior pose_prior;
  // Orientation 6: Rotate 90 CW. Gravity points to +X.
  pose_prior.gravity = Eigen::Vector3d(1, 0, 0);

  const FeatureMatcher::Image image1_rotated{
      1, &camera, keypoints, descriptors, &pose_prior};
  const FeatureMatcher::Image image2_rotated{
      2, &camera, keypoints, descriptors, &pose_prior};

  matcher->Match(image1_rotated, image2_rotated, &matches);

  // Self-matching with the same pose prior should still produce matches.
  EXPECT_GT(matches.size(), kNumKeypoints / 2);
  for (const auto& match : matches) {
    EXPECT_EQ(match.point2D_idx1, match.point2D_idx2);
  }
}

TEST_F(SiftLightGlueONNXMatcherTest, MinScoreFiltering) {
  constexpr int kNumKeypoints = 10;

  const auto keypoints1 = CreateRandomKeypoints(kNumKeypoints);
  const auto descriptors1 = CreateRandomDescriptors(kNumKeypoints);
  const auto keypoints2 = CreateRandomKeypoints(kNumKeypoints);
  const auto descriptors2 = CreateRandomDescriptors(kNumKeypoints);

  const Camera camera = CreateCamera();

  LightGlueONNXMatchingOptions options_low =
      CreateLightGlueONNXMatchingOptions();
  options_low.min_score = 0.1;
  auto matcher_low = CreateLightGlueONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), options_low);

  LightGlueONNXMatchingOptions options_high =
      CreateLightGlueONNXMatchingOptions();
  options_high.min_score = 0.9;
  auto matcher_high = CreateLightGlueONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), options_high);

  const FeatureMatcher::Image image1{1, &camera, keypoints1, descriptors1};
  const FeatureMatcher::Image image2{2, &camera, keypoints2, descriptors2};

  FeatureMatches matches_low, matches_high;
  matcher_low->Match(image1, image2, &matches_low);
  matcher_high->Match(image1, image2, &matches_high);

  EXPECT_GT(matches_low.size(), matches_high.size());
}

TEST_F(SiftLightGlueONNXMatcherTest, EmptyKeypoints) {
  const auto keypoints1 = std::make_shared<FeatureKeypoints>();
  const auto descriptors1 = CreateRandomDescriptors(0);
  const auto keypoints2 = CreateRandomKeypoints(10);
  const auto descriptors2 = CreateRandomDescriptors(10);
  const auto camera = CreateCamera();

  auto matcher = CreateLightGlueONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), CreateLightGlueONNXMatchingOptions());

  const FeatureMatcher::Image image1{1, &camera, keypoints1, descriptors1};
  const FeatureMatcher::Image image2{2, &camera, keypoints2, descriptors2};

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);

  EXPECT_EQ(matches.size(), 0);
}

TEST_F(SiftLightGlueONNXMatcherTest, Caching) {
  constexpr int kNumKeypoints = 10;

  const auto keypointsA = CreateRandomKeypoints(kNumKeypoints);
  const auto descriptorsA = CreateRandomDescriptors(kNumKeypoints);
  const auto keypointsB = CreateRandomKeypoints(kNumKeypoints);
  const auto descriptorsB = CreateRandomDescriptors(kNumKeypoints);
  const auto keypointsC = CreateRandomKeypoints(kNumKeypoints);
  const auto descriptorsC = CreateRandomDescriptors(kNumKeypoints);

  const Camera camera = CreateCamera();

  auto matcher = CreateLightGlueONNXFeatureMatcher(
      CreateFeatureMatcherOptions(), CreateLightGlueONNXMatchingOptions());

  const FeatureMatcher::Image imageA{1, &camera, keypointsA, descriptorsA};
  const FeatureMatcher::Image imageB{2, &camera, keypointsB, descriptorsB};
  const FeatureMatcher::Image imageC{3, &camera, keypointsC, descriptorsC};

  // Match (A, B).
  FeatureMatches matchesAB1;
  matcher->Match(imageA, imageB, &matchesAB1);

  // Match (A, B) again - should use cached features and produce same results.
  FeatureMatches matchesAB2;
  matcher->Match(imageA, imageB, &matchesAB2);
  ASSERT_EQ(matchesAB1.size(), matchesAB2.size());
  for (size_t i = 0; i < matchesAB1.size(); ++i) {
    EXPECT_EQ(matchesAB1[i].point2D_idx1, matchesAB2[i].point2D_idx1);
    EXPECT_EQ(matchesAB1[i].point2D_idx2, matchesAB2[i].point2D_idx2);
  }

  // Match (B, C) - B should be swapped from slot 2 to slot 1.
  FeatureMatches matchesBC;
  matcher->Match(imageB, imageC, &matchesBC);

  // Match (A, B) again - verify correctness after swap.
  FeatureMatches matchesAB3;
  matcher->Match(imageA, imageB, &matchesAB3);
  ASSERT_EQ(matchesAB1.size(), matchesAB3.size());
  for (size_t i = 0; i < matchesAB1.size(); ++i) {
    EXPECT_EQ(matchesAB1[i].point2D_idx1, matchesAB3[i].point2D_idx1);
    EXPECT_EQ(matchesAB1[i].point2D_idx2, matchesAB3[i].point2D_idx2);
  }
}

}  // namespace
}  // namespace colmap

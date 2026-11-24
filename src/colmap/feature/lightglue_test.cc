// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/lightglue.h"

#include "colmap/feature/matcher.h"
#include "colmap/feature/resources.h"
#include "colmap/feature/utils.h"
#include "colmap/math/random.h"
#include "colmap/scene/camera.h"

#include <cstring>
#include <memory>

#include <gtest/gtest.h>

namespace colmap {
namespace {

class AlikedLightGlueONNXMatcherTest : public testing::Test {
 protected:
  static constexpr int kWidth = 640;
  static constexpr int kHeight = 480;
  static constexpr int kDescriptorDim = 128;

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

  // One matcher per score threshold, shared across the TEST_Fs below to
  // avoid reloading the ~46MB model per TEST_F. Each TEST_F must use
  // distinct image ids for its image content, since the matcher's feature
  // cache keys by image id. Destroyed in TearDownTestSuite while the test
  // run is still active, since ONNX sessions must not be destroyed during
  // static teardown at process exit.
  static void SetUpTestSuite() {
    matcher_low_ = CreateLightGlueONNXFeatureMatcher(
        CreateFeatureMatcherOptions(), CreateLightGlueONNXMatchingOptions());
    LightGlueONNXMatchingOptions options_high =
        CreateLightGlueONNXMatchingOptions();
    options_high.min_score = 0.9;
    matcher_high_ = CreateLightGlueONNXFeatureMatcher(
        CreateFeatureMatcherOptions(), options_high);
  }

  static void TearDownTestSuite() {
    matcher_low_.reset();
    matcher_high_.reset();
  }

  static std::unique_ptr<FeatureMatcher> matcher_low_;
  static std::unique_ptr<FeatureMatcher> matcher_high_;
};

std::unique_ptr<FeatureMatcher> AlikedLightGlueONNXMatcherTest::matcher_low_;
std::unique_ptr<FeatureMatcher> AlikedLightGlueONNXMatcherTest::matcher_high_;

TEST_F(AlikedLightGlueONNXMatcherTest, SelfMatching) {
  constexpr int kNumKeypoints = 10;

  const auto keypoints = CreateRandomKeypoints(kNumKeypoints);
  const auto descriptors = CreateRandomDescriptors(kNumKeypoints);

  const Camera camera = CreateCamera();

  const FeatureMatcher::Image image1{1, &camera, keypoints, descriptors};
  const FeatureMatcher::Image image2{2, &camera, keypoints, descriptors};

  FeatureMatches matches;
  matcher_low_->Match(image1, image2, &matches);

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

  matcher_low_->Match(image1_rotated, image2_rotated, &matches);

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

  const FeatureMatcher::Image image1{3, &camera, keypoints1, descriptors1};
  const FeatureMatcher::Image image2{4, &camera, keypoints2, descriptors2};

  FeatureMatches matches_low, matches_high;
  matcher_low_->Match(image1, image2, &matches_low);
  matcher_high_->Match(image1, image2, &matches_high);

  EXPECT_GT(matches_low.size(), matches_high.size());
}

TEST_F(AlikedLightGlueONNXMatcherTest, EmptyKeypoints) {
  const auto keypoints1 = std::make_shared<FeatureKeypoints>();
  const auto descriptors1 = CreateRandomDescriptors(0);
  const auto keypoints2 = CreateRandomKeypoints(10);
  const auto descriptors2 = CreateRandomDescriptors(10);
  const auto camera = CreateCamera();

  const FeatureMatcher::Image image1{5, &camera, keypoints1, descriptors1};
  const FeatureMatcher::Image image2{6, &camera, keypoints2, descriptors2};

  FeatureMatches matches;
  matcher_low_->Match(image1, image2, &matches);

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

  // Fresh matcher: this test exercises the feature cache transitions and
  // must not share cache state with the other TEST_Fs.
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

  // One matcher per score threshold, shared across the TEST_Fs below to
  // avoid reloading the ~46MB model per TEST_F. Each TEST_F must use
  // distinct image ids for its image content, since the matcher's feature
  // cache keys by image id. Destroyed in TearDownTestSuite while the test
  // run is still active, since ONNX sessions must not be destroyed during
  // static teardown at process exit.
  static void SetUpTestSuite() {
    matcher_low_ = CreateLightGlueONNXFeatureMatcher(
        CreateFeatureMatcherOptions(), CreateLightGlueONNXMatchingOptions());
    LightGlueONNXMatchingOptions options_high =
        CreateLightGlueONNXMatchingOptions();
    options_high.min_score = 0.9;
    matcher_high_ = CreateLightGlueONNXFeatureMatcher(
        CreateFeatureMatcherOptions(), options_high);
  }

  static void TearDownTestSuite() {
    matcher_low_.reset();
    matcher_high_.reset();
  }

  static std::unique_ptr<FeatureMatcher> matcher_low_;
  static std::unique_ptr<FeatureMatcher> matcher_high_;
};

std::unique_ptr<FeatureMatcher> SiftLightGlueONNXMatcherTest::matcher_low_;
std::unique_ptr<FeatureMatcher> SiftLightGlueONNXMatcherTest::matcher_high_;

TEST_F(SiftLightGlueONNXMatcherTest, SelfMatching) {
  constexpr int kNumKeypoints = 10;

  const auto keypoints = CreateRandomKeypoints(kNumKeypoints);
  const auto descriptors = CreateRandomDescriptors(kNumKeypoints);

  const Camera camera = CreateCamera();

  const FeatureMatcher::Image image1{1, &camera, keypoints, descriptors};
  const FeatureMatcher::Image image2{2, &camera, keypoints, descriptors};

  FeatureMatches matches;
  matcher_low_->Match(image1, image2, &matches);

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

  matcher_low_->Match(image1_rotated, image2_rotated, &matches);

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

  const FeatureMatcher::Image image1{3, &camera, keypoints1, descriptors1};
  const FeatureMatcher::Image image2{4, &camera, keypoints2, descriptors2};

  FeatureMatches matches_low, matches_high;
  matcher_low_->Match(image1, image2, &matches_low);
  matcher_high_->Match(image1, image2, &matches_high);

  EXPECT_GT(matches_low.size(), matches_high.size());
}

TEST_F(SiftLightGlueONNXMatcherTest, EmptyKeypoints) {
  const auto keypoints1 = std::make_shared<FeatureKeypoints>();
  const auto descriptors1 = CreateRandomDescriptors(0);
  const auto keypoints2 = CreateRandomKeypoints(10);
  const auto descriptors2 = CreateRandomDescriptors(10);
  const auto camera = CreateCamera();

  const FeatureMatcher::Image image1{5, &camera, keypoints1, descriptors1};
  const FeatureMatcher::Image image2{6, &camera, keypoints2, descriptors2};

  FeatureMatches matches;
  matcher_low_->Match(image1, image2, &matches);

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

  // Fresh matcher: this test exercises the feature cache transitions and
  // must not share cache state with the other TEST_Fs.
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

// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/loma.h"

#include "colmap/feature/matcher.h"
#include "colmap/math/random.h"
#include "colmap/scene/camera.h"
#include "colmap/sensor/bitmap.h"

#include <cstdlib>
#include <memory>

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

std::unique_ptr<FeatureExtractor> CreateTestExtractor(FeatureExtractorType type,
                                                      double min_score,
                                                      int max_num_features) {
  FeatureExtractionOptions options(type);
  options.use_gpu = false;
  options.loma->min_score = min_score;
  options.loma->max_num_features = max_num_features;
  return CreateLomaFeatureExtractor(options);
}

// The LOMA_B descriptor (DeDoDe-G, ~1.3GB) takes ~10s per CPU inference, so
// the default suite exercises the shared extraction/matching logic through
// LOMA_B128 (same detector, VGG-only descriptor) and only smoke-tests LOMA_B
// when COLMAP_TEST_HEAVY is set (CI sets it for ctest).
//
// Extractor instances are shared across TESTs: each construction reloads the
// detector + descriptor models. They are destroyed in TearDownTestSuite while
// the test run is still active, since ONNX sessions must not be destroyed
// during static teardown at process exit.
class LomaTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    extractor_512_ =
        CreateTestExtractor(FeatureExtractorType::LOMA_B128, 0.0, 512);
    extractor_full_ =
        CreateTestExtractor(FeatureExtractorType::LOMA_B128, 0.0, 2048);
    extractor_high_score_ =
        CreateTestExtractor(FeatureExtractorType::LOMA_B128, 0.9, 2048);
  }

  static void TearDownTestSuite() {
    extractor_512_.reset();
    extractor_full_.reset();
    extractor_high_score_.reset();
  }

  static std::unique_ptr<FeatureExtractor> extractor_512_;
  static std::unique_ptr<FeatureExtractor> extractor_full_;
  static std::unique_ptr<FeatureExtractor> extractor_high_score_;
};

std::unique_ptr<FeatureExtractor> LomaTest::extractor_512_;
std::unique_ptr<FeatureExtractor> LomaTest::extractor_full_;
std::unique_ptr<FeatureExtractor> LomaTest::extractor_high_score_;

void CheckNominalExtractionAndMatching(FeatureExtractor& extractor,
                                       FeatureExtractorType extractor_type,
                                       FeatureMatcherType matcher_type) {
  Bitmap image;
  CreateRandomRgbImage(512, 512, &image);

  auto keypoints = std::make_shared<FeatureKeypoints>();
  auto descriptors = std::make_shared<FeatureDescriptors>();
  ASSERT_TRUE(extractor.Extract(image, keypoints.get(), descriptors.get()));

  EXPECT_GT(keypoints->size(), 0);
  EXPECT_EQ(keypoints->size(), descriptors->data.rows());
  EXPECT_EQ(descriptors->type, extractor_type);

  for (const auto& keypoint : *keypoints) {
    EXPECT_GE(keypoint.x, 0);
    EXPECT_GE(keypoint.y, 0);
    EXPECT_LE(keypoint.x, image.Width());
    EXPECT_LE(keypoint.y, image.Height());
  }

  Camera camera;
  camera.width = image.Width();
  camera.height = image.Height();

  FeatureMatchingOptions matching_options(matcher_type);
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

TEST_F(LomaTest, Nominal) {
  // Attention-based matching is quadratic in the feature count; 512 features
  // exercise the matcher at a fraction of the default-2048 cost.
  CheckNominalExtractionAndMatching(*extractor_512_,
                                    FeatureExtractorType::LOMA_B128,
                                    FeatureMatcherType::LOMA_B128);
}

TEST_F(LomaTest, HeavyNominalLomaB) {
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  if (std::getenv("COLMAP_TEST_HEAVY") == nullptr) {
    GTEST_SKIP() << "Set COLMAP_TEST_HEAVY=1 to run the LOMA_B smoke test.";
  }
  auto extractor = CreateTestExtractor(FeatureExtractorType::LOMA_B, 0.0, 512);
  CheckNominalExtractionAndMatching(
      *extractor, FeatureExtractorType::LOMA_B, FeatureMatcherType::LOMA_B);
}

TEST_F(LomaTest, ScoreFilteringAndDynamicNumKeypoints) {
  Bitmap image;
  CreateRandomRgbImage(256, 256, &image);

  // Score filtering and the detector are shared across variants; use the
  // cheaper LOMA_B128 descriptor (see above). Both checks share the
  // unfiltered full-size extraction to avoid a redundant inference run.
  FeatureKeypoints keypoints_full;
  FeatureDescriptors descriptors_full;
  ASSERT_TRUE(
      extractor_full_->Extract(image, &keypoints_full, &descriptors_full));

  FeatureKeypoints keypoints_high;
  FeatureDescriptors descriptors_high;
  ASSERT_TRUE(extractor_high_score_->Extract(
      image, &keypoints_high, &descriptors_high));
  EXPECT_LE(keypoints_high.size(), keypoints_full.size());

  FeatureKeypoints keypoints_512;
  FeatureDescriptors descriptors_512;
  ASSERT_TRUE(extractor_512_->Extract(image, &keypoints_512, &descriptors_512));

  // With min_score=0 nothing is filtered, so the exact requested count should
  // come back -- proves num_keypoints is a real runtime input to the detector
  // graph (see make_detector_dynamic_k() in deployment/export_onnx.py),
  // not just accepted-but-ignored.
  EXPECT_EQ(keypoints_512.size(), 512);
  EXPECT_EQ(keypoints_full.size(), 2048);
}

TEST_F(LomaTest, Bf16MatcherCpuFallback) {
  FeatureMatchingOptions options(FeatureMatcherType::LOMA_B);
  options.use_gpu = false;
  options.loma->use_bf16 = true;

  EXPECT_NE(CreateLomaFeatureMatcher(options), nullptr);
}

}  // namespace
}  // namespace colmap

// SPDX-License-Identifier: BSD-3-Clause

// Determinism of the cusift extractor and matcher: the same input must give
// bit-identical output on every call, from fresh and from reused instances.
//
// This is a property SiftGPU does not have. Its detector returns different
// keypoint sets for the same image from run to run (colmap#1197), which is why
// its parity case is marked siftgpu_flaky in the pycolmap tests. Cross-process
// repetition, the case colmap#1197 reports, is covered by
// src/pycolmap/feature/cusift_determinism_test.py.

#include "colmap/feature/cusift.h"
#include "colmap/feature/cusift_matcher.h"
#include "colmap/feature/extractor.h"
#include "colmap/feature/matcher.h"
#include "colmap/sensor/bitmap.h"

#include <algorithm>
#include <memory>
#include <random>
#include <utility>

#include <gtest/gtest.h>

namespace colmap {
namespace {

constexpr int kNumRepeats = 5;

// A synthetic scene of many overlapping squares, so that the extractor finds
// a few hundred keypoints across several octaves rather than the handful a
// single square gives. (dx, dy) shifts the whole scene, which yields a second
// view with real correspondences for the matcher tests.
Bitmap CreateTexturedImage(const int width,
                           const int height,
                           const int dx = 0,
                           const int dy = 0) {
  Bitmap bitmap(width, height, /*as_rgb=*/false);
  bitmap.Fill(BitmapColor<uint8_t>(40));
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> pos_x(0, width - 1);
  std::uniform_int_distribution<int> pos_y(0, height - 1);
  std::uniform_int_distribution<int> side(4, 28);
  std::uniform_int_distribution<int> value(70, 255);
  for (int k = 0; k < 150; ++k) {
    const int x0 = pos_x(rng) + dx;
    const int y0 = pos_y(rng) + dy;
    const int s = side(rng);
    const BitmapColor<uint8_t> color(static_cast<uint8_t>(value(rng)));
    for (int y = std::max(y0, 0); y < std::min(y0 + s, height); ++y) {
      for (int x = std::max(x0, 0); x < std::min(x0 + s, width); ++x) {
        bitmap.SetPixel(x, y, color);
      }
    }
  }
  return bitmap;
}

FeatureExtractionOptions CreateCusiftOptions() {
  FeatureExtractionOptions options(FeatureExtractorType::CUSIFT);
  options.use_gpu = true;
  return options;
}

FeatureMatchingOptions CreateCusiftMatchingOptions() {
  FeatureMatchingOptions options(FeatureMatcherType::CUSIFT_GEMM);
  options.use_gpu = true;
  return options;
}

struct Features {
  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
};

Features Extract(FeatureExtractor& extractor, const Bitmap& bitmap) {
  Features features;
  THROW_CHECK(
      extractor.Extract(bitmap, &features.keypoints, &features.descriptors));
  return features;
}

// Exact comparison, no tolerance: every float of every keypoint and every
// descriptor byte must match.
void ExpectIdentical(const Features& expected, const Features& actual) {
  ASSERT_EQ(expected.keypoints.size(), actual.keypoints.size());
  for (size_t i = 0; i < expected.keypoints.size(); ++i) {
    const FeatureKeypoint& a = expected.keypoints[i];
    const FeatureKeypoint& b = actual.keypoints[i];
    ASSERT_TRUE(a.x == b.x && a.y == b.y && a.a11 == b.a11 && a.a12 == b.a12 &&
                a.a21 == b.a21 && a.a22 == b.a22)
        << "keypoint " << i << " differs";
  }
  EXPECT_EQ(expected.descriptors.type, actual.descriptors.type);
  ASSERT_EQ(expected.descriptors.data.rows(), actual.descriptors.data.rows());
  ASSERT_EQ(expected.descriptors.data.cols(), actual.descriptors.data.cols());
  EXPECT_TRUE(expected.descriptors.data == actual.descriptors.data);
}

FeatureMatcher::Image MakeImage(const image_t image_id,
                                const Features& features) {
  return FeatureMatcher::Image{
      image_id,
      /*camera=*/nullptr,
      std::make_shared<FeatureKeypoints>(features.keypoints),
      std::make_shared<FeatureDescriptors>(features.descriptors)};
}

FeatureMatches Match(FeatureMatcher& matcher,
                     const FeatureMatcher::Image& image1,
                     const FeatureMatcher::Image& image2) {
  FeatureMatches matches;
  matcher.Match(image1, image2, &matches);
  return matches;
}

#if defined(COLMAP_CUDA_ENABLED)

TEST(CusiftDeterminism, ExtractionWithFreshExtractors) {
  const Bitmap bitmap = CreateTexturedImage(640, 480);

  auto reference_extractor =
      CreateCusiftFeatureExtractor(CreateCusiftOptions());
  ASSERT_NE(reference_extractor, nullptr);
  const Features reference = Extract(*reference_extractor, bitmap);
  // Guard against a vacuous pass: identical output is only meaningful if
  // there is a substantial amount of it.
  ASSERT_GE(reference.keypoints.size(), 200);

  for (int i = 0; i < kNumRepeats; ++i) {
    SCOPED_TRACE(i);
    auto extractor = CreateCusiftFeatureExtractor(CreateCusiftOptions());
    ASSERT_NE(extractor, nullptr);
    ExpectIdentical(reference, Extract(*extractor, bitmap));
  }
}

TEST(CusiftDeterminism, ExtractionWithReusedExtractor) {
  // cusift keeps grow-only device and pinned host buffers across calls.
  // Interleaving a different, larger image checks that no state from one call
  // leaks into the next.
  const Bitmap bitmap = CreateTexturedImage(640, 480);
  const Bitmap other = CreateTexturedImage(800, 600, /*dx=*/13, /*dy=*/7);

  auto extractor = CreateCusiftFeatureExtractor(CreateCusiftOptions());
  ASSERT_NE(extractor, nullptr);
  const Features reference = Extract(*extractor, bitmap);
  ASSERT_GE(reference.keypoints.size(), 200);

  for (int i = 0; i < kNumRepeats; ++i) {
    SCOPED_TRACE(i);
    Extract(*extractor, other);
    ExpectIdentical(reference, Extract(*extractor, bitmap));
  }
}

class CusiftMatchingDeterminism : public ::testing::Test {
 protected:
  void SetUp() override {
    auto extractor = CreateCusiftFeatureExtractor(CreateCusiftOptions());
    ASSERT_NE(extractor, nullptr);
    features1_ = Extract(*extractor, CreateTexturedImage(640, 480));
    features2_ =
        Extract(*extractor, CreateTexturedImage(640, 480, /*dx=*/11, /*dy=*/5));
    unrelated_ = Extract(*extractor, CreateTexturedImage(480, 640, 200, 150));
  }

  Features features1_;
  Features features2_;
  Features unrelated_;
};

TEST_F(CusiftMatchingDeterminism, FreshMatchers) {
  const FeatureMatcher::Image image1 = MakeImage(1, features1_);
  const FeatureMatcher::Image image2 = MakeImage(2, features2_);

  auto reference_matcher =
      CreateCusiftFeatureMatcher(CreateCusiftMatchingOptions());
  ASSERT_NE(reference_matcher, nullptr);
  const FeatureMatches reference = Match(*reference_matcher, image1, image2);
  ASSERT_GE(reference.size(), 50);

  for (int i = 0; i < kNumRepeats; ++i) {
    SCOPED_TRACE(i);
    auto matcher = CreateCusiftFeatureMatcher(CreateCusiftMatchingOptions());
    ASSERT_NE(matcher, nullptr);
    // Same matches in the same order, not merely the same set.
    EXPECT_EQ(reference, Match(*matcher, image1, image2));
  }
}

TEST_F(CusiftMatchingDeterminism, ReusedMatcher) {
  // The GEMM matcher reuses grow-only buffers across pairs. Matching a pair
  // of a different size in between must not change the next result.
  const FeatureMatcher::Image image1 = MakeImage(1, features1_);
  const FeatureMatcher::Image image2 = MakeImage(2, features2_);
  const FeatureMatcher::Image image3 = MakeImage(3, unrelated_);

  auto matcher = CreateCusiftFeatureMatcher(CreateCusiftMatchingOptions());
  ASSERT_NE(matcher, nullptr);
  const FeatureMatches reference = Match(*matcher, image1, image2);
  ASSERT_GE(reference.size(), 50);

  for (int i = 0; i < kNumRepeats; ++i) {
    SCOPED_TRACE(i);
    Match(*matcher, image3, image1);
    EXPECT_EQ(reference, Match(*matcher, image1, image2));
  }
}

#endif  // COLMAP_CUDA_ENABLED

}  // namespace
}  // namespace colmap

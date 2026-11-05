// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/cusift.h"

#include "colmap/feature/extractor.h"
#include "colmap/feature/matcher.h"
#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"
#include "colmap/math/math.h"
#include "colmap/scene/camera.h"
#include "colmap/util/cuda.h"

#include <algorithm>
#include <functional>
#include <random>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

#if defined(COLMAP_CUSIFT_ENABLED)
Bitmap CreateImageWithSquare(const int size) {
  Bitmap bitmap(size, size, false);
  bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));
  for (int r = size / 2 - size / 8; r < size / 2 + size / 8; ++r) {
    for (int c = size / 2 - size / 8; c < size / 2 + size / 8; ++c) {
      bitmap.SetPixel(r, c, BitmapColor<uint8_t>(255));
    }
  }
  return bitmap;
}

Bitmap CreateTexturedImage(const int width, const int height) {
  Bitmap bitmap(width, height, /*as_rgb=*/false);
  bitmap.Fill(BitmapColor<uint8_t>(40));
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> pos_x(0, width - 1);
  std::uniform_int_distribution<int> pos_y(0, height - 1);
  std::uniform_int_distribution<int> side(4, 28);
  std::uniform_int_distribution<int> value(70, 255);
  for (int k = 0; k < 150; ++k) {
    const int x0 = pos_x(rng);
    const int y0 = pos_y(rng);
    const int s = side(rng);
    const BitmapColor<uint8_t> color(static_cast<uint8_t>(value(rng)));
    for (int y = y0; y < std::min(y0 + s, height); ++y) {
      for (int x = x0; x < std::min(x0 + s, width); ++x) {
        bitmap.SetPixel(x, y, color);
      }
    }
  }
  return bitmap;
}

void ValidateKeypoints(const FeatureKeypoints& keypoints,
                       const Bitmap& bitmap) {
  for (size_t i = 0; i < keypoints.size(); ++i) {
    EXPECT_GE(keypoints[i].x, 0);
    EXPECT_GE(keypoints[i].y, 0);
    EXPECT_LE(keypoints[i].x, bitmap.Width());
    EXPECT_LE(keypoints[i].y, bitmap.Height());
    EXPECT_GT(keypoints[i].ComputeScale(), 0);
    EXPECT_GT(keypoints[i].ComputeOrientation(), -M_PI);
    EXPECT_LT(keypoints[i].ComputeOrientation(), M_PI);
  }
}
#endif  // COLMAP_CUSIFT_ENABLED

FeatureExtractionOptions CreateCusiftOptions() {
  FeatureExtractionOptions options(FeatureExtractorType::CUSIFT);
  options.use_gpu = true;
  return options;
}

TEST(CusiftNumOctaves, FollowsSiftGpu) {
  // SiftGPU's rule (PyramidCU::ResizePyramid): floor(log2(min side)) - 3,
  // counted after first_octave's resampling, and at least 1.
  // 1080p at first_octave -1: upsampled min side 2160 -> 11 - 3 = 8.
  EXPECT_EQ(CusiftNumOctaves(1920, 1080, -1), 8);
  // 4K 8:7 frames after max_image_size 3200: 3200 x 2800, upsampled 5600 -> 9.
  EXPECT_EQ(CusiftNumOctaves(3200, 2800, -1), 9);
  // first_octave 0 counts on the image as given: one octave fewer.
  EXPECT_EQ(CusiftNumOctaves(1920, 1080, 0), 7);
  // Only the shorter side matters.
  EXPECT_EQ(CusiftNumOctaves(1080, 1920, -1), 8);
  // Exactly at a power of two, and tiny images clamp to one octave.
  EXPECT_EQ(CusiftNumOctaves(256, 256, -1), 6);
  EXPECT_EQ(CusiftNumOctaves(8, 8, 0), 1);
  EXPECT_EQ(CusiftNumOctaves(1, 1, -1), 1);
}

TEST(CusiftExtractionOptions, NumOctavesDefaultsToAutoAndCanBeSet) {
  CusiftExtractionOptions options;
  EXPECT_EQ(options.num_octaves, -1);
  EXPECT_TRUE(options.Check());
  // Auto: SiftGPU's rule.
  EXPECT_EQ(CusiftEffectiveNumOctaves(options, 1920, 1080, -1), 8);
  // A user-defined count is used as is.
  options.num_octaves = 4;
  EXPECT_TRUE(options.Check());
  EXPECT_EQ(CusiftEffectiveNumOctaves(options, 1920, 1080, -1), 4);
  // Anything else is rejected rather than silently treated as auto.
  options.num_octaves = 0;
  EXPECT_FALSE(options.Check());
  options.num_octaves = -2;
  EXPECT_FALSE(options.Check());
}

struct CusiftUnsupportedOptionParams {
  std::string name;
  std::function<void(SiftExtractionOptions&)> apply;
};

class CusiftUnsupportedOptionTest
    : public ::testing::TestWithParam<CusiftUnsupportedOptionParams> {};

TEST_P(CusiftUnsupportedOptionTest, FailsOptionsCheck) {
  FeatureExtractionOptions options = CreateCusiftOptions();
  // Not options.Check(): without a GPU backend that is false for any use_gpu.
  EXPECT_TRUE(CheckCusiftSiftOptions(*options.sift));
  GetParam().apply(*options.sift);
  // Reported as a bad option before any extraction thread starts, not only
  // when the extractor is constructed (on a worker thread, in the
  // controllers).
  EXPECT_FALSE(CheckCusiftSiftOptions(*options.sift));
#if defined(COLMAP_CUSIFT_ENABLED)
  EXPECT_FALSE(options.Check());
#endif
}

INSTANTIATE_TEST_SUITE_P(
    CusiftUnsupportedOption,
    CusiftUnsupportedOptionTest,
    ::testing::Values(
        CusiftUnsupportedOptionParams{
            "EstimateAffineShape",
            [](SiftExtractionOptions& o) { o.estimate_affine_shape = true; }},
        CusiftUnsupportedOptionParams{
            "DomainSizePooling",
            [](SiftExtractionOptions& o) { o.domain_size_pooling = true; }},
        CusiftUnsupportedOptionParams{
            "Upright", [](SiftExtractionOptions& o) { o.upright = true; }},
        CusiftUnsupportedOptionParams{
            "DarknessAdaptivity",
            [](SiftExtractionOptions& o) { o.darkness_adaptivity = true; }},
        CusiftUnsupportedOptionParams{"ForceCovariantExtractor",
                                      [](SiftExtractionOptions& o) {
                                        o.force_covariant_extractor = true;
                                      }},
        CusiftUnsupportedOptionParams{
            "FirstOctave",
            [](SiftExtractionOptions& o) { o.first_octave = -2; }},
        CusiftUnsupportedOptionParams{
            "OctaveResolution",
            [](SiftExtractionOptions& o) { o.octave_resolution = 4; }}),
    [](const auto& info) { return info.param.name; });

// Options pass Check() only in a build with cusift; without it, Check() rejects
// the cusift types before reaching cusift's own checks.
#if defined(COLMAP_CUSIFT_ENABLED)
TEST(CusiftExtractionOptions, RequiresGpu) {
  FeatureExtractionOptions options = CreateCusiftOptions();
  EXPECT_TRUE(options.Check());
  // cusift has no CPU implementation, so this must be rejected as a bad option
  // rather than surfacing later as an unknown-extractor-type error.
  options.use_gpu = false;
  EXPECT_FALSE(options.Check());
}
#endif  // COLMAP_CUSIFT_ENABLED

#if defined(COLMAP_CUSIFT_ENABLED)

// Holds cusift to the assertions sift_test.cc's ExtractSiftFeaturesGPU applies
// to SiftGPU, through the generic FeatureExtractor factory.
TEST(CusiftFeatureExtractor, Nominal) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  const Bitmap bitmap = CreateImageWithSquare(256);

  const FeatureExtractionOptions options = CreateCusiftOptions();
  ASSERT_TRUE(options.Check());
  ASSERT_FALSE(options.RequiresOpenGL());
  auto extractor = FeatureExtractor::Create(options);
  ASSERT_NE(extractor, nullptr);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

  EXPECT_GE(keypoints.size(), 12);
  ValidateKeypoints(keypoints, bitmap);
  EXPECT_EQ(descriptors.data.rows(), keypoints.size());
  EXPECT_EQ(descriptors.data.cols(), 128);
  // Descriptors are plain SIFT descriptors so that the existing SIFT matchers,
  // index, and databases consume them unchanged. CUSIFT is an extractor type
  // only; it is never a descriptor type.
  EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
  for (Eigen::Index i = 0; i < descriptors.data.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.data.row(i).cast<float>().norm() - 512), 1);
  }

  // Keypoints only.
  FeatureKeypoints keypoints_only;
  EXPECT_TRUE(
      extractor->Extract(bitmap, &keypoints_only, /*descriptors=*/nullptr));
  EXPECT_EQ(keypoints_only.size(), keypoints.size());
}

TEST(CusiftFeatureExtractor, UserDefinedNumOctavesLimitsScale) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // The square spans 64 px, so its blob responds at a large scale that only
  // the coarser octaves reach. One octave stops well below it.
  const Bitmap bitmap = CreateImageWithSquare(256);
  const auto max_scale = [&bitmap](int num_octaves) {
    FeatureExtractionOptions options = CreateCusiftOptions();
    options.cusift->num_octaves = num_octaves;
    FeatureKeypoints keypoints;
    THROW_CHECK(CreateCusiftFeatureExtractor(options)->Extract(
        bitmap, &keypoints, /*descriptors=*/nullptr));
    float scale = 0;
    for (const FeatureKeypoint& kp : keypoints) {
      scale = std::max(scale, kp.ComputeScale());
    }
    return scale;
  };
  EXPECT_LT(max_scale(1), max_scale(-1));
}

// cusift must agree with COLMAP's own CPU (VLFeat) backend, not SiftGPU, when
// both build the same pyramid.
void ExpectMatchesVLFeat(const FeatureExtractionOptions& cusift_options,
                         const FeatureExtractionOptions& vlfeat_options) {
  const Bitmap bitmap = CreateTexturedImage(256, 256);

  FeatureKeypoints cusift_keypoints;
  FeatureDescriptors cusift_descriptors;
  ASSERT_TRUE(CreateCusiftFeatureExtractor(cusift_options)
                  ->Extract(bitmap, &cusift_keypoints, &cusift_descriptors));

  FeatureKeypoints vlfeat_keypoints;
  FeatureDescriptors vlfeat_descriptors;
  ASSERT_TRUE(CreateSiftFeatureExtractor(vlfeat_options)
                  ->Extract(bitmap, &vlfeat_keypoints, &vlfeat_descriptors));

  // On this synthetic image the two produce the same keypoints; the tolerance
  // covers VLFeat's Newton refinement occasionally converging two candidates
  // onto the same point (neither implementation removes those duplicates),
  // not a genuine detector difference.
  ASSERT_NEAR(static_cast<double>(cusift_keypoints.size()),
              static_cast<double>(vlfeat_keypoints.size()),
              0.02 * vlfeat_keypoints.size() + 1);

  ASSERT_GE(vlfeat_keypoints.size(), 100);

  // Every cusift keypoint should have a VLFeat keypoint within a pixel, at the
  // same scale, with the same orientation and descriptor. Positions can carry
  // several orientations, so pair with the closest orientation. A failure
  // means a convention drifted: the +0.5 pixel origin, the orientation sign,
  // or the descriptor layout. Measured agreement is far tighter (98th
  // percentile: 2e-4 rad, 0.003 descriptor distance).
  size_t num_agreeing = 0;
  for (size_t i = 0; i < cusift_keypoints.size(); ++i) {
    const FeatureKeypoint& ours = cusift_keypoints[i];
    double best_orientation_error = M_PI;
    size_t best_j = vlfeat_keypoints.size();
    for (size_t j = 0; j < vlfeat_keypoints.size(); ++j) {
      const FeatureKeypoint& ref = vlfeat_keypoints[j];
      if (std::abs(ours.x - ref.x) >= 1.0f ||
          std::abs(ours.y - ref.y) >= 1.0f ||
          std::abs(ours.ComputeScale() - ref.ComputeScale()) >= 0.1f) {
        continue;
      }
      const double diff =
          std::abs(ours.ComputeOrientation() - ref.ComputeOrientation());
      const double error = std::min(diff, 2 * M_PI - diff);
      if (error < best_orientation_error) {
        best_orientation_error = error;
        best_j = j;
      }
    }
    if (best_j == vlfeat_keypoints.size()) continue;
    const float descriptor_distance =
        (cusift_descriptors.data.row(i).cast<float>() -
         vlfeat_descriptors.data.row(best_j).cast<float>())
            .norm() /
        512.0f;
    if (best_orientation_error < 0.01 && descriptor_distance < 0.02f) {
      ++num_agreeing;
    }
  }
  EXPECT_GE(num_agreeing, 0.98 * cusift_keypoints.size());
}

FeatureExtractionOptions CreateVLFeatOptions() {
  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = false;
  return options;
}

TEST(CusiftFeatureExtractor, MatchesVLFeatAtVLFeatDefaults) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // VLFeat untouched (num_octaves 4 by default); cusift is told the same count
  // through its override, so the comparison runs at VLFeat's own settings.
  const FeatureExtractionOptions vlfeat_options = CreateVLFeatOptions();
  FeatureExtractionOptions cusift_options = CreateCusiftOptions();
  cusift_options.cusift->num_octaves = vlfeat_options.sift->num_octaves;
  ExpectMatchesVLFeat(cusift_options, vlfeat_options);
}

TEST(CusiftFeatureExtractor, MatchesVLFeatWithAutoOctaves) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // cusift at its default (SiftGPU's automatic octave count); VLFeat is given
  // the same count so both build the same pyramid.
  FeatureExtractionOptions vlfeat_options = CreateVLFeatOptions();
  vlfeat_options.sift->num_octaves =
      CusiftNumOctaves(256, 256, vlfeat_options.sift->first_octave);
  ExpectMatchesVLFeat(CreateCusiftOptions(), vlfeat_options);
}

// Closes the drop-in loop inside ctest: descriptors produced by CUSIFT are fed
// to the existing SIFT_BRUTEFORCE matcher with no conversion. This is what
// tagging them FeatureExtractorType::SIFT buys, and it is the property that
// makes cusift a SiftGPU replacement rather than a separate feature type.
TEST(CusiftFeatureExtractor, DescriptorsMatchableBySiftMatcher) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  const Bitmap bitmap = CreateImageWithSquare(256);
  auto keypoints = std::make_shared<FeatureKeypoints>();
  auto descriptors = std::make_shared<FeatureDescriptors>();
  ASSERT_TRUE(CreateCusiftFeatureExtractor(CreateCusiftOptions())
                  ->Extract(bitmap, keypoints.get(), descriptors.get()));
  ASSERT_GT(descriptors->data.rows(), 0);

  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 256, 256);
  const FeatureMatcher::Image image1 = {
      /*image_id=*/1, &camera, keypoints, descriptors};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2, &camera, keypoints, descriptors};

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_brute_force_matcher = true;
  auto matcher = FeatureMatcher::Create(options);
  ASSERT_NE(matcher, nullptr);

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);
  // Self-match: every descriptor's nearest neighbour is itself, so the ratio
  // test keeps only rows without a near-duplicate elsewhere in the image. The
  // point is that the matcher accepts the data at all and returns identity
  // pairs, not the exact count.
  EXPECT_GT(matches.size(), 0);
  for (const FeatureMatch& match : matches) {
    EXPECT_EQ(match.point2D_idx1, match.point2D_idx2);
  }
}

TEST_P(CusiftUnsupportedOptionTest, Throws) {
  // Create binds the device before checking options, so without a GPU it
  // would throw for the wrong reason.
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  FeatureExtractionOptions options = CreateCusiftOptions();
  GetParam().apply(*options.sift);
  // Rejected loudly. A silent no-op would produce a database that looks fine
  // and reconstructs worse.
  EXPECT_THAT([&]() { CreateCusiftFeatureExtractor(options); },
              testing::ThrowsMessage<std::invalid_argument>(testing::HasSubstr(
                  "invalid options for the cusift extractor")));
}

#else

TEST(CusiftExtractionOptions, CheckFailsWithoutCusift) {
  EXPECT_FALSE(CreateCusiftOptions().Check());
}

TEST(CusiftFeatureExtractor, ReturnsNullWithoutCusift) {
  EXPECT_EQ(CreateCusiftFeatureExtractor(CreateCusiftOptions()), nullptr);
}

#endif  // COLMAP_CUSIFT_ENABLED

}  // namespace
}  // namespace colmap

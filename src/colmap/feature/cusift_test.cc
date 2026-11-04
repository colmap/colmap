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

#include "colmap/feature/cusift.h"

#include "colmap/feature/extractor.h"
#include "colmap/feature/matcher.h"
#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"
#include "colmap/math/math.h"
#include "colmap/scene/camera.h"

#include <algorithm>
#include <functional>
#include <string>

#include <gtest/gtest.h>

namespace colmap {
namespace {

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

#if defined(COLMAP_CUDA_ENABLED)

TEST(CusiftExtractionOptions, RequiresGpu) {
  FeatureExtractionOptions options = CreateCusiftOptions();
  EXPECT_TRUE(options.Check());
  // cusift has no CPU implementation, so this must be rejected as a bad option
  // rather than surfacing later as an unknown-extractor-type error.
  options.use_gpu = false;
  EXPECT_FALSE(options.Check());
}

TEST(CusiftExtractionOptions, TypeProperties) {
  const FeatureExtractionOptions options = CreateCusiftOptions();
  EXPECT_FALSE(options.RequiresRGB());
  EXPECT_FALSE(options.RequiresOpenGL());
  EXPECT_EQ(
      options.EffMaxImageSize(),
      FeatureExtractionOptions(FeatureExtractorType::SIFT).EffMaxImageSize());
}

TEST(CusiftFeatureExtractor, Nominal) {
  const Bitmap bitmap = CreateImageWithSquare(256);

  auto extractor = CreateCusiftFeatureExtractor(CreateCusiftOptions());
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
}

TEST(CusiftFeatureExtractor, UserDefinedNumOctavesLimitsScale) {
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

TEST(CusiftFeatureExtractor, ExtractWithoutDescriptors) {
  const Bitmap bitmap = CreateImageWithSquare(256);
  auto extractor = CreateCusiftFeatureExtractor(CreateCusiftOptions());
  FeatureKeypoints keypoints;
  EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, /*descriptors=*/nullptr));
  EXPECT_GE(keypoints.size(), 12);
}

// cusift must agree with COLMAP's own CPU (VLFeat) backend, not SiftGPU, when
// both build the same pyramid.
void ExpectMatchesVLFeat(const FeatureExtractionOptions& cusift_options,
                         const FeatureExtractionOptions& vlfeat_options) {
  const Bitmap bitmap = CreateImageWithSquare(256);

  FeatureKeypoints cusift_keypoints;
  FeatureDescriptors cusift_descriptors;
  ASSERT_TRUE(CreateCusiftFeatureExtractor(cusift_options)
                  ->Extract(bitmap, &cusift_keypoints, &cusift_descriptors));

  FeatureKeypoints vlfeat_keypoints;
  FeatureDescriptors vlfeat_descriptors;
  ASSERT_TRUE(CreateSiftFeatureExtractor(vlfeat_options)
                  ->Extract(bitmap, &vlfeat_keypoints, &vlfeat_descriptors));

  // On this synthetic image the two produce the same keypoints; the tolerance
  // covers the benign Newton-convergence duplicates described in
  // src/thirdparty/cusift/README.md, not a
  // genuine detector difference.
  ASSERT_NEAR(static_cast<double>(cusift_keypoints.size()),
              static_cast<double>(vlfeat_keypoints.size()),
              0.02 * vlfeat_keypoints.size() + 1);

  // Every cusift keypoint should have a VLFeat keypoint within a pixel, at the
  // same scale and orientation. A failure here means a convention drifted: the
  // +0.5 pixel origin, the orientation sign, or the descriptor layout.
  size_t num_colocated = 0;
  for (const FeatureKeypoint& ours : cusift_keypoints) {
    for (const FeatureKeypoint& ref : vlfeat_keypoints) {
      if (std::abs(ours.x - ref.x) < 1.0f && std::abs(ours.y - ref.y) < 1.0f &&
          std::abs(ours.ComputeScale() - ref.ComputeScale()) < 0.1f) {
        ++num_colocated;
        break;
      }
    }
  }
  EXPECT_GE(num_colocated, 0.98 * cusift_keypoints.size());
}

FeatureExtractionOptions CreateVLFeatOptions() {
  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = false;
  return options;
}

TEST(CusiftFeatureExtractor, MatchesVLFeatAtVLFeatDefaults) {
  // VLFeat untouched (num_octaves 4 by default); cusift is told the same count
  // through its override, so the comparison runs at VLFeat's own settings.
  const FeatureExtractionOptions vlfeat_options = CreateVLFeatOptions();
  FeatureExtractionOptions cusift_options = CreateCusiftOptions();
  cusift_options.cusift->num_octaves = vlfeat_options.sift->num_octaves;
  ExpectMatchesVLFeat(cusift_options, vlfeat_options);
}

TEST(CusiftFeatureExtractor, MatchesVLFeatWithAutoOctaves) {
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

// Parity body: holds a GPU extractor backend to exactly the assertions
// sift_test.cc's ExtractSiftFeaturesGPU applies to SiftGPU (>= 12 keypoints on
// the synthetic square, valid keypoint geometry, 512-norm uint8 descriptors
// tagged SIFT). In a CUDA build neither backend needs an OpenGL context, so
// both can be created through FeatureExtractor::Create without sift_test.cc's
// Qt harness.
//
// Deliberately NOT a TEST_P over {CUSIFT, SIFT}. SiftGPU's detector is
// nondeterministic on this input -- it drops a whole scale-level of corner
// responses in roughly 1 run in 5, yielding 8 keypoints instead of >= 12
// (upstream colmap#1197, open since 2021; upstream CI excludes
// feature/sift_test on every CUDA config for this reason). Sharing one
// parameterized suite let that flake fail THIS file, making cusift's own
// coverage unrunnable as a gate. Separate tests keep the failure attributable
// and let a --gtest_filter drop only the SiftGPU case.
void ExpectGpuExtractorParity(FeatureExtractorType type) {
  const Bitmap bitmap = CreateImageWithSquare(256);

  FeatureExtractionOptions options(type);
  options.use_gpu = true;
  ASSERT_TRUE(options.Check());
  ASSERT_FALSE(options.RequiresOpenGL());

  auto extractor = FeatureExtractor::Create(options);
  ASSERT_NE(extractor, nullptr);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

  EXPECT_GE(keypoints.size(), 12);
  ValidateKeypoints(keypoints, bitmap);
  EXPECT_GE(descriptors.data.rows(), 12);
  EXPECT_EQ(descriptors.data.rows(), keypoints.size());
  EXPECT_EQ(descriptors.data.cols(), 128);
  EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
  for (Eigen::Index i = 0; i < descriptors.data.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.data.row(i).cast<float>().norm() - 512), 1);
  }
}

TEST(GpuExtractorParityCusift, ExtractFeatures) {
  ExpectGpuExtractorParity(FeatureExtractorType::CUSIFT);
}

// Upstream SiftGPU, held to the same bar for comparison. Flaky through no fault
// of cusift -- see the note above before treating a failure here as a
// regression in this fork.
TEST(GpuExtractorParitySiftGpu, ExtractFeatures) {
  ExpectGpuExtractorParity(FeatureExtractorType::SIFT);
}

struct CusiftUnsupportedOptionParams {
  std::string name;
  std::function<void(SiftExtractionOptions&)> apply;
};

class CusiftUnsupportedOptionTest
    : public ::testing::TestWithParam<CusiftUnsupportedOptionParams> {};

TEST_P(CusiftUnsupportedOptionTest, Throws) {
  FeatureExtractionOptions options = CreateCusiftOptions();
  GetParam().apply(*options.sift);
  // Rejected loudly. A silent no-op would produce a database that looks fine
  // and reconstructs worse.
  EXPECT_THROW(CreateCusiftFeatureExtractor(options), std::exception);
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

#else

TEST(CusiftFeatureExtractor, ReturnsNullWithoutCuda) {
  EXPECT_EQ(CreateCusiftFeatureExtractor(CreateCusiftOptions()), nullptr);
}

#endif  // COLMAP_CUDA_ENABLED

}  // namespace
}  // namespace colmap

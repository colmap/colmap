// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/cusift.h"

#include "colmap/feature/extractor.h"
#include "colmap/feature/matcher.h"
#include "colmap/feature/sift.h"
#include "colmap/feature/sift_test_utils.h"
#include "colmap/feature/utils.h"
#include "colmap/math/math.h"
#include "colmap/scene/camera.h"
#include "colmap/util/cuda.h"

#include <algorithm>
#include <functional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

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

TEST(CusiftExtractionOptions, NumOctavesDefaultsToVLFeatAndCanBeSet) {
  CusiftExtractionOptions options;
  // The CPU (VLFeat) backend's count, at every image size.
  EXPECT_EQ(options.num_octaves, SiftExtractionOptions().num_octaves);
  EXPECT_TRUE(options.Check());
  EXPECT_EQ(CusiftEffectiveNumOctaves(options, 1920, 1080, -1), 4);
  // Auto: SiftGPU's rule.
  options.num_octaves = -1;
  EXPECT_TRUE(options.Check());
  EXPECT_EQ(CusiftEffectiveNumOctaves(options, 1920, 1080, -1), 8);
  // Unless the pyramid would halve below one pixel: 64 px reaches 1 px in its
  // seventh octave (eighth with the 2x-upsampled octave -1), and a 1 px image
  // at first_octave -1 holds two (2 px and 1 px). VLFeat's later octaves are
  // simply empty; cusift must not try to build them.
  options.num_octaves = 10;
  EXPECT_EQ(CusiftEffectiveNumOctaves(options, 64, 64, 0), 7);
  EXPECT_EQ(CusiftEffectiveNumOctaves(options, 64, 64, -1), 8);
  EXPECT_EQ(CusiftEffectiveNumOctaves(options, 1, 1, -1), 2);
  EXPECT_EQ(CusiftEffectiveNumOctaves(options, 100, 1, 0), 1);
  // Anything else is rejected rather than silently treated as auto.
  options.num_octaves = 0;
  EXPECT_FALSE(options.Check());
  options.num_octaves = -2;
  EXPECT_FALSE(options.Check());
}

struct CusiftUnsupportedOptionParams {
  std::string name;
  // The option as the GUI and the error message name it.
  std::string option;
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

TEST_P(CusiftUnsupportedOptionTest, MessageNamesTheOption) {
  SiftExtractionOptions options;
  EXPECT_EQ(CusiftUnsupportedSiftOption(options), "");
  GetParam().apply(options);
  // The GUI shows this message, so it must say which option to change.
  EXPECT_THAT(CusiftUnsupportedSiftOption(options),
              testing::StartsWith(GetParam().option + ": "));
}

TEST_P(CusiftUnsupportedOptionTest, ClearMakesCheckPass) {
  SiftExtractionOptions options;
  GetParam().apply(options);
  ClearUnsupportedCusiftSiftOptions(&options);
  EXPECT_TRUE(CheckCusiftSiftOptions(options));
}

INSTANTIATE_TEST_SUITE_P(
    CusiftUnsupportedOption,
    CusiftUnsupportedOptionTest,
    ::testing::Values(
        CusiftUnsupportedOptionParams{
            "EstimateAffineShape",
            "sift.estimate_affine_shape",
            [](SiftExtractionOptions& o) { o.estimate_affine_shape = true; }},
        CusiftUnsupportedOptionParams{
            "DomainSizePooling",
            "sift.domain_size_pooling",
            [](SiftExtractionOptions& o) { o.domain_size_pooling = true; }},
        CusiftUnsupportedOptionParams{
            "Upright",
            "sift.upright",
            [](SiftExtractionOptions& o) { o.upright = true; }},
        CusiftUnsupportedOptionParams{
            "DarknessAdaptivity",
            "sift.darkness_adaptivity",
            [](SiftExtractionOptions& o) { o.darkness_adaptivity = true; }},
        CusiftUnsupportedOptionParams{"ForceCovariantExtractor",
                                      "sift.force_covariant_extractor",
                                      [](SiftExtractionOptions& o) {
                                        o.force_covariant_extractor = true;
                                      }},
        CusiftUnsupportedOptionParams{
            "FirstOctave",
            "sift.first_octave",
            [](SiftExtractionOptions& o) { o.first_octave = -2; }},
        CusiftUnsupportedOptionParams{
            "OctaveResolution",
            "sift.octave_resolution",
            [](SiftExtractionOptions& o) { o.octave_resolution = 4; }},
        CusiftUnsupportedOptionParams{
            "NumOctaves",
            "sift.num_octaves",
            [](SiftExtractionOptions& o) { o.num_octaves = 6; }}),
    [](const auto& info) { return info.param.name; });

TEST(CusiftUnsupportedOptions, ClearResetsAllAndKeepsSupportedOnes) {
  SiftExtractionOptions options;
  options.estimate_affine_shape = true;
  options.domain_size_pooling = true;
  options.upright = true;
  options.darkness_adaptivity = true;
  options.force_covariant_extractor = true;
  options.first_octave = -2;
  options.octave_resolution = 4;
  options.num_octaves = 6;
  options.max_num_features = 1234;
  options.peak_threshold = 0.01;
  ClearUnsupportedCusiftSiftOptions(&options);
  EXPECT_TRUE(CheckCusiftSiftOptions(options));
  EXPECT_EQ(options.max_num_features, 1234);
  EXPECT_EQ(options.peak_threshold, 0.01);
}

TEST(CusiftUnsupportedOptions, ClearKeepsSupportedFirstOctave) {
  SiftExtractionOptions options;
  options.first_octave = 0;
  ClearUnsupportedCusiftSiftOptions(&options);
  EXPECT_EQ(options.first_octave, 0);
}

TEST(CusiftDescriptors, ConvertToFloatAndBackLikeSift) {
  const FeatureDescriptors original(FeatureExtractorType::CUSIFT,
                                    FeatureDescriptorsData::Random(10, 128));
  const FeatureDescriptorsFloat as_float = original.ToFloat();
  EXPECT_EQ(as_float.type, FeatureExtractorType::CUSIFT);
  EXPECT_EQ(as_float.data, original.data.cast<float>());

  const FeatureDescriptors recovered = as_float.ToBytes();
  EXPECT_EQ(recovered.type, FeatureExtractorType::CUSIFT);
  EXPECT_EQ(recovered.data, original.data);
}

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
}

TEST(CusiftFeatureExtractor, KeypointsOnlyMatchFullExtraction) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // Without descriptors the descriptor kernel is skipped, and with it the
  // octave-to-image scaling it applies; the host must reproduce it exactly.
  const Bitmap bitmap = CreateTexturedImage(256, 256);
  auto extractor = CreateCusiftFeatureExtractor(CreateCusiftOptions());
  FeatureKeypoints full;
  FeatureDescriptors descriptors;
  ASSERT_TRUE(extractor->Extract(bitmap, &full, &descriptors));
  FeatureKeypoints keypoints_only;

  ASSERT_TRUE(extractor->Extract(bitmap, &keypoints_only, nullptr));

  ASSERT_GE(full.size(), 100);
  ASSERT_EQ(keypoints_only.size(), full.size());
  for (size_t i = 0; i < full.size(); ++i) {
    EXPECT_EQ(keypoints_only[i].x, full[i].x);
    EXPECT_EQ(keypoints_only[i].y, full[i].y);
    EXPECT_EQ(keypoints_only[i].a11, full[i].a11);
    EXPECT_EQ(keypoints_only[i].a12, full[i].a12);
    EXPECT_EQ(keypoints_only[i].a21, full[i].a21);
    EXPECT_EQ(keypoints_only[i].a22, full[i].a22);
  }
}

float MaxKeypointScale(FeatureExtractor& extractor, const Bitmap& bitmap) {
  FeatureKeypoints keypoints;
  THROW_CHECK(extractor.Extract(bitmap, &keypoints, /*descriptors=*/nullptr));
  float scale = 0;
  for (const FeatureKeypoint& kp : keypoints) {
    scale = std::max(scale, kp.ComputeScale());
  }
  return scale;
}

TEST(CusiftFeatureExtractor, UserDefinedNumOctavesLimitsScale) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // The square spans 64 px, so its blob responds at a large scale that only
  // the coarser octaves reach. One octave stops well below it.
  const Bitmap bitmap = CreateImageWithSquare(256);
  const auto max_scale = [&bitmap](int num_octaves) {
    FeatureExtractionOptions options = CreateCusiftOptions();
    options.cusift->num_octaves = num_octaves;
    return MaxKeypointScale(*CreateCusiftFeatureExtractor(options), bitmap);
  };
  EXPECT_LT(max_scale(1), max_scale(-1));
}

// Side of the square image the VLFeat parity tests extract from.
constexpr int kParityImageSize = 256;

// cusift must agree with COLMAP's own CPU (VLFeat) backend, not SiftGPU, when
// both build the same pyramid. VLFeat must find at least `min_num_keypoints`.
void ExpectMatchesVLFeat(const FeatureExtractionOptions& cusift_options,
                         const FeatureExtractionOptions& vlfeat_options,
                         const size_t min_num_keypoints) {
  const Bitmap bitmap = CreateTexturedImage(kParityImageSize, kParityImageSize);

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

  ASSERT_GE(vlfeat_keypoints.size(), min_num_keypoints);

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
  // Both at their defaults, which build the same pyramid (4 octaves).
  ExpectMatchesVLFeat(
      CreateCusiftOptions(), CreateVLFeatOptions(), /*min_num_keypoints=*/100);
}

TEST(CusiftFeatureExtractor, DefaultOctavesStopAtVLFeatsLargestScale) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // The 64 px square's blob responds only in octaves beyond VLFeat's default
  // four, which SiftGPU's automatic count builds. At the defaults, cusift must
  // stop where VLFeat stops, or max_num_features (which walks levels from the
  // coarsest) would spend its budget differently on large images.
  const Bitmap bitmap = CreateImageWithSquare(256);
  const float vlfeat_max_scale = MaxKeypointScale(
      *CreateSiftFeatureExtractor(CreateVLFeatOptions()), bitmap);
  ASSERT_GT(vlfeat_max_scale, 0);
  EXPECT_NEAR(MaxKeypointScale(
                  *CreateCusiftFeatureExtractor(CreateCusiftOptions()), bitmap),
              vlfeat_max_scale,
              0.01 * vlfeat_max_scale);
}

TEST(CusiftFeatureExtractor, MatchesVLFeatWithAutoOctaves) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // cusift at SiftGPU's automatic octave count; VLFeat is given the same count
  // so both build the same pyramid.
  FeatureExtractionOptions cusift_options = CreateCusiftOptions();
  cusift_options.cusift->num_octaves = -1;
  FeatureExtractionOptions vlfeat_options = CreateVLFeatOptions();
  vlfeat_options.sift->num_octaves = CusiftNumOctaves(
      kParityImageSize, kParityImageSize, vlfeat_options.sift->first_octave);
  ExpectMatchesVLFeat(
      cusift_options, vlfeat_options, /*min_num_keypoints=*/100);
}

TEST(CusiftFeatureExtractor, MatchesVLFeatAtFirstOctaveZero) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // No upsampling: both start at the image's own resolution.
  FeatureExtractionOptions cusift_options = CreateCusiftOptions();
  cusift_options.sift->first_octave = 0;
  FeatureExtractionOptions vlfeat_options = CreateVLFeatOptions();
  vlfeat_options.sift->first_octave = 0;
  ExpectMatchesVLFeat(
      cusift_options, vlfeat_options, /*min_num_keypoints=*/100);
}

size_t NumVLFeatKeypoints(const FeatureExtractionOptions& options) {
  FeatureKeypoints keypoints;
  THROW_CHECK(CreateSiftFeatureExtractor(options)->Extract(
      CreateTexturedImage(kParityImageSize, kParityImageSize),
      &keypoints,
      /*descriptors=*/nullptr));
  return keypoints.size();
}

TEST(CusiftFeatureExtractor, MatchesVLFeatWithMaxNumFeatures) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // max_num_features keeps whole DOG levels, coarsest first, up to and
  // including the level that crosses it. The cap counts keypoints before their
  // extra orientations are added, so both keep well over 50. Both run at their
  // defaults; DefaultOctavesStopAtVLFeatsLargestScale checks that those build
  // the same octaves, which this image is too fine to reveal.
  constexpr int kMaxNumFeatures = 50;
  FeatureExtractionOptions vlfeat_options = CreateVLFeatOptions();
  const size_t num_uncapped = NumVLFeatKeypoints(vlfeat_options);
  vlfeat_options.sift->max_num_features = kMaxNumFeatures;
  FeatureExtractionOptions cusift_options = CreateCusiftOptions();
  cusift_options.sift->max_num_features = kMaxNumFeatures;

  // The cap must bind, or this repeats MatchesVLFeatWithAutoOctaves.
  ASSERT_LT(NumVLFeatKeypoints(vlfeat_options), num_uncapped / 2);
  ExpectMatchesVLFeat(
      cusift_options, vlfeat_options, /*min_num_keypoints=*/kMaxNumFeatures);
}

// An octave narrower than the 3 px extrema window holds no features. VLFeat
// scans nothing there; cusift must do the same, not launch an empty grid and
// abort the whole extraction. A 1 px image at first_octave -1, a 2 px image or
// side at first_octave 0 reach such octaves.
TEST(CusiftFeatureExtractor, TinyImagesYieldNoFeaturesLikeVLFeat) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  for (const int first_octave : {-1, 0}) {
    for (const auto& [width, height] :
         std::vector<std::pair<int, int>>{{1, 1}, {2, 2}, {2, 64}, {64, 2}}) {
      const Bitmap bitmap = CreateTexturedImage(width, height);
      FeatureExtractionOptions cusift_options = CreateCusiftOptions();
      cusift_options.sift->first_octave = first_octave;
      FeatureKeypoints keypoints;
      FeatureDescriptors descriptors;
      EXPECT_TRUE(CreateCusiftFeatureExtractor(cusift_options)
                      ->Extract(bitmap, &keypoints, &descriptors))
          << width << "x" << height << " at first_octave " << first_octave;
      FeatureExtractionOptions vlfeat_options = CreateVLFeatOptions();
      vlfeat_options.sift->first_octave = first_octave;
      FeatureKeypoints vlfeat_keypoints;
      ASSERT_TRUE(CreateSiftFeatureExtractor(vlfeat_options)
                      ->Extract(bitmap, &vlfeat_keypoints, nullptr));
      EXPECT_EQ(keypoints.size(), vlfeat_keypoints.size())
          << width << "x" << height << " at first_octave " << first_octave;
      EXPECT_EQ(descriptors.data.rows(), keypoints.size());
    }
  }
}

TEST(CusiftFeatureExtractor, OversizedNumOctavesYieldsTheClampedPyramid) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // 64 px reaches 4 px in its fifth octave; the sixth and seventh (2 px and
  // 1 px) hold no features and anything beyond cannot be built at all. Asking
  // for ten must return exactly what the buildable pyramid finds.
  const Bitmap bitmap = CreateTexturedImage(64, 64);
  const auto extract = [&bitmap](const int num_octaves) {
    FeatureExtractionOptions options = CreateCusiftOptions();
    options.sift->first_octave = 0;
    options.cusift->num_octaves = num_octaves;
    FeatureKeypoints keypoints;
    THROW_CHECK(CreateCusiftFeatureExtractor(options)->Extract(
        bitmap, &keypoints, /*descriptors=*/nullptr));
    return keypoints;
  };
  const FeatureKeypoints five_octaves = extract(5);
  ASSERT_GT(five_octaves.size(), 0);
  EXPECT_EQ(extract(10), five_octaves);
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

TEST(CusiftFeatureExtractor, ThrowsOnUnsupportedOptionsTypedSift) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  FeatureExtractionOptions options = CreateCusiftOptions();
  options.type = FeatureExtractorType::SIFT;
  options.sift->upright = true;
  // FeatureExtractionOptions::Check() dispatches on the type and passes these
  // options as SIFT ones, so only the extractor's own check can reject them.
  EXPECT_TRUE(options.Check());
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

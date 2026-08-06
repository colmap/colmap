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

#include <gtest/gtest.h>

#if defined(COLMAP_GUI_ENABLED)
#include <QApplication>
#else
#include "colmap/exe/gui.h"
#endif

#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/math/random.h"
#include "colmap/util/opengl_utils.h"

#include <functional>
#include <set>
#include <utility>

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

// Helper to create empty descriptors for testing.
FeatureDescriptors CreateEmptyDescriptors() {
  return FeatureDescriptors(FeatureExtractorType::SIFT,
                            FeatureDescriptorsData(0, 128));
}

// Helper to create reversed descriptors for testing matcher symmetry.
FeatureDescriptors CreateReversedDescriptors(const FeatureDescriptors& src) {
  return FeatureDescriptors(src.type, src.data.colwise().reverse());
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

void ValidateDescriptorNorms(const FeatureDescriptors& descriptors) {
  EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
  for (Eigen::Index i = 0; i < descriptors.data.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.data.row(i).cast<float>().norm() - 512), 1);
  }
}

void ExpectReversedMatches(const FeatureMatches& matches) {
  EXPECT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0].point2D_idx1, 0);
  EXPECT_EQ(matches[0].point2D_idx2, 1);
  EXPECT_EQ(matches[1].point2D_idx1, 1);
  EXPECT_EQ(matches[1].point2D_idx2, 0);
}

void ExpectReversedInlierMatches(const TwoViewGeometry& two_view_geometry) {
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 2);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx1, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx2, 0);
}

TwoViewGeometry CreatePlanarTwoViewGeometry() {
  TwoViewGeometry tvg;
  tvg.config = TwoViewGeometry::PLANAR_OR_PANORAMIC;
  tvg.H = Eigen::Matrix3d::Identity();
  return tvg;
}

void RunGpuTest(std::function<void()> test_body) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  class TestThread : public Thread {
   public:
    std::function<void()> body;

   private:
    void Run() {
      opengl_context_.MakeCurrent();
      body();
    }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  thread.body = std::move(test_body);
  RunThreadWithOpenGLContext(&thread);
}

struct SiftCpuExtractionParams {
  std::string name;
  bool estimate_affine_shape;
  bool domain_size_pooling;
  bool force_covariant_extractor;
  bool upright;
  size_t expected_keypoints;
};

class SiftCpuExtractionTest
    : public ::testing::TestWithParam<SiftCpuExtractionParams> {};

TEST_P(SiftCpuExtractionTest, Nominal) {
  const auto& p = GetParam();
  const Bitmap bitmap = CreateImageWithSquare(256);

  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = false;
  options.sift->estimate_affine_shape = p.estimate_affine_shape;
  options.sift->domain_size_pooling = p.domain_size_pooling;
  options.sift->force_covariant_extractor = p.force_covariant_extractor;
  options.sift->upright = p.upright;
  auto extractor = CreateSiftFeatureExtractor(options);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

  EXPECT_EQ(keypoints.size(), p.expected_keypoints);
  ValidateKeypoints(keypoints, bitmap);
  EXPECT_EQ(descriptors.data.rows(), p.expected_keypoints);
  ValidateDescriptorNorms(descriptors);
}

INSTANTIATE_TEST_SUITE_P(
    SiftCpuExtraction,
    SiftCpuExtractionTest,
    ::testing::Values(
        SiftCpuExtractionParams{"Sift", false, false, false, false, 22},
        SiftCpuExtractionParams{"CovariantSift", false, false, true, false, 22},
        SiftCpuExtractionParams{
            "CovariantAffineSift", true, false, false, false, 22},
        SiftCpuExtractionParams{
            "CovariantAffineSiftUpright", true, false, false, true, 10},
        SiftCpuExtractionParams{
            "CovariantDSPSift", false, true, false, false, 22},
        SiftCpuExtractionParams{
            "CovariantAffineDSPSift", true, true, false, false, 22}),
    [](const auto& info) { return info.param.name; });

TEST(ExtractSiftFeaturesGPU, Nominal) {
  RunGpuTest([] {
    const Bitmap bitmap = CreateImageWithSquare(256);

    FeatureExtractionOptions options(FeatureExtractorType::SIFT);
    options.use_gpu = true;
    options.sift->estimate_affine_shape = false;
    options.sift->domain_size_pooling = false;
    options.sift->force_covariant_extractor = false;
    auto extractor = CreateSiftFeatureExtractor(options);

    FeatureKeypoints keypoints;
    FeatureDescriptors descriptors;
    EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

    EXPECT_GE(keypoints.size(), 12);
    ValidateKeypoints(keypoints, bitmap);
    EXPECT_GE(descriptors.data.rows(), 12);
    ValidateDescriptorNorms(descriptors);
  });
}

FeatureDescriptors CreateRandomFeatureDescriptors(const size_t num_features) {
  SetPRNGSeed(0);
  FeatureDescriptorsFloatData descriptors_float =
      FeatureDescriptorsFloatData::Zero(num_features, 128);
  std::vector<int> dims(128);
  std::iota(dims.begin(), dims.end(), 0);
  for (size_t i = 0; i < num_features; ++i) {
    std::shuffle(dims.begin(), dims.end(), *PRNG);
    for (size_t j = 0; j < 10; ++j) {
      descriptors_float(i, dims[j]) = 1.0f;
    }
  }
  L2NormalizeFeatureDescriptors(&descriptors_float);
  return FeatureDescriptors(
      FeatureExtractorType::SIFT,
      FeatureDescriptorsToUnsignedByte(descriptors_float));
}

void CheckEqualMatches(const FeatureMatches& matches1,
                       const FeatureMatches& matches2) {
  ASSERT_EQ(matches1.size(), matches2.size());
  for (size_t i = 0; i < matches1.size(); ++i) {
    EXPECT_EQ(matches1[i].point2D_idx1, matches2[i].point2D_idx1);
    EXPECT_EQ(matches1[i].point2D_idx2, matches2[i].point2D_idx2);
  }
}

TEST(CreateSiftGPUMatcherOpenGL, Nominal) {
  RunGpuTest([] {
    FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
    options.use_gpu = true;
    options.max_num_matches = 1000;
    EXPECT_NE(CreateSiftFeatureMatcher(options), nullptr);
  });
}

TEST(CreateSiftGPUMatcherCUDA, Nominal) {
#if defined(COLMAP_CUDA_ENABLED)
  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = true;
  options.gpu_index = "0";
  options.max_num_matches = 1000;
  EXPECT_NE(CreateSiftFeatureMatcher(options), nullptr);
#endif
}

struct FeatureDescriptorIndexCacheHelper {
  explicit FeatureDescriptorIndexCacheHelper(
      const std::vector<FeatureMatcher::Image>& images)
      : index_cache(100, [this](const image_t image_id) {
          auto index = FeatureDescriptorIndex::Create();
          const auto& desc = this->image_descriptors_.at(image_id);
          index->Build(desc->ToFloat());
          return index;
        }) {
    for (const auto& image : images) {
      image_descriptors_.emplace(image.image_id, image.descriptors);
    }
  }

  ThreadSafeLRUCache<image_t, FeatureDescriptorIndex> index_cache;

 private:
  std::map<image_t, std::shared_ptr<const FeatureDescriptors>>
      image_descriptors_;
};

TEST(SiftCPUFeatureMatcher, Nominal) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  const FeatureMatcher::Image image0 = {
      /*image_id=*/0,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  FeatureDescriptorIndexCacheHelper index_cache_helper(
      {image0, image1, image2});

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_brute_force_matcher = false;
  options.sift->cpu_descriptor_index_cache = &index_cache_helper.index_cache;
  auto matcher = CreateSiftFeatureMatcher(options);

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);
  ExpectReversedMatches(matches);

  matcher->Match(image1, image2, &matches);
  ExpectReversedMatches(matches);

  matcher->Match(image0, image2, &matches);
  EXPECT_EQ(matches.size(), 0);
  matcher->Match(image1, image0, &matches);
  EXPECT_EQ(matches.size(), 0);
  matcher->Match(image0, image0, &matches);
  EXPECT_EQ(matches.size(), 0);
}

TEST(SiftCPUFeatureMatcher, TypeMismatch) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);

  FeatureDescriptors sift_desc = CreateRandomFeatureDescriptors(2);
  ASSERT_EQ(sift_desc.type, FeatureExtractorType::SIFT);

  FeatureDescriptors undefined_desc = CreateRandomFeatureDescriptors(2);
  undefined_desc.type = FeatureExtractorType::UNDEFINED;

  const FeatureMatcher::Image image_sift = {
      /*image_id=*/1,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(sift_desc)};
  const FeatureMatcher::Image image_undefined = {
      /*image_id=*/2,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(undefined_desc)};

  FeatureDescriptorIndexCacheHelper index_cache_helper(
      {image_sift, image_undefined});

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_brute_force_matcher = true;
  auto matcher = CreateSiftFeatureMatcher(options);

  FeatureMatches matches;
  EXPECT_THROW(matcher->Match(image_sift, image_undefined, &matches),
               std::invalid_argument);
}

TEST(MatchGuidedSiftFeaturesCPU, TypeMismatch) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);

  FeatureDescriptors sift_desc = CreateRandomFeatureDescriptors(2);
  ASSERT_EQ(sift_desc.type, FeatureExtractorType::SIFT);

  FeatureDescriptors undefined_desc = CreateRandomFeatureDescriptors(2);
  undefined_desc.type = FeatureExtractorType::UNDEFINED;

  const FeatureMatcher::Image image_sift = {
      /*image_id=*/1,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{1, 0}, {2, 0}}),
      std::make_shared<FeatureDescriptors>(sift_desc)};
  const FeatureMatcher::Image image_undefined = {
      /*image_id=*/2,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{2, 0}, {1, 0}}),
      std::make_shared<FeatureDescriptors>(undefined_desc)};

  FeatureDescriptorIndexCacheHelper index_cache_helper(
      {image_sift, image_undefined});

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_brute_force_matcher = true;
  auto matcher = CreateSiftFeatureMatcher(options);

  TwoViewGeometry two_view_geometry = CreatePlanarTwoViewGeometry();

  EXPECT_THROW(matcher->MatchGuided(
                   1.0, image_sift, image_undefined, &two_view_geometry),
               std::invalid_argument);
}

TEST(MatchSiftFeaturesGPU, TypeMismatch) {
  RunGpuTest([] {
    FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
    options.use_gpu = true;
    options.max_num_matches = 1000;
    auto matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));

    const Camera camera = Camera::CreateFromModelId(
        1, CameraModelId::kSimplePinhole, 100.0, 100, 200);

    FeatureDescriptors sift_desc = CreateRandomFeatureDescriptors(2);
    FeatureDescriptors undefined_desc = CreateRandomFeatureDescriptors(2);
    undefined_desc.type = FeatureExtractorType::UNDEFINED;

    const FeatureMatcher::Image image_sift = {
        /*image_id=*/1,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(sift_desc)};
    const FeatureMatcher::Image image_undefined = {
        /*image_id=*/2,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(undefined_desc)};

    FeatureMatches matches;
    EXPECT_THROW(matcher->Match(image_sift, image_undefined, &matches),
                 std::invalid_argument);
  });
}

TEST(MatchGuidedSiftFeaturesGPU, TypeMismatch) {
  RunGpuTest([] {
    FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
    options.use_gpu = true;
    options.max_num_matches = 1000;
    auto matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));

    Camera camera = Camera::CreateFromModelId(
        1, CameraModelId::kSimpleRadial, 100.0, 100, 200);

    FeatureDescriptors sift_desc = CreateRandomFeatureDescriptors(2);
    FeatureDescriptors undefined_desc = CreateRandomFeatureDescriptors(2);
    undefined_desc.type = FeatureExtractorType::UNDEFINED;

    const FeatureMatcher::Image image_sift = {
        /*image_id=*/1,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(
            std::vector<FeatureKeypoint>{{1, 0}, {2, 0}}),
        std::make_shared<FeatureDescriptors>(sift_desc)};
    const FeatureMatcher::Image image_undefined = {
        /*image_id=*/2,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(
            std::vector<FeatureKeypoint>{{2, 0}, {1, 0}}),
        std::make_shared<FeatureDescriptors>(undefined_desc)};

    TwoViewGeometry two_view_geometry = CreatePlanarTwoViewGeometry();

    EXPECT_THROW(matcher->MatchGuided(
                     1.0, image_sift, image_undefined, &two_view_geometry),
                 std::invalid_argument);
  });
}

TEST(SiftCPUFeatureMatcherFaissVsBruteForce, Nominal) {
  FeatureMatchingOptions match_options;
  match_options.max_num_matches = 1000;

  auto TestFaissVsBruteForce = [](const FeatureMatchingOptions& options,
                                  const FeatureDescriptors& descriptors1,
                                  const FeatureDescriptors& descriptors2) {
    const Camera camera = Camera::CreateFromModelId(
        1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
    const FeatureMatcher::Image image0 = {
        /*image_id=*/0,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
    const FeatureMatcher::Image image1 = {
        /*image_id=*/1,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(descriptors1)};
    const FeatureMatcher::Image image2 = {
        /*image_id=*/2,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(descriptors2)};

    FeatureDescriptorIndexCacheHelper index_cache_helper(
        {image0, image1, image2});

    FeatureMatches matches_bf;
    FeatureMatches matches_faiss;

    FeatureMatchingOptions custom_options = options;
    custom_options.use_gpu = false;
    custom_options.sift->cpu_brute_force_matcher = true;
    auto bf_matcher = CreateSiftFeatureMatcher(custom_options);
    custom_options.sift->cpu_brute_force_matcher = false;
    custom_options.sift->cpu_descriptor_index_cache =
        &index_cache_helper.index_cache;
    auto faiss_matcher = CreateSiftFeatureMatcher(custom_options);

    bf_matcher->Match(image1, image2, &matches_bf);
    faiss_matcher->Match(image1, image2, &matches_faiss);
    CheckEqualMatches(matches_bf, matches_faiss);

    const size_t num_matches = matches_bf.size();

    bf_matcher->Match(image0, image2, &matches_bf);
    faiss_matcher->Match(image0, image2, &matches_faiss);
    CheckEqualMatches(matches_bf, matches_faiss);

    bf_matcher->Match(image1, image0, &matches_bf);
    faiss_matcher->Match(image1, image0, &matches_faiss);
    CheckEqualMatches(matches_bf, matches_faiss);

    bf_matcher->Match(image0, image0, &matches_bf);
    faiss_matcher->Match(image0, image0, &matches_faiss);
    CheckEqualMatches(matches_bf, matches_faiss);

    return num_matches;
  };

  {
    const FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
    const FeatureDescriptors descriptors2 = CreateRandomFeatureDescriptors(50);
    FeatureMatchingOptions match_options;
    TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
  }

  {
    const FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
    FeatureDescriptors descriptors2;
    descriptors2.data = descriptors1.data.colwise().reverse();
    descriptors2.type = descriptors1.type;
    FeatureMatchingOptions match_options;
    const size_t num_matches =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches, 50);
  }

  // Check the ratio test.
  {
    FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
    FeatureDescriptors descriptors2 = descriptors1;

    FeatureMatchingOptions match_options;
    const size_t num_matches1 =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches1, 50);

    descriptors2.data.row(49) = descriptors2.data.row(0);
    descriptors2.data(0, 0) += 50;
    descriptors2.data.row(0) = FeatureDescriptorsToUnsignedByte(
        descriptors2.data.row(0).cast<float>().normalized());
    descriptors2.data(49, 0) += 100;
    descriptors2.data.row(49) = FeatureDescriptorsToUnsignedByte(
        descriptors2.data.row(49).cast<float>().normalized());

    match_options.sift->max_ratio = 0.4;
    FeatureDescriptors descriptors1_top49;
    descriptors1_top49.data = descriptors1.data.topRows(49);
    descriptors1_top49.type = descriptors1.type;
    const size_t num_matches2 =
        TestFaissVsBruteForce(match_options, descriptors1_top49, descriptors2);
    EXPECT_EQ(num_matches2, 48);

    match_options.sift->max_ratio = 0.6;
    const size_t num_matches3 =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches3, 49);
  }

  // Check the cross check.
  {
    FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
    FeatureDescriptors descriptors2 = descriptors1;
    descriptors1.data.row(0) = descriptors1.data.row(1);

    FeatureMatchingOptions match_options;

    match_options.sift->cross_check = false;
    const size_t num_matches1 =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches1, 50);

    match_options.sift->cross_check = true;
    const size_t num_matches2 =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches2, 48);
  }
}

TEST(MatchGuidedSiftFeaturesCPU, Nominal) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  const FeatureMatcher::Image image0 = {
      /*image_id=*/0,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(0),
      std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{1, 0}, {2, 0}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{2, 0}, {1, 0}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};
  const FeatureMatcher::Image image3 = {
      /*image_id=*/3,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{100, 0}, {2, 0}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};

  FeatureDescriptorIndexCacheHelper index_cache_helper(
      {image0, image1, image2, image3});

  TwoViewGeometry two_view_geometry = CreatePlanarTwoViewGeometry();

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_descriptor_index_cache = &index_cache_helper.index_cache;
  auto matcher = CreateSiftFeatureMatcher(options);

  constexpr double kMaxError = 1.0;

  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);

  matcher->MatchGuided(kMaxError, image3, image2, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 0);

  matcher->MatchGuided(kMaxError, image0, image2, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
  matcher->MatchGuided(kMaxError, image1, image0, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
  matcher->MatchGuided(kMaxError, image0, image0, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
}

void TestGuidedMatchingWithCameraDistortion(
    const std::function<std::unique_ptr<FeatureMatcher>(
        const std::vector<FeatureMatcher::Image>&)>& matcher_factory) {
  // Test guided matching with essential matrix using calibrated cameras.
  // This exercises the code path that uses normalized coordinates.
  // Use the OPENCV model with strong radial and tangential distortion. Its
  // params are fx, fy, cx, cy, k1, k2, p1, p2. The distortion is strong enough
  // that the pixel-coordinate fundamental matrix finds no matches, but must
  // stay invertible over the keypoints used below, which is what bounds how
  // large these coefficients can be; p2 is left at zero for that reason.
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kOpenCV, 100.0, 100, 200);
  camera.params[4] = -0.5;  // k1
  camera.params[5] = 0.5;   // k2
  camera.params[6] = -0.5;  // p1

  // Two points on the epipolar line (v=0 in normalized coordinates).
  const Eigen::Vector2f img_point11 =
      camera.ImgFromCam({-0.5, 0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point12 =
      camera.ImgFromCam({0.4, -0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point21 =
      camera.ImgFromCam({0.3, -0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point22 =
      camera.ImgFromCam({-0.4, 0.1, 1.0}).value().cast<float>();

  const FeatureMatcher::Image image0 = {
      /*image_id=*/0,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(0),
      std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point11.x(), img_point11.y()},
                                       {img_point12.x(), img_point12.y()}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point21.x(), img_point21.y()},
                                       {img_point22.x(), img_point22.y()}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  auto matcher = matcher_factory({image0, image1, image2});

  TwoViewGeometry two_view_geometry;
  two_view_geometry.E = EssentialMatrixFromPose(
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0)));
  two_view_geometry.F =
      FundamentalFromEssentialMatrix(camera.CalibrationMatrix(),
                                     two_view_geometry.E.value(),
                                     camera.CalibrationMatrix());

  constexpr double kMaxError = 1.0;

  // With uncalibrated cameras, the fundamental matrix is used with pixel
  // coordinates and no matches are expected to be found due to strong
  // distortion.
  two_view_geometry.config = TwoViewGeometry::UNCALIBRATED;
  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ASSERT_EQ(two_view_geometry.inlier_matches.size(), 0);

  // With calibrated cameras, the essential matrix is used with normalized
  // coordinates and matches are expected to be found.
  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);

  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  matcher->MatchGuided(kMaxError, image0, image2, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
}

// Guided matching for a spherical camera, with correspondences deliberately in
// the back hemisphere. Those pixels have no normalized image plane
// representation at all - CamFromImg fails for them - so they are only
// matchable via the full-sphere bearing.
void TestGuidedMatchingSpherical(
    const std::function<std::unique_ptr<FeatureMatcher>(
        const std::vector<FeatureMatcher::Image>&)>& matcher_factory) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, /*focal_length=*/0, 512, 256);

  const Rigid3d cam2_from_cam1(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d(1, 0, 0));

  // Both points are behind both cameras, i.e. in the back hemisphere.
  const Eigen::Vector3d point3D1(0.3, 0.1, -2.0);
  const Eigen::Vector3d point3D2(-0.25, -0.15, -2.5);

  auto project = [&camera](const Eigen::Vector3d& point3D) {
    const std::optional<Eigen::Vector2d> image_point =
        camera.ImgFromCam(point3D);
    THROW_CHECK(image_point.has_value());
    // The premise of this test: these pixels are unprojectable through the
    // normalized image plane. If this ever starts failing, the test is no
    // longer exercising the back hemisphere.
    EXPECT_FALSE(camera.CamFromImg(*image_point).has_value());
    return image_point->cast<float>().eval();
  };

  const Eigen::Vector2f img_point11 = project(point3D1);
  const Eigen::Vector2f img_point12 = project(point3D2);
  const Eigen::Vector2f img_point21 = project(cam2_from_cam1 * point3D2);
  const Eigen::Vector2f img_point22 = project(cam2_from_cam1 * point3D1);

  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point11.x(), img_point11.y()},
                                       {img_point12.x(), img_point12.y()}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point21.x(), img_point21.y()},
                                       {img_point22.x(), img_point22.y()}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  // Same as image2, but with the second correspondence replaced by a decoy far
  // off the epipolar great circle, which must be rejected.
  //
  // This is the load-bearing assertion. Recovering the matches above is
  // necessary but not sufficient: when every keypoint is unprojectable, the old
  // sentinel mapped them all to the same location, so the guided filter scored
  // every pair identically and degenerated into accepting everything - and
  // plain descriptor matching then produced the right answer anyway. Only a
  // decoy that the filter must actively reject distinguishes "the epipolar
  // constraint is evaluated correctly for back-hemisphere rays" from "the
  // constraint has stopped constraining anything".
  const Eigen::Vector2f img_point_decoy =
      project(cam2_from_cam1 * Eigen::Vector3d(2.0, -1.5, -0.5));
  const FeatureMatcher::Image image3 = {
      /*image_id=*/3,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(std::vector<FeatureKeypoint>{
          {img_point21.x(), img_point21.y()},
          {img_point_decoy.x(), img_point_decoy.y()}}),
      image2.descriptors};

  auto matcher = matcher_factory({image1, image2, image3});

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  two_view_geometry.E = EssentialMatrixFromPose(cam2_from_cam1);

  constexpr double kMaxError = 4.0;

  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);

  matcher->MatchGuided(kMaxError, image1, image3, &two_view_geometry);
  ASSERT_EQ(two_view_geometry.inlier_matches.size(), 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 0);
}

// One correspondence in the front hemisphere and one in the back, to verify
// the two are handled by the same code path rather than being swapped.
void TestGuidedMatchingSphericalMixedHemispheres(
    const std::function<std::unique_ptr<FeatureMatcher>(
        const std::vector<FeatureMatcher::Image>&)>& matcher_factory) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, /*focal_length=*/0, 512, 256);

  const Rigid3d cam2_from_cam1(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d(1, 0, 0));

  const Eigen::Vector3d point3D_front(0.2, 0.1, 2.0);
  const Eigen::Vector3d point3D_back(-0.25, -0.15, -2.5);

  auto project = [&camera](const Eigen::Vector3d& point3D) {
    const std::optional<Eigen::Vector2d> image_point =
        camera.ImgFromCam(point3D);
    THROW_CHECK(image_point.has_value());
    return image_point->cast<float>().eval();
  };

  const Eigen::Vector2f img_point11 = project(point3D_front);
  const Eigen::Vector2f img_point12 = project(point3D_back);
  const Eigen::Vector2f img_point21 = project(cam2_from_cam1 * point3D_back);
  const Eigen::Vector2f img_point22 = project(cam2_from_cam1 * point3D_front);

  EXPECT_TRUE(camera.CamFromImg(img_point11.cast<double>()).has_value());
  EXPECT_FALSE(camera.CamFromImg(img_point12.cast<double>()).has_value());

  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point11.x(), img_point11.y()},
                                       {img_point12.x(), img_point12.y()}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point21.x(), img_point21.y()},
                                       {img_point22.x(), img_point22.y()}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  // Replaces the back-hemisphere correspondence with a decoy off the epipolar
  // great circle, so that the filter has to actively reject it. See the
  // comment in TestGuidedMatchingSpherical for why this is what discriminates.
  const Eigen::Vector2f img_point_decoy =
      project(cam2_from_cam1 * Eigen::Vector3d(2.0, -1.5, -0.5));
  const FeatureMatcher::Image image3 = {
      /*image_id=*/3,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(std::vector<FeatureKeypoint>{
          {img_point_decoy.x(), img_point_decoy.y()},
          {img_point22.x(), img_point22.y()}}),
      image2.descriptors};

  auto matcher = matcher_factory({image1, image2, image3});

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  two_view_geometry.E = EssentialMatrixFromPose(cam2_from_cam1);

  constexpr double kMaxError = 4.0;

  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);

  // Only the front-hemisphere correspondence survives.
  matcher->MatchGuided(kMaxError, image1, image3, &two_view_geometry);
  ASSERT_EQ(two_view_geometry.inlier_matches.size(), 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
}

// A keypoint the camera cannot unproject must be rejected outright. It used to
// be relocated to a (1e6, 1e6) sentinel, which does not reject: the Sampson
// error is a ratio whose numerator and denominator scale together, so the
// residual converges to the finite squared distance from the partner to the
// epipolar line of the point at infinity in direction (1, 1, 0). Any partner
// near that line was therefore silently accepted.
void TestGuidedMatchingUnprojectableKeypoints(
    const std::function<std::unique_ptr<FeatureMatcher>(
        const std::vector<FeatureMatcher::Image>&)>& matcher_factory) {
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kOpenCV, 100.0, 100, 200);
  camera.params[4] = -0.5;  // k1
  camera.params[5] = 0.5;   // k2
  camera.params[6] = -0.5;  // p1

  // Well inside a region where the iterative undistortion does not converge.
  const Eigen::Vector2d unprojectable(50.0, 150.0);
  ASSERT_FALSE(camera.CamFromImg(unprojectable).has_value());

  auto project = [&camera](const Eigen::Vector3d& point3D) {
    const std::optional<Eigen::Vector2d> image_point =
        camera.ImgFromCam(point3D);
    THROW_CHECK(image_point.has_value());
    // Everything except the sentinel keypoint must be a normal, usable
    // keypoint, or the test would pass for the wrong reason.
    EXPECT_TRUE(camera.CamFromImg(*image_point).has_value());
    return image_point->cast<float>().eval();
  };

  // A translation with tx == ty, so that the epipolar line of the sentinel
  // direction (1, 1, 0) passes through the image center and the decoy below
  // can sit on it at a well-behaved location.
  const Rigid3d cam2_from_cam1(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d(1, 1, 1));
  const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);

  const Eigen::Vector3d point3D(-0.3, -0.2, 2.0);
  const Eigen::Vector2f img_point_good1 = project(point3D);
  const Eigen::Vector2f img_point_good2 = project(cam2_from_cam1 * point3D);

  // The old (1e6, 1e6) sentinel converges to the direction (1, 1, 0); its
  // epipolar line is where spurious matches used to concentrate, so the decoy
  // is placed exactly on it.
  const Eigen::Vector3d sentinel_line = E * Eigen::Vector3d(1, 1, 0);
  const double decoy_x = 0.2;
  const double decoy_y =
      -(sentinel_line.x() * decoy_x + sentinel_line.z()) / sentinel_line.y();
  const Eigen::Vector2f img_point_decoy = project({decoy_x, decoy_y, 1.0});

  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(std::vector<FeatureKeypoint>{
          {static_cast<float>(unprojectable.x()),
           static_cast<float>(unprojectable.y())},
          {img_point_good1.x(), img_point_good1.y()}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(std::vector<FeatureKeypoint>{
          {img_point_good2.x(), img_point_good2.y()},
          {img_point_decoy.x(), img_point_decoy.y()}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  auto matcher = matcher_factory({image1, image2});

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  two_view_geometry.E = E;

  matcher->MatchGuided(/*max_error=*/1.0, image1, image2, &two_view_geometry);

  // Only the good pair survives; the unprojectable keypoint 0 matches nothing.
  ASSERT_EQ(two_view_geometry.inlier_matches.size(), 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 0);
}

TEST(MatchGuidedSiftFeaturesCPU, EssentialMatrix) {
  std::unique_ptr<FeatureDescriptorIndexCacheHelper> index_cache_helper;
  TestGuidedMatchingWithCameraDistortion(
      [&index_cache_helper](const std::vector<FeatureMatcher::Image>& images) {
        index_cache_helper =
            std::make_unique<FeatureDescriptorIndexCacheHelper>(images);
        FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
        options.use_gpu = false;
        options.sift->cpu_descriptor_index_cache =
            &index_cache_helper->index_cache;
        return CreateSiftFeatureMatcher(options);
      });
}

TEST(MatchGuidedSiftFeaturesCPU, Spherical) {
  std::unique_ptr<FeatureDescriptorIndexCacheHelper> index_cache_helper;
  TestGuidedMatchingSpherical(
      [&index_cache_helper](const std::vector<FeatureMatcher::Image>& images) {
        index_cache_helper =
            std::make_unique<FeatureDescriptorIndexCacheHelper>(images);
        FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
        options.use_gpu = false;
        options.sift->cpu_descriptor_index_cache =
            &index_cache_helper->index_cache;
        return CreateSiftFeatureMatcher(options);
      });
}

TEST(MatchGuidedSiftFeaturesCPU, SphericalMixedHemispheres) {
  std::unique_ptr<FeatureDescriptorIndexCacheHelper> index_cache_helper;
  TestGuidedMatchingSphericalMixedHemispheres(
      [&index_cache_helper](const std::vector<FeatureMatcher::Image>& images) {
        index_cache_helper =
            std::make_unique<FeatureDescriptorIndexCacheHelper>(images);
        FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
        options.use_gpu = false;
        options.sift->cpu_descriptor_index_cache =
            &index_cache_helper->index_cache;
        return CreateSiftFeatureMatcher(options);
      });
}

TEST(MatchGuidedSiftFeaturesCPU, UnprojectableKeypoints) {
  std::unique_ptr<FeatureDescriptorIndexCacheHelper> index_cache_helper;
  TestGuidedMatchingUnprojectableKeypoints(
      [&index_cache_helper](const std::vector<FeatureMatcher::Image>& images) {
        index_cache_helper =
            std::make_unique<FeatureDescriptorIndexCacheHelper>(images);
        FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
        options.use_gpu = false;
        options.sift->cpu_descriptor_index_cache =
            &index_cache_helper->index_cache;
        return CreateSiftFeatureMatcher(options);
      });
}

void TestGuidedMatchingSharedFocal(
    const std::function<std::unique_ptr<FeatureMatcher>(
        const std::vector<FeatureMatcher::Image>&)>& matcher_factory) {
  // An UNCALIBRATED pair carrying solver-estimated intrinsics (camera1/camera2)
  // is guided-matched via the essential matrix in normalized coordinates, using
  // those estimated intrinsics rather than the images' cameras, whose focal
  // length is only a placeholder. Distortion is strong enough that the
  // pixel-coordinate F path finds nothing, and the placeholder focal is wrong
  // enough that normalizing with it finds nothing either, so the test passes
  // only if the estimated camera is the one used.
  //
  // As elsewhere on the E path, E is taken to relate undistorted rays.
  constexpr double kEstimatedFocal = 100.0;
  constexpr double kPlaceholderFocal = 500.0;
  // OPENCV params are fx, fy, cx, cy, k1, k2, p1, p2. Same distortion as in
  // TestGuidedMatchingWithCameraDistortion: strong, but invertible over the
  // keypoints used below.
  Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kOpenCV, kEstimatedFocal, 100, 200);
  camera.params[4] = -0.5;  // k1
  camera.params[5] = 0.5;   // k2
  camera.params[6] = -0.5;  // p1

  // The camera as stored in the database: same model and distortion, but the
  // focal length has not been recovered yet.
  Camera placeholder_camera = camera;
  placeholder_camera.SetFocalLength(kPlaceholderFocal);

  // Two points on the epipolar line (v=0 in normalized coordinates).
  const Eigen::Vector2f img_point11 =
      camera.ImgFromCam({-0.5, 0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point12 =
      camera.ImgFromCam({0.4, -0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point21 =
      camera.ImgFromCam({0.3, -0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point22 =
      camera.ImgFromCam({-0.4, 0.1, 1.0}).value().cast<float>();

  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&placeholder_camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point11.x(), img_point11.y()},
                                       {img_point12.x(), img_point12.y()}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&placeholder_camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point21.x(), img_point21.y()},
                                       {img_point22.x(), img_point22.y()}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  auto matcher = matcher_factory({image1, image2});

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::UNCALIBRATED;
  two_view_geometry.E = EssentialMatrixFromPose(
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0)));
  two_view_geometry.camera1 = camera;
  two_view_geometry.camera2 = camera;
  // F = K^-T E K^-1, as the estimator populates it for this config.
  two_view_geometry.F =
      FundamentalFromEssentialMatrix(camera.CalibrationMatrix(),
                                     two_view_geometry.E.value(),
                                     camera.CalibrationMatrix());

  constexpr double kMaxError = 1.0;

  // Matches are found only by normalizing with the estimated intrinsics. The
  // complementary case, an UNCALIBRATED pair without them falling back to the
  // F path, is covered by TestGuidedMatchingWithCameraDistortion.
  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);
}

// The normalizing camera is not a function of the image alone: a shared-focal
// pair carries a focal length estimated per pair, so the same image matched
// against different partners must be renormalized. Guards the GPU matcher's
// feature-location cache, which keys on the image id.
void TestGuidedMatchingSharedFocalPerPairFocal(
    const std::function<std::unique_ptr<FeatureMatcher>(
        const std::vector<FeatureMatcher::Image>&)>& matcher_factory) {
  constexpr double kFocalA = 100.0;
  constexpr double kFocalB = 200.0;
  const Camera camera_a = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, kFocalA, 100, 200);
  const Camera camera_b = Camera::CreateFromModelId(
      2, CameraModelId::kSimplePinhole, kFocalB, 100, 200);

  // image1's pixels normalize to y = +-0.1 under camera_a, and to half that,
  // y = +-0.05, under camera_b. The relative pose is a pure x-translation, so a
  // match requires the partner's normalized y to agree.
  const Eigen::Vector2f img_point11 =
      camera_a.ImgFromCam({-0.5, 0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point12 =
      camera_a.ImgFromCam({0.4, -0.1, 1.0}).value().cast<float>();
  // Partner for the camera_a pair.
  const Eigen::Vector2f img_point21 =
      camera_a.ImgFromCam({0.3, -0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point22 =
      camera_a.ImgFromCam({-0.4, 0.1, 1.0}).value().cast<float>();
  // Partner for the camera_b pair.
  const Eigen::Vector2f img_point31 =
      camera_b.ImgFromCam({0.3, -0.05, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point32 =
      camera_b.ImgFromCam({-0.4, 0.05, 1.0}).value().cast<float>();

  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera_a,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point11.x(), img_point11.y()},
                                       {img_point12.x(), img_point12.y()}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera_a,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point21.x(), img_point21.y()},
                                       {img_point22.x(), img_point22.y()}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};
  const FeatureMatcher::Image image3 = {
      /*image_id=*/3,
      /*camera=*/&camera_a,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point31.x(), img_point31.y()},
                                       {img_point32.x(), img_point32.y()}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  auto matcher = matcher_factory({image1, image2, image3});

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::UNCALIBRATED;
  two_view_geometry.E = EssentialMatrixFromPose(
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0)));

  constexpr double kMaxError = 1.0;

  two_view_geometry.camera1 = camera_a;
  two_view_geometry.camera2 = camera_a;
  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);

  // Same image1, different estimated focal: stale normalized keypoints from the
  // previous call would put image1 at y = +-0.1 instead of +-0.05, far outside
  // the ~1/f normalized threshold.
  two_view_geometry.camera1 = camera_b;
  two_view_geometry.camera2 = camera_b;
  matcher->MatchGuided(kMaxError, image1, image3, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);
}

TEST(MatchGuidedSiftFeaturesCPU, SharedFocal) {
  std::unique_ptr<FeatureDescriptorIndexCacheHelper> index_cache_helper;
  TestGuidedMatchingSharedFocal(
      [&index_cache_helper](const std::vector<FeatureMatcher::Image>& images) {
        index_cache_helper =
            std::make_unique<FeatureDescriptorIndexCacheHelper>(images);
        FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
        options.use_gpu = false;
        options.sift->cpu_descriptor_index_cache =
            &index_cache_helper->index_cache;
        return CreateSiftFeatureMatcher(options);
      });
}

TEST(MatchGuidedSiftFeaturesCPU, SharedFocalPerPairFocal) {
  std::unique_ptr<FeatureDescriptorIndexCacheHelper> index_cache_helper;
  TestGuidedMatchingSharedFocalPerPairFocal(
      [&index_cache_helper](const std::vector<FeatureMatcher::Image>& images) {
        index_cache_helper =
            std::make_unique<FeatureDescriptorIndexCacheHelper>(images);
        FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
        options.use_gpu = false;
        options.sift->cpu_descriptor_index_cache =
            &index_cache_helper->index_cache;
        return CreateSiftFeatureMatcher(options);
      });
}

TEST(MatchSiftFeaturesGPU, Nominal) {
  RunGpuTest([] {
    FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
    options.use_gpu = true;
    options.max_num_matches = 1000;
    auto matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));

    const Camera camera = Camera::CreateFromModelId(
        1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
    const FeatureMatcher::Image image0 = {
        /*image_id=*/0,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
    const FeatureMatcher::Image image1 = {
        /*image_id=*/1,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(
            CreateRandomFeatureDescriptors(2))};
    const FeatureMatcher::Image image2 = {
        /*image_id=*/2,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(
            CreateReversedDescriptors(*image1.descriptors))};

    FeatureMatches matches;

    matcher->Match(image1, image2, &matches);
    ExpectReversedMatches(matches);

    matcher->Match(image1, image2, &matches);
    ExpectReversedMatches(matches);

    matcher->Match(image0, image2, &matches);
    EXPECT_EQ(matches.size(), 0);
    matcher->Match(image1, image0, &matches);
    EXPECT_EQ(matches.size(), 0);
    matcher->Match(image0, image0, &matches);
    EXPECT_EQ(matches.size(), 0);
  });
}

TEST(MatchSiftFeaturesCPUvsGPU, Nominal) {
  RunGpuTest([] {
    auto TestCPUvsGPU = [](const FeatureMatchingOptions& options,
                           const FeatureDescriptors& descriptors1,
                           const FeatureDescriptors& descriptors2) {
      const Camera camera = Camera::CreateFromModelId(
          1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
      const FeatureMatcher::Image image0 = {
          /*image_id=*/0,
          /*camera=*/&camera,
          /*keypoints=*/nullptr,
          std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
      const FeatureMatcher::Image image1 = {
          /*image_id=*/1,
          /*camera=*/&camera,
          /*keypoints=*/nullptr,
          std::make_shared<FeatureDescriptors>(descriptors1)};
      const FeatureMatcher::Image image2 = {
          /*image_id=*/2,
          /*camera=*/&camera,
          /*keypoints=*/nullptr,
          std::make_shared<FeatureDescriptors>(descriptors2)};

      FeatureDescriptorIndexCacheHelper index_cache_helper(
          {image0, image1, image2});

      FeatureMatchingOptions custom_options = options;
      custom_options.use_gpu = true;
      custom_options.max_num_matches = 1000;
      auto gpu_matcher =
          THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(custom_options));
      custom_options.use_gpu = false;
      custom_options.sift->cpu_descriptor_index_cache =
          &index_cache_helper.index_cache;
      auto cpu_matcher = CreateSiftFeatureMatcher(custom_options);

      FeatureMatches matches_cpu;
      FeatureMatches matches_gpu;

      cpu_matcher->Match(image1, image2, &matches_cpu);
      gpu_matcher->Match(image1, image2, &matches_gpu);
      CheckEqualMatches(matches_cpu, matches_gpu);

      const size_t num_matches = matches_cpu.size();

      cpu_matcher->Match(image0, image2, &matches_cpu);
      gpu_matcher->Match(image0, image2, &matches_gpu);
      CheckEqualMatches(matches_cpu, matches_gpu);

      cpu_matcher->Match(image1, image0, &matches_cpu);
      gpu_matcher->Match(image1, image0, &matches_gpu);
      CheckEqualMatches(matches_cpu, matches_gpu);

      cpu_matcher->Match(image0, image0, &matches_cpu);
      gpu_matcher->Match(image0, image0, &matches_gpu);
      CheckEqualMatches(matches_cpu, matches_gpu);

      return num_matches;
    };

    {
      const FeatureDescriptors descriptors1 =
          CreateRandomFeatureDescriptors(50);
      const FeatureDescriptors descriptors2 =
          CreateRandomFeatureDescriptors(50);
      FeatureMatchingOptions match_options;
      TestCPUvsGPU(match_options, descriptors1, descriptors2);
    }

    {
      const FeatureDescriptors descriptors1 =
          CreateRandomFeatureDescriptors(50);
      FeatureDescriptors descriptors2;
      descriptors2.data = descriptors1.data.colwise().reverse();
      descriptors2.type = descriptors1.type;
      FeatureMatchingOptions match_options;
      const size_t num_matches =
          TestCPUvsGPU(match_options, descriptors1, descriptors2);
      EXPECT_EQ(num_matches, 50);
    }

    // Check the ratio test.
    {
      FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
      FeatureDescriptors descriptors2 = descriptors1;

      FeatureMatchingOptions match_options;
      const size_t num_matches1 =
          TestCPUvsGPU(match_options, descriptors1, descriptors2);
      EXPECT_EQ(num_matches1, 50);

      descriptors2.data.row(49) = descriptors2.data.row(0);
      descriptors2.data(0, 0) += 50;
      descriptors2.data.row(0) = FeatureDescriptorsToUnsignedByte(
          descriptors2.data.row(0).cast<float>().normalized());
      descriptors2.data(49, 0) += 100;
      descriptors2.data.row(49) = FeatureDescriptorsToUnsignedByte(
          descriptors2.data.row(49).cast<float>().normalized());

      match_options.sift->max_ratio = 0.4;
      FeatureDescriptors descriptors1_top49;
      descriptors1_top49.data = descriptors1.data.topRows(49);
      descriptors1_top49.type = descriptors1.type;
      const size_t num_matches2 =
          TestCPUvsGPU(match_options, descriptors1_top49, descriptors2);
      EXPECT_EQ(num_matches2, 48);

      match_options.sift->max_ratio = 0.6;
      const size_t num_matches3 =
          TestCPUvsGPU(match_options, descriptors1, descriptors2);
      EXPECT_EQ(num_matches3, 49);
    }

    // Check the cross check.
    {
      FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
      FeatureDescriptors descriptors2 = descriptors1;
      descriptors1.data.row(0) = descriptors1.data.row(1);

      FeatureMatchingOptions match_options;

      match_options.sift->cross_check = false;
      const size_t num_matches1 =
          TestCPUvsGPU(match_options, descriptors1, descriptors2);
      EXPECT_EQ(num_matches1, 50);

      match_options.sift->cross_check = true;
      const size_t num_matches2 =
          TestCPUvsGPU(match_options, descriptors1, descriptors2);
      EXPECT_EQ(num_matches2, 48);
    }
  });
}

TEST(MatchGuidedSiftFeaturesGPU, Nominal) {
  RunGpuTest([] {
    Camera camera = Camera::CreateFromModelId(
        1, CameraModelId::kSimpleRadial, 100.0, 100, 200);
    const FeatureMatcher::Image image0 = {
        /*image_id=*/0,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(0),
        std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
    const FeatureMatcher::Image image1 = {
        /*image_id=*/1,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(
            std::vector<FeatureKeypoint>{{1, 0}, {2, 0}}),
        std::make_shared<FeatureDescriptors>(
            CreateRandomFeatureDescriptors(2))};
    const FeatureMatcher::Image image2 = {
        /*image_id=*/2,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(
            std::vector<FeatureKeypoint>{{2, 0}, {1, 0}}),
        std::make_shared<FeatureDescriptors>(
            CreateReversedDescriptors(*image1.descriptors))};
    const FeatureMatcher::Image image3 = {
        /*image_id=*/3,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(
            std::vector<FeatureKeypoint>{{100, 0}, {1, 0}}),
        std::make_shared<FeatureDescriptors>(
            CreateRandomFeatureDescriptors(2))};

    FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
    options.use_gpu = true;
    options.max_num_matches = 1000;
    auto matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));

    TwoViewGeometry two_view_geometry = CreatePlanarTwoViewGeometry();

    constexpr double kMaxError = 4.0;

    matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
    ExpectReversedInlierMatches(two_view_geometry);

    matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
    ExpectReversedInlierMatches(two_view_geometry);

    matcher->MatchGuided(kMaxError, image3, image2, &two_view_geometry);
    EXPECT_EQ(two_view_geometry.inlier_matches.size(), 1);
    EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 1);
    EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 0);

    matcher->MatchGuided(kMaxError, image0, image2, &two_view_geometry);
    EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
    matcher->MatchGuided(kMaxError, image1, image0, &two_view_geometry);
    EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
    matcher->MatchGuided(kMaxError, image0, image0, &two_view_geometry);
    EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
  });
}

TEST(MatchGuidedSiftFeaturesGPU, EssentialMatrix) {
  RunGpuTest([] {
    TestGuidedMatchingWithCameraDistortion(
        [](const std::vector<FeatureMatcher::Image>& images) {
          FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
          options.use_gpu = true;
          options.max_num_matches = 1000;
          return THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));
        });
  });
}

TEST(MatchGuidedSiftFeaturesGPU, Spherical) {
  RunGpuTest([] {
    TestGuidedMatchingSpherical(
        [](const std::vector<FeatureMatcher::Image>& images) {
          FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
          options.use_gpu = true;
          options.max_num_matches = 1000;
          return THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));
        });
  });
}

TEST(MatchGuidedSiftFeaturesGPU, SphericalMixedHemispheres) {
  RunGpuTest([] {
    TestGuidedMatchingSphericalMixedHemispheres(
        [](const std::vector<FeatureMatcher::Image>& images) {
          FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
          options.use_gpu = true;
          options.max_num_matches = 1000;
          return THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));
        });
  });
}

TEST(MatchGuidedSiftFeaturesGPU, UnprojectableKeypoints) {
  RunGpuTest([] {
    TestGuidedMatchingUnprojectableKeypoints(
        [](const std::vector<FeatureMatcher::Image>& images) {
          FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
          options.use_gpu = true;
          options.max_num_matches = 1000;
          return THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));
        });
  });
}

// The GPU tangent Sampson kernel must reproduce the CPU reference
// implementation on the same input. This is the check that distinguishes a
// genuine kernel bug from a plumbing bug, since the CPU path is independently
// tested above.
TEST(MatchGuidedSiftFeaturesCPUvsGPUGuided, EssentialMatrix) {
  const size_t kNumFeatures = 200;
  std::vector<Camera> cameras;
  cameras.push_back(Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 650.0, 1024, 768));
  cameras.push_back(Camera::CreateFromModelId(
      2, CameraModelId::kOpenCVFisheye, 350.0, 1024, 768));
  cameras.push_back(Camera::CreateFromModelId(
      3, CameraModelId::kEquirectangular, 0.0, 1000, 500));

  for (const Camera& camera : cameras) {
    SetPRNGSeed(42);
    FeatureKeypoints keypoints1(kNumFeatures);
    FeatureKeypoints keypoints2(kNumFeatures);
    for (size_t i = 0; i < kNumFeatures; ++i) {
      keypoints1[i] =
          FeatureKeypoint(RandomUniformReal<float>(1.0f, camera.width - 1.0f),
                          RandomUniformReal<float>(1.0f, camera.height - 1.0f));
      keypoints2[i] =
          FeatureKeypoint(RandomUniformReal<float>(1.0f, camera.width - 1.0f),
                          RandomUniformReal<float>(1.0f, camera.height - 1.0f));
    }

    const FeatureMatcher::Image image1 = {
        /*image_id=*/1,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(keypoints1),
        std::make_shared<FeatureDescriptors>(
            CreateRandomFeatureDescriptors(kNumFeatures))};
    const FeatureMatcher::Image image2 = {
        /*image_id=*/2,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(keypoints2),
        std::make_shared<FeatureDescriptors>(
            CreateRandomFeatureDescriptors(kNumFeatures))};

    TwoViewGeometry geometry;
    geometry.config = TwoViewGeometry::CALIBRATED;
    geometry.E = EssentialMatrixFromPose(
        Rigid3d(Eigen::Quaterniond(Eigen::AngleAxisd(
                    0.2, Eigen::Vector3d(0.3, 1.0, 0.2).normalized())),
                Eigen::Vector3d(1.0, 0.15, 0.05).normalized()));

    // A loose threshold so a non-trivial number of pairs survive the filter.
    constexpr double kMaxError = 30.0;

    FeatureDescriptorIndexCacheHelper index_cache_helper({image1, image2});
    FeatureMatchingOptions cpu_options(FeatureMatcherType::SIFT_BRUTEFORCE);
    cpu_options.use_gpu = false;
    cpu_options.sift->cpu_descriptor_index_cache =
        &index_cache_helper.index_cache;
    TwoViewGeometry cpu_geometry = geometry;
    CreateSiftFeatureMatcher(cpu_options)
        ->MatchGuided(kMaxError, image1, image2, &cpu_geometry);

    EXPECT_GT(cpu_geometry.inlier_matches.size(), 0u)
        << "model " << camera.ModelName();

    // Note that the assertions must live inside the lambda, since the body is
    // not executed if the GPU/OpenGL context is unavailable.
    RunGpuTest([&] {
      TwoViewGeometry gpu_geometry = geometry;
      FeatureMatchingOptions gpu_options(FeatureMatcherType::SIFT_BRUTEFORCE);
      gpu_options.use_gpu = true;
      gpu_options.max_num_matches = 4 * kNumFeatures;
      THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(gpu_options))
          ->MatchGuided(kMaxError, image1, image2, &gpu_geometry);

      // The CPU scores in double and the GPU in float, so a pair whose tangent
      // Sampson residual sits within float epsilon of kMaxError can flip
      // inclusion. Compare as sets and tolerate a few boundary flips rather
      // than requiring identical size and order.
      const auto to_set = [](const FeatureMatches& matches) {
        std::set<std::pair<point2D_t, point2D_t>> set;
        for (const auto& match : matches) {
          set.emplace(match.point2D_idx1, match.point2D_idx2);
        }
        return set;
      };
      const std::set<std::pair<point2D_t, point2D_t>> cpu_set =
          to_set(cpu_geometry.inlier_matches);
      const std::set<std::pair<point2D_t, point2D_t>> gpu_set =
          to_set(gpu_geometry.inlier_matches);
      size_t num_disagree = 0;
      for (const auto& match : cpu_set) num_disagree += !gpu_set.count(match);
      for (const auto& match : gpu_set) num_disagree += !cpu_set.count(match);
      EXPECT_LE(num_disagree, 4u) << "model " << camera.ModelName();
    });
  }
}

TEST(MatchGuidedSiftFeaturesGPU, SharedFocal) {
  RunGpuTest([] {
    TestGuidedMatchingSharedFocal(
        [](const std::vector<FeatureMatcher::Image>& images) {
          FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
          options.use_gpu = true;
          options.max_num_matches = 1000;
          return THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));
        });
  });
}

TEST(MatchGuidedSiftFeaturesGPU, SharedFocalPerPairFocal) {
  RunGpuTest([] {
    TestGuidedMatchingSharedFocalPerPairFocal(
        [](const std::vector<FeatureMatcher::Image>& images) {
          FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
          options.use_gpu = true;
          options.max_num_matches = 1000;
          return THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));
        });
  });
}

}  // namespace
}  // namespace colmap

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
  // Use kRadial model with strong radial and tangential distortion.
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kOpenCV, 100.0, 100, 200);
  camera.params[3] = 0.5;   // k1
  camera.params[4] = -0.5;  // k2
  camera.params[5] = 0.5;   // p1
  camera.params[6] = -0.5;  // p2

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

void TestGuidedMatchingSharedFocal(
    const std::function<std::unique_ptr<FeatureMatcher>(
        const std::vector<FeatureMatcher::Image>&)>& matcher_factory) {
  // Regression guard: an UNCALIBRATED pair that carries solver-estimated
  // intrinsics (camera1/camera2) and an essential matrix must still be
  // guided-matched via its fundamental matrix, like any UNCALIBRATED pair,
  // rather than diverted to the E path or dropped. A pinhole camera keeps the
  // pixel-coordinate F exact so matches are actually found.
  constexpr double kFocal = 100.0;
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, kFocal, 100, 200);

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

  // Shared-focal pairs are guided-matched (not dropped) via the F path.
  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);

  // Non-corresponding images still find nothing.
  two_view_geometry.config = TwoViewGeometry::UNCALIBRATED;
  matcher->MatchGuided(kMaxError, image0, image2, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
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

// Guided matching of a spherical camera must work over the whole sphere. The
// normalized image plane representation used previously is undefined for any
// bearing with rz <= 0, i.e. for half of an equirectangular image, so every
// correspondence placed there was silently discarded.
void TestGuidedMatchingSpherical(
    const std::function<std::unique_ptr<FeatureMatcher>(
        const std::vector<FeatureMatcher::Image>&)>& matcher_factory) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, /*focal_length=*/0.0, 1000, 500);

  // Bearings deliberately in the BACK hemisphere (rz < 0), which is where
  // CamFromImg fails. E comes from a pure x-translation, whose epipolar
  // constraint on bearings is ry1 / rz1 == ry2 / rz2, so pairs sharing an
  // elevation are exact correspondences.
  // image1[0] corresponds to image2[1] and image1[1] to image2[0], matching the
  // reversed descriptors below. Corresponding rays share ry / rz and differ
  // only in rx, i.e. they lie on a common epipolar plane.
  const Eigen::Vector3d ray11(-0.30, 0.20, -0.93);
  const Eigen::Vector3d ray12(0.40, -0.15, -0.90);
  const Eigen::Vector3d ray21(0.55, -0.15, -0.90);
  const Eigen::Vector3d ray22(-0.10, 0.20, -0.93);

  auto to_keypoint = [&camera](const Eigen::Vector3d& ray) {
    const Eigen::Vector2d xy = camera.ImgFromCam(ray).value();
    return FeatureKeypoint(static_cast<float>(xy.x()),
                           static_cast<float>(xy.y()));
  };

  // Confirm the premise: these pixels have no normalized image plane
  // representation at all, so the previous implementation could not match them.
  for (const Eigen::Vector3d& ray : {ray11, ray12, ray21, ray22}) {
    const Eigen::Vector2d xy = camera.ImgFromCam(ray).value();
    ASSERT_FALSE(camera.CamFromImg(xy).has_value());
    ASSERT_TRUE(camera.CamRayFromImgWithJac(xy).has_value());
  }

  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{to_keypoint(ray11), to_keypoint(ray12)}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{to_keypoint(ray21), to_keypoint(ray22)}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  const FeatureMatcher::Image image2_off = {
      /*image_id=*/3,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(std::vector<FeatureKeypoint>{
          to_keypoint(Eigen::Vector3d(0.55, 0.60, -0.58)),
          to_keypoint(Eigen::Vector3d(-0.10, -0.65, -0.75))}),
      image2.descriptors};

  auto matcher = matcher_factory({image1, image2, image2_off});

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  two_view_geometry.E = EssentialMatrixFromPose(
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0)));

  // A generous threshold: the point of the test is that back-hemisphere
  // features are matched at all, not the precise tolerance.
  matcher->MatchGuided(20.0, image1, image2, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);

  // The geometry must actually constrain the result. Moving image 2 well off
  // the epipolar planes has to reject both pairs.
  //
  // This half of the test is what makes the first half meaningful. The previous
  // implementation mapped every back-hemisphere keypoint to the same sentinel
  // coordinate, which makes the epipolar residual identically zero for *all*
  // pairs - so it would accept these mismatches, and would have "passed" the
  // check above for the wrong reason.
  matcher->MatchGuided(20.0, image1, image2_off, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
}

// Keypoints that cannot be unprojected must be excluded from guided matching
// outright.
//
// They used to be relocated to a far-away sentinel coordinate on the assumption
// that this makes the epipolar residual large. It does not: the Sampson error
// is a ratio whose numerator and denominator scale together, so as a point
// recedes along a direction d the residual converges to the finite distance
// between its partner and the epipolar line of d. Partners near that line were
// therefore accepted, turning every unprojectable keypoint into a source of
// spurious matches concentrated on one line.
void TestGuidedMatchingUnprojectableKeypoints(
    const std::function<std::unique_ptr<FeatureMatcher>(
        const std::vector<FeatureMatcher::Image>&)>& matcher_factory) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, /*focal_length=*/0.0, 1000, 500);

  // Back-hemisphere bearings: unprojectable to the normalized image plane.
  const Eigen::Vector2d unprojectable1 =
      camera.ImgFromCam(Eigen::Vector3d(0.1, 0.1, -0.99)).value();
  const Eigen::Vector2d unprojectable2 =
      camera.ImgFromCam(Eigen::Vector3d(-0.1, -0.35, -0.93)).value();
  ASSERT_FALSE(camera.CamFromImg(unprojectable1).has_value());
  ASSERT_FALSE(camera.CamFromImg(unprojectable2).has_value());

  // The sentinel the old implementation used, in normalized coordinates.
  constexpr double kSentinel = 1e6;
  const Eigen::Vector3d sentinel_point(kSentinel, kSentinel, 1.0);
  const Eigen::Matrix3d E = EssentialMatrixFromPose(
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0)));

  // Demonstrate the failure mode directly: two keypoints both pushed to the
  // sentinel have an *exactly zero* epipolar residual, i.e. the sentinel admits
  // rather than rejects.
  EXPECT_LT(ComputeSquaredSampsonError(sentinel_point, sentinel_point, E),
            1e-12);

  // End to end, such keypoints must now yield no matches at all rather than
  // matching each other.
  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(std::vector<FeatureKeypoint>{
          FeatureKeypoint(static_cast<float>(unprojectable1.x()),
                          static_cast<float>(unprojectable1.y()))}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(1))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(std::vector<FeatureKeypoint>{
          FeatureKeypoint(static_cast<float>(unprojectable2.x()),
                          static_cast<float>(unprojectable2.y()))}),
      std::make_shared<FeatureDescriptors>(*image1.descriptors)};

  auto matcher = matcher_factory({image1, image2});

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  two_view_geometry.E = E;

  // Identical descriptors, so only the geometry can reject this pair. It lies
  // far off the epipolar plane, so it must be rejected - whereas both keypoints
  // would previously have been collapsed onto the sentinel and matched.
  matcher->MatchGuided(1.0, image1, image2, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
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

      EXPECT_EQ(cpu_geometry.inlier_matches.size(),
                gpu_geometry.inlier_matches.size())
          << "model " << camera.ModelName();
      for (size_t i = 0; i < std::min(cpu_geometry.inlier_matches.size(),
                                      gpu_geometry.inlier_matches.size());
           ++i) {
        EXPECT_EQ(cpu_geometry.inlier_matches[i].point2D_idx1,
                  gpu_geometry.inlier_matches[i].point2D_idx1)
            << "model " << camera.ModelName() << " match " << i;
        EXPECT_EQ(cpu_geometry.inlier_matches[i].point2D_idx2,
                  gpu_geometry.inlier_matches[i].point2D_idx2)
            << "model " << camera.ModelName() << " match " << i;
      }
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

}  // namespace
}  // namespace colmap

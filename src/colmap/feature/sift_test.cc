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
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/util/opengl_utils.h"

#include "thirdparty/SiftGPU/SiftGPU.h"

namespace colmap {
namespace {

void CreateImageWithSquare(const int size, Bitmap* bitmap) {
  bitmap->Allocate(size, size, false);
  bitmap->Fill(BitmapColor<uint8_t>(0, 0, 0));
  for (int r = size / 2 - size / 8; r < size / 2 + size / 8; ++r) {
    for (int c = size / 2 - size / 8; c < size / 2 + size / 8; ++c) {
      bitmap->SetPixel(r, c, BitmapColor<uint8_t>(255));
    }
  }
}

TEST(ExtractSiftFeaturesCPU, Nominal) {
  Bitmap bitmap;
  CreateImageWithSquare(256, &bitmap);

  SiftExtractionOptions options;
  options.use_gpu = false;
  options.estimate_affine_shape = false;
  options.domain_size_pooling = false;
  options.force_covariant_extractor = false;
  auto extractor = CreateSiftFeatureExtractor(options);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

  EXPECT_EQ(keypoints.size(), 22);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    EXPECT_GE(keypoints[i].x, 0);
    EXPECT_GE(keypoints[i].y, 0);
    EXPECT_LE(keypoints[i].x, bitmap.Width());
    EXPECT_LE(keypoints[i].y, bitmap.Height());
    EXPECT_GT(keypoints[i].ComputeScale(), 0);
    EXPECT_GT(keypoints[i].ComputeOrientation(), -M_PI);
    EXPECT_LT(keypoints[i].ComputeOrientation(), M_PI);
  }

  EXPECT_EQ(descriptors.rows(), 22);
  for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.row(i).cast<float>().norm() - 512), 1);
  }
}

TEST(ExtractCovariantSiftFeaturesCPU, Nominal) {
  Bitmap bitmap;
  CreateImageWithSquare(256, &bitmap);

  SiftExtractionOptions options;
  options.use_gpu = false;
  options.estimate_affine_shape = false;
  options.domain_size_pooling = false;
  options.force_covariant_extractor = true;
  auto extractor = CreateSiftFeatureExtractor(options);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

  EXPECT_EQ(keypoints.size(), 22);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    EXPECT_GE(keypoints[i].x, 0);
    EXPECT_GE(keypoints[i].y, 0);
    EXPECT_LE(keypoints[i].x, bitmap.Width());
    EXPECT_LE(keypoints[i].y, bitmap.Height());
    EXPECT_GT(keypoints[i].ComputeScale(), 0);
    EXPECT_GT(keypoints[i].ComputeOrientation(), -M_PI);
    EXPECT_LT(keypoints[i].ComputeOrientation(), M_PI);
  }

  EXPECT_EQ(descriptors.rows(), 22);
  for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.row(i).cast<float>().norm() - 512), 1);
  }
}

TEST(ExtractCovariantAffineSiftFeaturesCPU, Nominal) {
  Bitmap bitmap;
  CreateImageWithSquare(256, &bitmap);

  SiftExtractionOptions options;
  options.use_gpu = false;
  options.estimate_affine_shape = true;
  options.domain_size_pooling = false;
  options.force_covariant_extractor = false;
  auto extractor = CreateSiftFeatureExtractor(options);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

  EXPECT_EQ(keypoints.size(), 22);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    EXPECT_GE(keypoints[i].x, 0);
    EXPECT_GE(keypoints[i].y, 0);
    EXPECT_LE(keypoints[i].x, bitmap.Width());
    EXPECT_LE(keypoints[i].y, bitmap.Height());
    EXPECT_GT(keypoints[i].ComputeScale(), 0);
    EXPECT_GT(keypoints[i].ComputeOrientation(), -M_PI);
    EXPECT_LT(keypoints[i].ComputeOrientation(), M_PI);
  }

  EXPECT_EQ(descriptors.rows(), 22);
  for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.row(i).cast<float>().norm() - 512), 1);
  }
}

TEST(ExtractCovariantAffineSiftFeaturesCPU, Upright) {
  Bitmap bitmap;
  CreateImageWithSquare(256, &bitmap);

  SiftExtractionOptions options;
  options.use_gpu = false;
  options.estimate_affine_shape = true;
  options.upright = true;
  options.domain_size_pooling = false;
  options.force_covariant_extractor = false;
  auto extractor = CreateSiftFeatureExtractor(options);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

  EXPECT_EQ(keypoints.size(), 10);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    EXPECT_GE(keypoints[i].x, 0);
    EXPECT_GE(keypoints[i].y, 0);
    EXPECT_LE(keypoints[i].x, bitmap.Width());
    EXPECT_LE(keypoints[i].y, bitmap.Height());
    EXPECT_GT(keypoints[i].ComputeScale(), 0);
    EXPECT_GT(keypoints[i].ComputeOrientation(), -M_PI);
    EXPECT_LT(keypoints[i].ComputeOrientation(), M_PI);
  }

  EXPECT_EQ(descriptors.rows(), 10);
  for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.row(i).cast<float>().norm() - 512), 1);
  }
}

TEST(ExtractCovariantDSPSiftFeaturesCPU, Nominal) {
  Bitmap bitmap;
  CreateImageWithSquare(256, &bitmap);

  SiftExtractionOptions options;
  options.use_gpu = false;
  options.estimate_affine_shape = false;
  options.domain_size_pooling = true;
  options.force_covariant_extractor = false;
  auto extractor = CreateSiftFeatureExtractor(options);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

  EXPECT_EQ(keypoints.size(), 22);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    EXPECT_GE(keypoints[i].x, 0);
    EXPECT_GE(keypoints[i].y, 0);
    EXPECT_LE(keypoints[i].x, bitmap.Width());
    EXPECT_LE(keypoints[i].y, bitmap.Height());
    EXPECT_GT(keypoints[i].ComputeScale(), 0);
    EXPECT_GT(keypoints[i].ComputeOrientation(), -M_PI);
    EXPECT_LT(keypoints[i].ComputeOrientation(), M_PI);
  }

  EXPECT_EQ(descriptors.rows(), 22);
  for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.row(i).cast<float>().norm() - 512), 1);
  }
}

TEST(ExtractCovariantAffineDSPSiftFeaturesCPU, Nominal) {
  Bitmap bitmap;
  CreateImageWithSquare(256, &bitmap);

  SiftExtractionOptions options;
  options.use_gpu = false;
  options.estimate_affine_shape = true;
  options.domain_size_pooling = true;
  options.force_covariant_extractor = false;
  auto extractor = CreateSiftFeatureExtractor(options);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

  EXPECT_EQ(keypoints.size(), 22);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    EXPECT_GE(keypoints[i].x, 0);
    EXPECT_GE(keypoints[i].y, 0);
    EXPECT_LE(keypoints[i].x, bitmap.Width());
    EXPECT_LE(keypoints[i].y, bitmap.Height());
    EXPECT_GT(keypoints[i].ComputeScale(), 0);
    EXPECT_GT(keypoints[i].ComputeOrientation(), -M_PI);
    EXPECT_LT(keypoints[i].ComputeOrientation(), M_PI);
  }

  EXPECT_EQ(descriptors.rows(), 22);
  for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.row(i).cast<float>().norm() - 512), 1);
  }
}

TEST(ExtractSiftFeaturesGPU, Nominal) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  class TestThread : public Thread {
   private:
    void Run() {
      opengl_context_.MakeCurrent();

      Bitmap bitmap;
      CreateImageWithSquare(256, &bitmap);

      SiftExtractionOptions options;
      options.use_gpu = true;
      options.estimate_affine_shape = false;
      options.domain_size_pooling = false;
      options.force_covariant_extractor = false;
      auto extractor = CreateSiftFeatureExtractor(options);

      FeatureKeypoints keypoints;
      FeatureDescriptors descriptors;
      EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

      EXPECT_GE(keypoints.size(), 12);
      for (size_t i = 0; i < keypoints.size(); ++i) {
        EXPECT_GE(keypoints[i].x, 0);
        EXPECT_GE(keypoints[i].y, 0);
        EXPECT_LE(keypoints[i].x, bitmap.Width());
        EXPECT_LE(keypoints[i].y, bitmap.Height());
        EXPECT_GT(keypoints[i].ComputeScale(), 0);
        EXPECT_GT(keypoints[i].ComputeOrientation(), -M_PI);
        EXPECT_LT(keypoints[i].ComputeOrientation(), M_PI);
      }

      EXPECT_GE(descriptors.rows(), 12);
      for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
        EXPECT_LT(std::abs(descriptors.row(i).cast<float>().norm() - 512), 1);
      }
    }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  RunThreadWithOpenGLContext(&thread);
}

FeatureDescriptors CreateRandomFeatureDescriptors(const size_t num_features) {
  SetPRNGSeed(0);
  FeatureDescriptorsFloat descriptors_float =
      FeatureDescriptorsFloat::Zero(num_features, 128);
  std::vector<int> dims(128);
  std::iota(dims.begin(), dims.end(), 0);
  for (size_t i = 0; i < num_features; ++i) {
    std::shuffle(dims.begin(), dims.end(), *PRNG);
    for (size_t j = 0; j < 10; ++j) {
      descriptors_float(i, dims[j]) = 1.0f;
    }
  }
  L2NormalizeFeatureDescriptors(&descriptors_float);
  return FeatureDescriptorsToUnsignedByte(descriptors_float);
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
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  class TestThread : public Thread {
   private:
    void Run() {
      opengl_context_.MakeCurrent();
      SiftMatchingOptions options;
      options.use_gpu = true;
      options.max_num_matches = 1000;
      EXPECT_NE(CreateSiftFeatureMatcher(options), nullptr);
    }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  RunThreadWithOpenGLContext(&thread);
}

TEST(CreateSiftGPUMatcherCUDA, Nominal) {
#if defined(COLMAP_CUDA_ENABLED)
  SiftMatchingOptions options;
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
          index->Build(this->image_descriptors_.at(image_id)->cast<float>());
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
  const FeatureMatcher::Image image0 = {
      0, std::make_shared<FeatureDescriptors>(0, 128)};
  const FeatureMatcher::Image image1 = {
      1,
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      2,
      std::make_shared<FeatureDescriptors>(
          image1.descriptors->colwise().reverse())};

  FeatureDescriptorIndexCacheHelper index_cache_helper(
      {image0, image1, image2});

  SiftMatchingOptions options;
  options.use_gpu = false;
  options.cpu_brute_force_matcher = false;
  options.cpu_descriptor_index_cache = &index_cache_helper.index_cache;
  auto matcher = CreateSiftFeatureMatcher(options);

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);
  EXPECT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0].point2D_idx1, 0);
  EXPECT_EQ(matches[0].point2D_idx2, 1);
  EXPECT_EQ(matches[1].point2D_idx1, 1);
  EXPECT_EQ(matches[1].point2D_idx2, 0);

  matcher->Match(image1, image2, &matches);
  EXPECT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0].point2D_idx1, 0);
  EXPECT_EQ(matches[0].point2D_idx2, 1);
  EXPECT_EQ(matches[1].point2D_idx1, 1);
  EXPECT_EQ(matches[1].point2D_idx2, 0);

  matcher->Match(image0, image2, &matches);
  EXPECT_EQ(matches.size(), 0);
  matcher->Match(image1, image0, &matches);
  EXPECT_EQ(matches.size(), 0);
  matcher->Match(image0, image0, &matches);
  EXPECT_EQ(matches.size(), 0);
}

TEST(SiftCPUFeatureMatcherFaissVsBruteForce, Nominal) {
  SiftMatchingOptions match_options;
  match_options.max_num_matches = 1000;

  auto TestFaissVsBruteForce = [](const SiftMatchingOptions& options,
                                  const FeatureDescriptors& descriptors1,
                                  const FeatureDescriptors& descriptors2) {
    const FeatureMatcher::Image image0 = {
        0, std::make_shared<FeatureDescriptors>(0, 128)};
    const FeatureMatcher::Image image1 = {
        1, std::make_shared<FeatureDescriptors>(descriptors1)};
    const FeatureMatcher::Image image2 = {
        2, std::make_shared<FeatureDescriptors>(descriptors2)};

    FeatureDescriptorIndexCacheHelper index_cache_helper(
        {image0, image1, image2});

    FeatureMatches matches_bf;
    FeatureMatches matches_faiss;

    SiftMatchingOptions custom_options = options;
    custom_options.use_gpu = false;
    custom_options.cpu_brute_force_matcher = true;
    auto bf_matcher = CreateSiftFeatureMatcher(custom_options);
    custom_options.cpu_brute_force_matcher = false;
    custom_options.cpu_descriptor_index_cache = &index_cache_helper.index_cache;
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
    SiftMatchingOptions match_options;
    TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
  }

  {
    const FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
    const FeatureDescriptors descriptors2 = descriptors1.colwise().reverse();
    SiftMatchingOptions match_options;
    const size_t num_matches =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches, 50);
  }

  // Check the ratio test.
  {
    FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
    FeatureDescriptors descriptors2 = descriptors1;

    SiftMatchingOptions match_options;
    const size_t num_matches1 =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches1, 50);

    descriptors2.row(49) = descriptors2.row(0);
    descriptors2(0, 0) += 50.0f;
    descriptors2.row(0) = FeatureDescriptorsToUnsignedByte(
        descriptors2.row(0).cast<float>().normalized());
    descriptors2(49, 0) += 100.0f;
    descriptors2.row(49) = FeatureDescriptorsToUnsignedByte(
        descriptors2.row(49).cast<float>().normalized());

    match_options.max_ratio = 0.4;
    const size_t num_matches2 = TestFaissVsBruteForce(
        match_options, descriptors1.topRows(49), descriptors2);
    EXPECT_EQ(num_matches2, 48);

    match_options.max_ratio = 0.6;
    const size_t num_matches3 =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches3, 49);
  }

  // Check the cross check.
  {
    FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
    FeatureDescriptors descriptors2 = descriptors1;
    descriptors1.row(0) = descriptors1.row(1);

    SiftMatchingOptions match_options;

    match_options.cross_check = false;
    const size_t num_matches1 =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches1, 50);

    match_options.cross_check = true;
    const size_t num_matches2 =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches2, 48);
  }
}

TEST(MatchGuidedSiftFeaturesCPU, Nominal) {
  const FeatureMatcher::Image image0 = {
      0,
      std::make_shared<FeatureDescriptors>(0, 128),
      std::make_shared<FeatureKeypoints>(0)};
  const FeatureMatcher::Image image1 = {
      1,
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2)),
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{1, 0}, {2, 0}})};
  const FeatureMatcher::Image image2 = {
      2,
      std::make_shared<FeatureDescriptors>(
          image1.descriptors->colwise().reverse()),
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{2, 0}, {1, 0}})};
  const FeatureMatcher::Image image3 = {
      3,
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2)),
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{100, 0}, {2, 0}})};

  FeatureDescriptorIndexCacheHelper index_cache_helper(
      {image0, image1, image2, image3});

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::PLANAR_OR_PANORAMIC;
  two_view_geometry.H = Eigen::Matrix3d::Identity();

  SiftMatchingOptions options;
  options.use_gpu = false;
  options.cpu_descriptor_index_cache = &index_cache_helper.index_cache;
  auto matcher = CreateSiftFeatureMatcher(options);

  constexpr double kMaxError = 4.0;

  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 2);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx1, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx2, 0);

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

TEST(MatchSiftFeaturesGPU, Nominal) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  class TestThread : public Thread {
   private:
    void Run() {
      opengl_context_.MakeCurrent();
      SiftMatchingOptions options;
      options.use_gpu = true;
      options.max_num_matches = 1000;
      auto matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));

      const FeatureMatcher::Image image0 = {
          0, std::make_shared<FeatureDescriptors>(0, 128)};
      const FeatureMatcher::Image image1 = {
          1,
          std::make_shared<FeatureDescriptors>(
              CreateRandomFeatureDescriptors(2))};
      const FeatureMatcher::Image image2 = {
          2,
          std::make_shared<FeatureDescriptors>(
              image1.descriptors->colwise().reverse())};

      FeatureMatches matches;

      matcher->Match(image1, image2, &matches);
      EXPECT_EQ(matches.size(), 2);
      EXPECT_EQ(matches[0].point2D_idx1, 0);
      EXPECT_EQ(matches[0].point2D_idx2, 1);
      EXPECT_EQ(matches[1].point2D_idx1, 1);
      EXPECT_EQ(matches[1].point2D_idx2, 0);

      matcher->Match(image1, image2, &matches);
      EXPECT_EQ(matches.size(), 2);
      EXPECT_EQ(matches[0].point2D_idx1, 0);
      EXPECT_EQ(matches[0].point2D_idx2, 1);
      EXPECT_EQ(matches[1].point2D_idx1, 1);
      EXPECT_EQ(matches[1].point2D_idx2, 0);

      matcher->Match(image0, image2, &matches);
      EXPECT_EQ(matches.size(), 0);
      matcher->Match(image1, image0, &matches);
      EXPECT_EQ(matches.size(), 0);
      matcher->Match(image0, image0, &matches);
      EXPECT_EQ(matches.size(), 0);
    }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  RunThreadWithOpenGLContext(&thread);
}

TEST(MatchSiftFeaturesCPUvsGPU, Nominal) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  class TestThread : public Thread {
   private:
    void Run() {
      opengl_context_.MakeCurrent();

      auto TestCPUvsGPU = [](const SiftMatchingOptions& options,
                             const FeatureDescriptors& descriptors1,
                             const FeatureDescriptors& descriptors2) {
        const FeatureMatcher::Image image0 = {
            0, std::make_shared<FeatureDescriptors>(0, 128)};
        const FeatureMatcher::Image image1 = {
            1, std::make_shared<FeatureDescriptors>(descriptors1)};
        const FeatureMatcher::Image image2 = {
            2, std::make_shared<FeatureDescriptors>(descriptors2)};

        FeatureDescriptorIndexCacheHelper index_cache_helper(
            {image0, image1, image2});

        SiftMatchingOptions custom_options = options;
        custom_options.use_gpu = true;
        custom_options.max_num_matches = 1000;
        auto gpu_matcher =
            THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(custom_options));
        custom_options.use_gpu = false;
        custom_options.cpu_descriptor_index_cache =
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
        SiftMatchingOptions match_options;
        TestCPUvsGPU(match_options, descriptors1, descriptors2);
      }

      {
        const FeatureDescriptors descriptors1 =
            CreateRandomFeatureDescriptors(50);
        const FeatureDescriptors descriptors2 =
            descriptors1.colwise().reverse();
        SiftMatchingOptions match_options;
        const size_t num_matches =
            TestCPUvsGPU(match_options, descriptors1, descriptors2);
        EXPECT_EQ(num_matches, 50);
      }

      // Check the ratio test.
      {
        FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
        FeatureDescriptors descriptors2 = descriptors1;

        SiftMatchingOptions match_options;
        const size_t num_matches1 =
            TestCPUvsGPU(match_options, descriptors1, descriptors2);
        EXPECT_EQ(num_matches1, 50);

        descriptors2.row(49) = descriptors2.row(0);
        descriptors2(0, 0) += 50.0f;
        descriptors2.row(0) = FeatureDescriptorsToUnsignedByte(
            descriptors2.row(0).cast<float>().normalized());
        descriptors2(49, 0) += 100.0f;
        descriptors2.row(49) = FeatureDescriptorsToUnsignedByte(
            descriptors2.row(49).cast<float>().normalized());

        match_options.max_ratio = 0.4;
        const size_t num_matches2 =
            TestCPUvsGPU(match_options, descriptors1.topRows(49), descriptors2);
        EXPECT_EQ(num_matches2, 48);

        match_options.max_ratio = 0.6;
        const size_t num_matches3 =
            TestCPUvsGPU(match_options, descriptors1, descriptors2);
        EXPECT_EQ(num_matches3, 49);
      }

      // Check the cross check.
      {
        FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
        FeatureDescriptors descriptors2 = descriptors1;
        descriptors1.row(0) = descriptors1.row(1);

        SiftMatchingOptions match_options;

        match_options.cross_check = false;
        const size_t num_matches1 =
            TestCPUvsGPU(match_options, descriptors1, descriptors2);
        EXPECT_EQ(num_matches1, 50);

        match_options.cross_check = true;
        const size_t num_matches2 =
            TestCPUvsGPU(match_options, descriptors1, descriptors2);
        EXPECT_EQ(num_matches2, 48);
      }
    }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  RunThreadWithOpenGLContext(&thread);
}

TEST(MatchGuidedSiftFeaturesGPU, Nominal) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  class TestThread : public Thread {
   private:
    void Run() {
      const FeatureMatcher::Image image0 = {
          0,
          std::make_shared<FeatureDescriptors>(0, 128),
          std::make_shared<FeatureKeypoints>(0)};
      const FeatureMatcher::Image image1 = {
          1,
          std::make_shared<FeatureDescriptors>(
              CreateRandomFeatureDescriptors(2)),
          std::make_shared<FeatureKeypoints>(
              std::vector<FeatureKeypoint>{{1, 0}, {2, 0}})};
      const FeatureMatcher::Image image2 = {
          2,
          std::make_shared<FeatureDescriptors>(
              image1.descriptors->colwise().reverse()),
          std::make_shared<FeatureKeypoints>(
              std::vector<FeatureKeypoint>{{2, 0}, {1, 0}})};
      const FeatureMatcher::Image image3 = {
          3,
          std::make_shared<FeatureDescriptors>(
              CreateRandomFeatureDescriptors(2)),
          std::make_shared<FeatureKeypoints>(
              std::vector<FeatureKeypoint>{{100, 0}, {1, 0}})};

      opengl_context_.MakeCurrent();
      SiftMatchingOptions options;
      options.use_gpu = true;
      options.max_num_matches = 1000;
      auto matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));

      TwoViewGeometry two_view_geometry;
      two_view_geometry.config = TwoViewGeometry::PLANAR_OR_PANORAMIC;
      two_view_geometry.H = Eigen::Matrix3d::Identity();

      constexpr double kMaxError = 4.0;

      matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
      EXPECT_EQ(two_view_geometry.inlier_matches.size(), 2);
      EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
      EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
      EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx1, 1);
      EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx2, 0);

      matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
      EXPECT_EQ(two_view_geometry.inlier_matches.size(), 2);
      EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
      EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
      EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx1, 1);
      EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx2, 0);

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
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  RunThreadWithOpenGLContext(&thread);
}

}  // namespace
}  // namespace colmap

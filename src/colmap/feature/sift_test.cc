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
#include "colmap/util/testing.h"

#include <fstream>

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

TEST(ExtractSiftFeaturesCPU, Nominal) {
  const Bitmap bitmap = CreateImageWithSquare(256);

  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = false;
  options.sift->estimate_affine_shape = false;
  options.sift->domain_size_pooling = false;
  options.sift->force_covariant_extractor = false;
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

  EXPECT_EQ(descriptors.data.rows(), 22);
  EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
  for (Eigen::Index i = 0; i < descriptors.data.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.data.row(i).cast<float>().norm() - 512), 1);
  }
}

TEST(ExtractCovariantSiftFeaturesCPU, Nominal) {
  const Bitmap bitmap = CreateImageWithSquare(256);

  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = false;
  options.sift->estimate_affine_shape = false;
  options.sift->domain_size_pooling = false;
  options.sift->force_covariant_extractor = true;
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

  EXPECT_EQ(descriptors.data.rows(), 22);
  EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
  for (Eigen::Index i = 0; i < descriptors.data.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.data.row(i).cast<float>().norm() - 512), 1);
  }
}

TEST(ExtractCovariantAffineSiftFeaturesCPU, Nominal) {
  const Bitmap bitmap = CreateImageWithSquare(256);

  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = false;
  options.sift->estimate_affine_shape = true;
  options.sift->domain_size_pooling = false;
  options.sift->force_covariant_extractor = false;
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

  EXPECT_EQ(descriptors.data.rows(), 22);
  EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
  for (Eigen::Index i = 0; i < descriptors.data.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.data.row(i).cast<float>().norm() - 512), 1);
  }
}

TEST(ExtractCovariantAffineSiftFeaturesCPU, Upright) {
  const Bitmap bitmap = CreateImageWithSquare(256);

  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = false;
  options.sift->estimate_affine_shape = true;
  options.sift->upright = true;
  options.sift->domain_size_pooling = false;
  options.sift->force_covariant_extractor = false;
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

  EXPECT_EQ(descriptors.data.rows(), 10);
  EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
  for (Eigen::Index i = 0; i < descriptors.data.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.data.row(i).cast<float>().norm() - 512), 1);
  }
}

TEST(ExtractCovariantDSPSiftFeaturesCPU, Nominal) {
  const Bitmap bitmap = CreateImageWithSquare(256);

  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = false;
  options.sift->estimate_affine_shape = false;
  options.sift->domain_size_pooling = true;
  options.sift->force_covariant_extractor = false;
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

  EXPECT_EQ(descriptors.data.rows(), 22);
  EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
  for (Eigen::Index i = 0; i < descriptors.data.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.data.row(i).cast<float>().norm() - 512), 1);
  }
}

TEST(ExtractCovariantAffineDSPSiftFeaturesCPU, Nominal) {
  const Bitmap bitmap = CreateImageWithSquare(256);

  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = false;
  options.sift->estimate_affine_shape = true;
  options.sift->domain_size_pooling = true;
  options.sift->force_covariant_extractor = false;
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

  EXPECT_EQ(descriptors.data.rows(), 22);
  EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
  for (Eigen::Index i = 0; i < descriptors.data.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.data.row(i).cast<float>().norm() - 512), 1);
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
      for (size_t i = 0; i < keypoints.size(); ++i) {
        EXPECT_GE(keypoints[i].x, 0);
        EXPECT_GE(keypoints[i].y, 0);
        EXPECT_LE(keypoints[i].x, bitmap.Width());
        EXPECT_LE(keypoints[i].y, bitmap.Height());
        EXPECT_GT(keypoints[i].ComputeScale(), 0);
        EXPECT_GT(keypoints[i].ComputeOrientation(), -M_PI);
        EXPECT_LT(keypoints[i].ComputeOrientation(), M_PI);
      }

      EXPECT_GE(descriptors.data.rows(), 12);
      EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
      for (Eigen::Index i = 0; i < descriptors.data.rows(); ++i) {
        EXPECT_LT(std::abs(descriptors.data.row(i).cast<float>().norm() - 512),
                  1);
      }
    }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  RunThreadWithOpenGLContext(&thread);
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
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  class TestThread : public Thread {
   private:
    void Run() {
      opengl_context_.MakeCurrent();
      FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
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

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::PLANAR_OR_PANORAMIC;
  two_view_geometry.H = Eigen::Matrix3d::Identity();

  EXPECT_THROW(matcher->MatchGuided(
                   1.0, image_sift, image_undefined, &two_view_geometry),
               std::invalid_argument);
}

TEST(MatchSiftFeaturesGPU, TypeMismatch) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  class TestThread : public Thread {
   private:
    void Run() {
      opengl_context_.MakeCurrent();
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
    }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  RunThreadWithOpenGLContext(&thread);
}

TEST(MatchGuidedSiftFeaturesGPU, TypeMismatch) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  class TestThread : public Thread {
   private:
    void Run() {
      opengl_context_.MakeCurrent();
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

      TwoViewGeometry two_view_geometry;
      two_view_geometry.config = TwoViewGeometry::PLANAR_OR_PANORAMIC;
      two_view_geometry.H = Eigen::Matrix3d::Identity();

      EXPECT_THROW(matcher->MatchGuided(
                       1.0, image_sift, image_undefined, &two_view_geometry),
                   std::invalid_argument);
    }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  RunThreadWithOpenGLContext(&thread);
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

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::PLANAR_OR_PANORAMIC;
  two_view_geometry.H = Eigen::Matrix3d::Identity();

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_descriptor_index_cache = &index_cache_helper.index_cache;
  auto matcher = CreateSiftFeatureMatcher(options);

  constexpr double kMaxError = 1.0;

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
  ASSERT_EQ(two_view_geometry.inlier_matches.size(), 2);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx1, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx2, 0);

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

TEST(MatchSiftFeaturesGPU, Nominal) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  class TestThread : public Thread {
   private:
    void Run() {
      opengl_context_.MakeCurrent();
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

      opengl_context_.MakeCurrent();
      FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
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

TEST(MatchGuidedSiftFeaturesGPU, EssentialMatrix) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  class TestThread : public Thread {
   private:
    void Run() {
      opengl_context_.MakeCurrent();
      TestGuidedMatchingWithCameraDistortion(
          [](const std::vector<FeatureMatcher::Image>& images) {
            FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
            options.use_gpu = true;
            options.max_num_matches = 1000;
            return THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));
          });
    }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  RunThreadWithOpenGLContext(&thread);
}

////////////////////////////////////////////////////////////////////////////////
// New tests for expanded coverage
////////////////////////////////////////////////////////////////////////////////

TEST(SiftExtractionOptions, CheckValid) {
  SiftExtractionOptions options;
  EXPECT_TRUE(options.Check());
}

TEST(SiftExtractionOptions, CheckInvalidMaxNumFeatures) {
  SiftExtractionOptions options;
  options.max_num_features = 0;
  EXPECT_FALSE(options.Check());

  options.max_num_features = -1;
  EXPECT_FALSE(options.Check());
}

TEST(SiftExtractionOptions, CheckInvalidOctaveResolution) {
  SiftExtractionOptions options;
  options.octave_resolution = 0;
  EXPECT_FALSE(options.Check());
}

TEST(SiftExtractionOptions, CheckInvalidPeakThreshold) {
  SiftExtractionOptions options;
  options.peak_threshold = 0.0;
  EXPECT_FALSE(options.Check());

  options.peak_threshold = -1.0;
  EXPECT_FALSE(options.Check());
}

TEST(SiftExtractionOptions, CheckInvalidEdgeThreshold) {
  SiftExtractionOptions options;
  options.edge_threshold = 0.0;
  EXPECT_FALSE(options.Check());
}

TEST(SiftExtractionOptions, CheckInvalidMaxNumOrientations) {
  SiftExtractionOptions options;
  options.max_num_orientations = 0;
  EXPECT_FALSE(options.Check());
}

TEST(SiftExtractionOptions, CheckDSPValid) {
  SiftExtractionOptions options;
  options.domain_size_pooling = true;
  EXPECT_TRUE(options.Check());
}

TEST(SiftExtractionOptions, CheckDSPInvalidMinScale) {
  SiftExtractionOptions options;
  options.domain_size_pooling = true;
  options.dsp_min_scale = 0;
  EXPECT_FALSE(options.Check());
}

TEST(SiftExtractionOptions, CheckDSPInvalidMaxScale) {
  SiftExtractionOptions options;
  options.domain_size_pooling = true;
  // max_scale < min_scale violates GE check
  options.dsp_min_scale = 3.0;
  options.dsp_max_scale = 1.0;
  EXPECT_FALSE(options.Check());
}

TEST(SiftExtractionOptions, CheckDSPInvalidNumScales) {
  SiftExtractionOptions options;
  options.domain_size_pooling = true;
  options.dsp_num_scales = 0;
  EXPECT_FALSE(options.Check());
}

TEST(SiftExtractionOptions, CheckDSPNotEnabledSkipsValidation) {
  // When DSP is disabled, invalid DSP params should not cause failure.
  SiftExtractionOptions options;
  options.domain_size_pooling = false;
  options.dsp_min_scale = 0;
  options.dsp_max_scale = -1;
  options.dsp_num_scales = 0;
  EXPECT_TRUE(options.Check());
}

TEST(SiftMatchingOptions, CheckValid) {
  SiftMatchingOptions options;
  EXPECT_TRUE(options.Check());
}

TEST(SiftMatchingOptions, CheckInvalidMaxRatio) {
  SiftMatchingOptions options;
  options.max_ratio = 0.0;
  EXPECT_FALSE(options.Check());

  options.max_ratio = -0.5;
  EXPECT_FALSE(options.Check());
}

TEST(SiftMatchingOptions, CheckInvalidMaxDistance) {
  SiftMatchingOptions options;
  options.max_distance = 0.0;
  EXPECT_FALSE(options.Check());
}

TEST(ExtractSiftFeaturesCPU, L2Normalization) {
  const Bitmap bitmap = CreateImageWithSquare(256);

  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = false;
  options.sift->estimate_affine_shape = false;
  options.sift->domain_size_pooling = false;
  options.sift->force_covariant_extractor = false;
  options.sift->normalization = SiftExtractionOptions::Normalization::L2;
  auto extractor = CreateSiftFeatureExtractor(options);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

  EXPECT_EQ(keypoints.size(), 22);
  EXPECT_EQ(descriptors.data.rows(), 22);
  EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
  for (Eigen::Index i = 0; i < descriptors.data.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.data.row(i).cast<float>().norm() - 512), 1);
  }
}

TEST(ExtractCovariantSiftFeaturesCPU, L2Normalization) {
  const Bitmap bitmap = CreateImageWithSquare(256);

  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = false;
  options.sift->estimate_affine_shape = false;
  options.sift->domain_size_pooling = false;
  options.sift->force_covariant_extractor = true;
  options.sift->normalization = SiftExtractionOptions::Normalization::L2;
  auto extractor = CreateSiftFeatureExtractor(options);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

  EXPECT_EQ(keypoints.size(), 22);
  EXPECT_EQ(descriptors.data.rows(), 22);
  EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
  for (Eigen::Index i = 0; i < descriptors.data.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.data.row(i).cast<float>().norm() - 512), 2);
  }
}

TEST(ExtractSiftFeaturesCPU, UprightSingleOrientation) {
  const Bitmap bitmap = CreateImageWithSquare(256);

  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = false;
  options.sift->estimate_affine_shape = false;
  options.sift->domain_size_pooling = false;
  options.sift->force_covariant_extractor = false;
  options.sift->upright = true;
  auto extractor = CreateSiftFeatureExtractor(options);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

  // Upright forces orientation to 0 and only one orientation per keypoint,
  // so we get fewer features than the non-upright case.
  EXPECT_GT(keypoints.size(), 0);
  EXPECT_LE(keypoints.size(), 22);
  EXPECT_EQ(descriptors.data.rows(), keypoints.size());
  EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
}

TEST(ExtractSiftFeaturesCPU, SmallImage) {
  // Very small image may produce zero or very few features.
  Bitmap bitmap(16, 16, false);
  bitmap.Fill(BitmapColor<uint8_t>(128, 128, 128));

  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = false;
  options.sift->estimate_affine_shape = false;
  options.sift->domain_size_pooling = false;
  options.sift->force_covariant_extractor = false;
  auto extractor = CreateSiftFeatureExtractor(options);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

  // A uniform small image should produce no features.
  EXPECT_EQ(keypoints.size(), 0);
  EXPECT_EQ(descriptors.data.rows(), 0);
}

TEST(ExtractCovariantSiftFeaturesCPU, SmallImage) {
  Bitmap bitmap(16, 16, false);
  bitmap.Fill(BitmapColor<uint8_t>(128, 128, 128));

  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = false;
  options.sift->estimate_affine_shape = false;
  options.sift->domain_size_pooling = false;
  options.sift->force_covariant_extractor = true;
  auto extractor = CreateSiftFeatureExtractor(options);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

  EXPECT_EQ(keypoints.size(), 0);
  EXPECT_EQ(descriptors.data.rows(), 0);
}

TEST(ExtractSiftFeaturesCPU, MaxNumFeaturesLimit) {
  // Use a rich image that generates many features, then limit.
  const Bitmap bitmap = CreateImageWithSquare(256);

  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = false;
  options.sift->estimate_affine_shape = false;
  options.sift->domain_size_pooling = false;
  options.sift->force_covariant_extractor = false;

  // First extract with a high limit.
  options.sift->max_num_features = 8192;
  auto extractor_full = CreateSiftFeatureExtractor(options);
  FeatureKeypoints keypoints_full;
  FeatureDescriptors descriptors_full;
  EXPECT_TRUE(
      extractor_full->Extract(bitmap, &keypoints_full, &descriptors_full));
  const size_t full_count = keypoints_full.size();
  EXPECT_GT(full_count, 1);

  // Now extract with a very low limit.
  options.sift->max_num_features = 1;
  auto extractor_limited = CreateSiftFeatureExtractor(options);
  FeatureKeypoints keypoints_limited;
  FeatureDescriptors descriptors_limited;
  EXPECT_TRUE(extractor_limited->Extract(
      bitmap, &keypoints_limited, &descriptors_limited));

  EXPECT_LE(keypoints_limited.size(), full_count);
  EXPECT_EQ(descriptors_limited.data.rows(), keypoints_limited.size());
}

TEST(ExtractSiftFeaturesCPU, ExtractWithoutDescriptors) {
  // Passing nullptr for descriptors should still extract keypoints.
  const Bitmap bitmap = CreateImageWithSquare(256);

  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = false;
  options.sift->estimate_affine_shape = false;
  options.sift->domain_size_pooling = false;
  options.sift->force_covariant_extractor = false;
  auto extractor = CreateSiftFeatureExtractor(options);

  FeatureKeypoints keypoints;
  EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, nullptr));

  EXPECT_EQ(keypoints.size(), 22);
}

TEST(ExtractSiftFeaturesCPU, ReuseExtractorDifferentSizes) {
  // The SiftCPU extractor caches the VlSiftFilt; extracting images of
  // different sizes should still work (it re-creates the filter).
  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = false;
  options.sift->estimate_affine_shape = false;
  options.sift->domain_size_pooling = false;
  options.sift->force_covariant_extractor = false;
  auto extractor = CreateSiftFeatureExtractor(options);

  const Bitmap bitmap1 = CreateImageWithSquare(256);
  FeatureKeypoints keypoints1;
  FeatureDescriptors descriptors1;
  EXPECT_TRUE(extractor->Extract(bitmap1, &keypoints1, &descriptors1));
  EXPECT_GT(keypoints1.size(), 0);

  const Bitmap bitmap2 = CreateImageWithSquare(128);
  FeatureKeypoints keypoints2;
  FeatureDescriptors descriptors2;
  EXPECT_TRUE(extractor->Extract(bitmap2, &keypoints2, &descriptors2));
  EXPECT_GT(keypoints2.size(), 0);

  // Different image sizes should produce different feature counts.
  EXPECT_NE(keypoints1.size(), keypoints2.size());
}

TEST(CreateSiftFeatureExtractor, GPUReturnsNullWithoutGPU) {
#if !defined(COLMAP_GPU_ENABLED)
  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = true;
  options.sift->estimate_affine_shape = false;
  options.sift->domain_size_pooling = false;
  options.sift->force_covariant_extractor = false;
  auto extractor = CreateSiftFeatureExtractor(options);
  EXPECT_EQ(extractor, nullptr);
#endif
}

TEST(CreateSiftFeatureMatcher, GPUReturnsNullWithoutGPU) {
#if !defined(COLMAP_GPU_ENABLED)
  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = true;
  auto matcher = CreateSiftFeatureMatcher(options);
  EXPECT_EQ(matcher, nullptr);
#endif
}

TEST(SiftCPUFeatureMatcher, BruteForceNominal) {
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

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_brute_force_matcher = true;
  auto matcher = CreateSiftFeatureMatcher(options);

  FeatureMatches matches;
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

TEST(MatchGuidedSiftFeaturesCPU, UnsupportedConfigReturnsEmpty) {
  // Guided matching with DEGENERATE config should return immediately
  // with no matches (the else branch in the guided filter setup).
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
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

  FeatureDescriptorIndexCacheHelper index_cache_helper({image1, image2});

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_brute_force_matcher = true;
  auto matcher = CreateSiftFeatureMatcher(options);

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::DEGENERATE;

  matcher->MatchGuided(1.0, image1, image2, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
}

TEST(MatchGuidedSiftFeaturesCPU, WatermarkConfigReturnsEmpty) {
  // WATERMARK is also an unsupported config for guided matching.
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
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

  FeatureDescriptorIndexCacheHelper index_cache_helper({image1, image2});

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_brute_force_matcher = true;
  auto matcher = CreateSiftFeatureMatcher(options);

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::WATERMARK;

  matcher->MatchGuided(1.0, image1, image2, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
}

TEST(MatchGuidedSiftFeaturesCPU, BruteForceUncalibrated) {
  // Exercises the UNCALIBRATED guided matching path with brute-force matcher.
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);

  // Create points that lie on an epipolar line for identity fundamental matrix.
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

  FeatureDescriptorIndexCacheHelper index_cache_helper({image1, image2});

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_brute_force_matcher = true;
  auto matcher = CreateSiftFeatureMatcher(options);

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::UNCALIBRATED;
  // Use a fundamental matrix close to identity so all points satisfy epipolar.
  two_view_geometry.F =
      (Eigen::Matrix3d() << 0, 0, 0, 0, 0, -1, 0, 1, 0).finished();

  matcher->MatchGuided(10.0, image1, image2, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 2);
}

TEST(LoadSiftFeaturesFromTextFile, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto path = test_dir / "features.txt";

  {
    std::ofstream file(path);
    file << "2 128\n";
    // Feature 1
    file << "0.5 1.5 2.0 0.1";
    for (int j = 0; j < 128; ++j) {
      file << " " << (j % 256);
    }
    file << "\n";
    // Feature 2
    file << "10.0 20.0 3.0 1.57";
    for (int j = 0; j < 128; ++j) {
      file << " " << ((j + 50) % 256);
    }
    file << "\n";
  }

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  LoadSiftFeaturesFromTextFile(path, &keypoints, &descriptors);

  ASSERT_EQ(keypoints.size(), 2);
  EXPECT_FLOAT_EQ(keypoints[0].x, 0.5f);
  EXPECT_FLOAT_EQ(keypoints[0].y, 1.5f);
  EXPECT_NEAR(keypoints[1].x, 10.0f, 1e-4f);
  EXPECT_NEAR(keypoints[1].y, 20.0f, 1e-4f);

  EXPECT_EQ(descriptors.data.rows(), 2);
  EXPECT_EQ(descriptors.data.cols(), 128);
  EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);

  // Verify descriptor values.
  for (int j = 0; j < 128; ++j) {
    EXPECT_EQ(descriptors.data(0, j), static_cast<uint8_t>(j % 256));
    EXPECT_EQ(descriptors.data(1, j), static_cast<uint8_t>((j + 50) % 256));
  }
}

TEST(LoadSiftFeaturesFromTextFile, ZeroFeatures) {
  const auto test_dir = CreateTestDir();
  const auto path = test_dir / "empty_features.txt";

  {
    std::ofstream file(path);
    file << "0 128\n";
  }

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  LoadSiftFeaturesFromTextFile(path, &keypoints, &descriptors);

  EXPECT_EQ(keypoints.size(), 0);
  EXPECT_EQ(descriptors.data.rows(), 0);
  EXPECT_EQ(descriptors.data.cols(), 128);
  EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
}

TEST(LoadSiftFeaturesFromTextFile, InvalidDimension) {
  const auto test_dir = CreateTestDir();
  const auto path = test_dir / "bad_dim_features.txt";

  {
    std::ofstream file(path);
    // Wrong dimension (64 instead of 128).
    file << "1 64\n";
    file << "0.5 1.5 2.0 0.1";
    for (int j = 0; j < 64; ++j) {
      file << " " << j;
    }
    file << "\n";
  }

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  EXPECT_THROW(LoadSiftFeaturesFromTextFile(path, &keypoints, &descriptors),
               std::invalid_argument);
}

TEST(LoadSiftFeaturesFromTextFile, NonexistentFile) {
  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  EXPECT_THROW(LoadSiftFeaturesFromTextFile(
                   "/nonexistent/path.txt", &keypoints, &descriptors),
               std::invalid_argument);
}

TEST(MatchGuidedSiftFeaturesCPU, PlanarConfig) {
  // Test guided matching with PLANAR config (separate from
  // PLANAR_OR_PANORAMIC).
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
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

  FeatureDescriptorIndexCacheHelper index_cache_helper({image1, image2});

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_descriptor_index_cache = &index_cache_helper.index_cache;
  auto matcher = CreateSiftFeatureMatcher(options);

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::PLANAR;
  two_view_geometry.H = Eigen::Matrix3d::Identity();

  constexpr double kMaxError = 4.0;
  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 2);
}

TEST(MatchGuidedSiftFeaturesCPU, PanoramicConfig) {
  // Test guided matching with PANORAMIC config.
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
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

  FeatureDescriptorIndexCacheHelper index_cache_helper({image1, image2});

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_descriptor_index_cache = &index_cache_helper.index_cache;
  auto matcher = CreateSiftFeatureMatcher(options);

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::PANORAMIC;
  two_view_geometry.H = Eigen::Matrix3d::Identity();

  constexpr double kMaxError = 4.0;
  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 2);
}

TEST(SiftCPUFeatureMatcher, BruteForceMaxDistanceFilter) {
  // Test that max_distance filters out matches when descriptors are dissimilar.
  // Create two images with descriptors that have partial overlap:
  //   desc1[0]: dims 0-9,   desc1[1]: dims 10-19
  //   desc2[0]: dims 0-4 + 64-68,  desc2[1]: dims 10-14 + 74-78
  // Best match pairs share 5 out of 10 dims each.
  // dot_product ≈ 0.5 * 512^2, distance ≈ acos(0.5) ≈ 1.05 radians.
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);

  FeatureDescriptorsFloatData float1 =
      FeatureDescriptorsFloatData::Zero(2, 128);
  for (int j = 0; j < 10; ++j) {
    float1(0, j) = 1.0f;
    float1(1, 10 + j) = 1.0f;
  }
  L2NormalizeFeatureDescriptors(&float1);

  FeatureDescriptorsFloatData float2 =
      FeatureDescriptorsFloatData::Zero(2, 128);
  for (int j = 0; j < 5; ++j) {
    float2(0, j) = 1.0f;
    float2(0, 64 + j) = 1.0f;
    float2(1, 10 + j) = 1.0f;
    float2(1, 74 + j) = 1.0f;
  }
  L2NormalizeFeatureDescriptors(&float2);

  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(
          FeatureExtractorType::SIFT,
          FeatureDescriptorsToUnsignedByte(float1))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(
          FeatureExtractorType::SIFT,
          FeatureDescriptorsToUnsignedByte(float2))};

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_brute_force_matcher = true;

  // With a generous max_distance (> 1.05), matches pass.
  options.sift->max_distance = 1.2;
  auto matcher_normal = CreateSiftFeatureMatcher(options);
  FeatureMatches matches_normal;
  matcher_normal->Match(image1, image2, &matches_normal);
  EXPECT_EQ(matches_normal.size(), 2);

  // With a strict max_distance (< 1.05), no matches pass.
  options.sift->max_distance = 0.5;
  auto matcher_strict = CreateSiftFeatureMatcher(options);
  FeatureMatches matches_strict;
  matcher_strict->Match(image1, image2, &matches_strict);
  EXPECT_EQ(matches_strict.size(), 0);
}

}  // namespace
}  // namespace colmap

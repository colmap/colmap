// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#define TEST_NAME "feature/sift_test"
#include "util/testing.h"

#ifdef GUI_ENABLED
#include <QApplication>
#else
#include "exe/gui.h"
#endif

#include "SiftGPU/SiftGPU.h"
#include "feature/sift.h"
#include "feature/utils.h"
#include "util/math.h"
#include "util/opengl_utils.h"
#include "util/random.h"

using namespace colmap;

void CreateImageWithSquare(const int size, Bitmap* bitmap) {
  bitmap->Allocate(size, size, false);
  bitmap->Fill(BitmapColor<uint8_t>(0, 0, 0));
  for (int r = size / 2 - size / 8; r < size / 2 + size / 8; ++r) {
    for (int c = size / 2 - size / 8; c < size / 2 + size / 8; ++c) {
      bitmap->SetPixel(r, c, BitmapColor<uint8_t>(255));
    }
  }
}

BOOST_AUTO_TEST_CASE(TestExtractSiftFeaturesCPU) {
  Bitmap bitmap;
  CreateImageWithSquare(256, &bitmap);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  BOOST_CHECK(ExtractSiftFeaturesCPU(SiftExtractionOptions(), bitmap,
                                     &keypoints, &descriptors));

  BOOST_CHECK_EQUAL(keypoints.size(), 22);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    BOOST_CHECK_GE(keypoints[i].x, 0);
    BOOST_CHECK_GE(keypoints[i].y, 0);
    BOOST_CHECK_LE(keypoints[i].x, bitmap.Width());
    BOOST_CHECK_LE(keypoints[i].y, bitmap.Height());
    BOOST_CHECK_GT(keypoints[i].ComputeScale(), 0);
    BOOST_CHECK_GT(keypoints[i].ComputeOrientation(), -M_PI);
    BOOST_CHECK_LT(keypoints[i].ComputeOrientation(), M_PI);
  }

  BOOST_CHECK_EQUAL(descriptors.rows(), 22);
  for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
    BOOST_CHECK_LT(std::abs(descriptors.row(i).cast<float>().norm() - 512), 1);
  }
}

BOOST_AUTO_TEST_CASE(TestExtractCovariantSiftFeaturesCPU) {
  Bitmap bitmap;
  CreateImageWithSquare(256, &bitmap);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  BOOST_CHECK(ExtractCovariantSiftFeaturesCPU(SiftExtractionOptions(), bitmap,
                                              &keypoints, &descriptors));

  BOOST_CHECK_EQUAL(keypoints.size(), 22);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    BOOST_CHECK_GE(keypoints[i].x, 0);
    BOOST_CHECK_GE(keypoints[i].y, 0);
    BOOST_CHECK_LE(keypoints[i].x, bitmap.Width());
    BOOST_CHECK_LE(keypoints[i].y, bitmap.Height());
    BOOST_CHECK_GT(keypoints[i].ComputeScale(), 0);
    BOOST_CHECK_GT(keypoints[i].ComputeOrientation(), -M_PI);
    BOOST_CHECK_LT(keypoints[i].ComputeOrientation(), M_PI);
  }

  BOOST_CHECK_EQUAL(descriptors.rows(), 22);
  for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
    BOOST_CHECK_LT(std::abs(descriptors.row(i).cast<float>().norm() - 512), 1);
  }
}

BOOST_AUTO_TEST_CASE(TestExtractCovariantAffineSiftFeaturesCPU) {
  Bitmap bitmap;
  CreateImageWithSquare(256, &bitmap);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  SiftExtractionOptions options;
  options.estimate_affine_shape = true;
  BOOST_CHECK(ExtractCovariantSiftFeaturesCPU(options, bitmap, &keypoints,
                                              &descriptors));

  BOOST_CHECK_EQUAL(keypoints.size(), 10);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    BOOST_CHECK_GE(keypoints[i].x, 0);
    BOOST_CHECK_GE(keypoints[i].y, 0);
    BOOST_CHECK_LE(keypoints[i].x, bitmap.Width());
    BOOST_CHECK_LE(keypoints[i].y, bitmap.Height());
    BOOST_CHECK_GT(keypoints[i].ComputeScale(), 0);
    BOOST_CHECK_GT(keypoints[i].ComputeOrientation(), -M_PI);
    BOOST_CHECK_LT(keypoints[i].ComputeOrientation(), M_PI);
  }

  BOOST_CHECK_EQUAL(descriptors.rows(), 10);
  for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
    BOOST_CHECK_LT(std::abs(descriptors.row(i).cast<float>().norm() - 512), 1);
  }
}

BOOST_AUTO_TEST_CASE(TestExtractCovariantDSPSiftFeaturesCPU) {
  Bitmap bitmap;
  CreateImageWithSquare(256, &bitmap);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  SiftExtractionOptions options;
  options.domain_size_pooling = true;
  BOOST_CHECK(ExtractCovariantSiftFeaturesCPU(options, bitmap, &keypoints,
                                              &descriptors));

  BOOST_CHECK_EQUAL(keypoints.size(), 22);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    BOOST_CHECK_GE(keypoints[i].x, 0);
    BOOST_CHECK_GE(keypoints[i].y, 0);
    BOOST_CHECK_LE(keypoints[i].x, bitmap.Width());
    BOOST_CHECK_LE(keypoints[i].y, bitmap.Height());
    BOOST_CHECK_GT(keypoints[i].ComputeScale(), 0);
    BOOST_CHECK_GT(keypoints[i].ComputeOrientation(), -M_PI);
    BOOST_CHECK_LT(keypoints[i].ComputeOrientation(), M_PI);
  }

  BOOST_CHECK_EQUAL(descriptors.rows(), 22);
  for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
    BOOST_CHECK_LT(std::abs(descriptors.row(i).cast<float>().norm() - 512), 1);
  }
}

BOOST_AUTO_TEST_CASE(TestExtractCovariantAffineDSPSiftFeaturesCPU) {
  Bitmap bitmap;
  CreateImageWithSquare(256, &bitmap);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  SiftExtractionOptions options;
  options.estimate_affine_shape = true;
  options.domain_size_pooling = true;
  BOOST_CHECK(ExtractCovariantSiftFeaturesCPU(options, bitmap, &keypoints,
                                              &descriptors));

  BOOST_CHECK_EQUAL(keypoints.size(), 10);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    BOOST_CHECK_GE(keypoints[i].x, 0);
    BOOST_CHECK_GE(keypoints[i].y, 0);
    BOOST_CHECK_LE(keypoints[i].x, bitmap.Width());
    BOOST_CHECK_LE(keypoints[i].y, bitmap.Height());
    BOOST_CHECK_GT(keypoints[i].ComputeScale(), 0);
    BOOST_CHECK_GT(keypoints[i].ComputeOrientation(), -M_PI);
    BOOST_CHECK_LT(keypoints[i].ComputeOrientation(), M_PI);
  }

  BOOST_CHECK_EQUAL(descriptors.rows(), 10);
  for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
    BOOST_CHECK_LT(std::abs(descriptors.row(i).cast<float>().norm() - 512), 1);
  }
}

BOOST_AUTO_TEST_CASE(TestExtractSiftFeaturesGPU) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  if (!OpenGLContextManager::HasOpenGL()) {
    return;
  }

  class TestThread : public Thread {
   private:
    void Run() {
      opengl_context_.MakeCurrent();

      Bitmap bitmap;
      CreateImageWithSquare(256, &bitmap);

      SiftGPU sift_gpu;
      BOOST_CHECK(CreateSiftGPUExtractor(SiftExtractionOptions(), &sift_gpu));

      FeatureKeypoints keypoints;
      FeatureDescriptors descriptors;
      BOOST_CHECK(ExtractSiftFeaturesGPU(SiftExtractionOptions(), bitmap,
                                         &sift_gpu, &keypoints, &descriptors));

      BOOST_CHECK_EQUAL(keypoints.size(), 24);
      for (size_t i = 0; i < keypoints.size(); ++i) {
        BOOST_CHECK_GE(keypoints[i].x, 0);
        BOOST_CHECK_GE(keypoints[i].y, 0);
        BOOST_CHECK_LE(keypoints[i].x, bitmap.Width());
        BOOST_CHECK_LE(keypoints[i].y, bitmap.Height());
        BOOST_CHECK_GT(keypoints[i].ComputeScale(), 0);
        BOOST_CHECK_GT(keypoints[i].ComputeOrientation(), -M_PI);
        BOOST_CHECK_LT(keypoints[i].ComputeOrientation(), M_PI);
      }

      BOOST_CHECK_EQUAL(descriptors.rows(), 24);
      for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
        BOOST_CHECK_LT(std::abs(descriptors.row(i).cast<float>().norm() - 512),
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
  Eigen::MatrixXf descriptors(num_features, 128);
  for (size_t i = 0; i < num_features; ++i) {
    for (size_t j = 0; j < 128; ++j) {
      descriptors(i, j) = std::pow(RandomReal(0.0f, 1.0f), 2);
    }
  }
  return FeatureDescriptorsToUnsignedByte(
      L2NormalizeFeatureDescriptors(descriptors));
}

void CheckEqualMatches(const FeatureMatches& matches1,
                       const FeatureMatches& matches2) {
  BOOST_REQUIRE_EQUAL(matches1.size(), matches2.size());
  for (size_t i = 0; i < matches1.size(); ++i) {
    BOOST_CHECK_EQUAL(matches1[i].point2D_idx1, matches2[i].point2D_idx1);
    BOOST_CHECK_EQUAL(matches1[i].point2D_idx2, matches2[i].point2D_idx2);
  }
}

BOOST_AUTO_TEST_CASE(TestCreateSiftGPUMatcherOpenGL) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  if (!OpenGLContextManager::HasOpenGL()) {
    return;
  }

  class TestThread : public Thread {
   private:
    void Run() {
      opengl_context_.MakeCurrent();
      SiftMatchGPU sift_match_gpu;
      SiftMatchingOptions match_options;
      match_options.max_num_matches = 1000;
      BOOST_CHECK(CreateSiftGPUMatcher(match_options, &sift_match_gpu));
    }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  RunThreadWithOpenGLContext(&thread);
}

BOOST_AUTO_TEST_CASE(TestCreateSiftGPUMatcherCUDA) {
#ifdef CUDA_ENABLED
  SiftMatchGPU sift_match_gpu;
  SiftMatchingOptions match_options;
  match_options.gpu_index = "0";
  match_options.max_num_matches = 1000;
  BOOST_CHECK(CreateSiftGPUMatcher(match_options, &sift_match_gpu));
#endif
}

BOOST_AUTO_TEST_CASE(TestMatchSiftFeaturesCPU) {
  const FeatureDescriptors empty_descriptors =
      CreateRandomFeatureDescriptors(0);
  const FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(2);
  const FeatureDescriptors descriptors2 = descriptors1.colwise().reverse();

  FeatureMatches matches;

  MatchSiftFeaturesCPU(SiftMatchingOptions(), descriptors1, descriptors2,
                       &matches);
  BOOST_CHECK_EQUAL(matches.size(), 2);
  BOOST_CHECK_EQUAL(matches[0].point2D_idx1, 0);
  BOOST_CHECK_EQUAL(matches[0].point2D_idx2, 1);
  BOOST_CHECK_EQUAL(matches[1].point2D_idx1, 1);
  BOOST_CHECK_EQUAL(matches[1].point2D_idx2, 0);

  MatchSiftFeaturesCPU(SiftMatchingOptions(), empty_descriptors, descriptors2,
                       &matches);
  BOOST_CHECK_EQUAL(matches.size(), 0);
  MatchSiftFeaturesCPU(SiftMatchingOptions(), descriptors1, empty_descriptors,
                       &matches);
  BOOST_CHECK_EQUAL(matches.size(), 0);
  MatchSiftFeaturesCPU(SiftMatchingOptions(), empty_descriptors,
                       empty_descriptors, &matches);
  BOOST_CHECK_EQUAL(matches.size(), 0);
}

BOOST_AUTO_TEST_CASE(TestMatchSiftFeaturesCPUFLANNvsBruteForce) {
  SiftMatchingOptions match_options;
  match_options.max_num_matches = 1000;

  auto TestFLANNvsBruteForce = [](const SiftMatchingOptions& options,
                                  const FeatureDescriptors& descriptors1,
                                  const FeatureDescriptors& descriptors2) {
    FeatureMatches matches_bf;
    FeatureMatches matches_flann;

    MatchSiftFeaturesCPUBruteForce(options, descriptors1, descriptors2,
                                   &matches_bf);
    MatchSiftFeaturesCPUFLANN(options, descriptors1, descriptors2,
                              &matches_flann);
    CheckEqualMatches(matches_bf, matches_flann);

    const size_t num_matches = matches_bf.size();

    const FeatureDescriptors empty_descriptors =
        CreateRandomFeatureDescriptors(0);

    MatchSiftFeaturesCPUBruteForce(options, empty_descriptors, descriptors2,
                                   &matches_bf);
    MatchSiftFeaturesCPUFLANN(options, empty_descriptors, descriptors2,
                              &matches_flann);
    CheckEqualMatches(matches_bf, matches_flann);

    MatchSiftFeaturesCPUBruteForce(options, descriptors1, empty_descriptors,
                                   &matches_bf);
    MatchSiftFeaturesCPUFLANN(options, descriptors1, empty_descriptors,
                              &matches_flann);
    CheckEqualMatches(matches_bf, matches_flann);

    MatchSiftFeaturesCPUBruteForce(options, empty_descriptors,
                                   empty_descriptors, &matches_bf);
    MatchSiftFeaturesCPUFLANN(options, empty_descriptors, empty_descriptors,
                              &matches_flann);
    CheckEqualMatches(matches_bf, matches_flann);

    return num_matches;
  };

  {
    const FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(100);
    const FeatureDescriptors descriptors2 = CreateRandomFeatureDescriptors(100);
    SiftMatchingOptions match_options;
    TestFLANNvsBruteForce(match_options, descriptors1, descriptors2);
  }

  {
    const FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(100);
    const FeatureDescriptors descriptors2 = descriptors1.colwise().reverse();
    SiftMatchingOptions match_options;
    const size_t num_matches =
        TestFLANNvsBruteForce(match_options, descriptors1, descriptors2);
    BOOST_CHECK_EQUAL(num_matches, 100);
  }

  // Check the ratio test.
  {
    FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(100);
    FeatureDescriptors descriptors2 = descriptors1;

    SiftMatchingOptions match_options;
    const size_t num_matches1 =
        TestFLANNvsBruteForce(match_options, descriptors1, descriptors2);
    BOOST_CHECK_EQUAL(num_matches1, 100);

    descriptors2.row(99) = descriptors2.row(0);
    descriptors2(0, 0) += 50.0f;
    descriptors2.row(0) = FeatureDescriptorsToUnsignedByte(
        L2NormalizeFeatureDescriptors(descriptors2.row(0).cast<float>()));
    descriptors2(99, 0) += 100.0f;
    descriptors2.row(99) = FeatureDescriptorsToUnsignedByte(
        L2NormalizeFeatureDescriptors(descriptors2.row(99).cast<float>()));

    match_options.max_ratio = 0.4;
    const size_t num_matches2 = TestFLANNvsBruteForce(
        match_options, descriptors1.topRows(99), descriptors2);
    BOOST_CHECK_EQUAL(num_matches2, 98);

    match_options.max_ratio = 0.5;
    const size_t num_matches3 =
        TestFLANNvsBruteForce(match_options, descriptors1, descriptors2);
    BOOST_CHECK_EQUAL(num_matches3, 99);
  }

  // Check the cross check.
  {
    FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(100);
    FeatureDescriptors descriptors2 = descriptors1;
    descriptors1.row(0) = descriptors1.row(1);

    SiftMatchingOptions match_options;

    match_options.cross_check = false;
    const size_t num_matches1 =
        TestFLANNvsBruteForce(match_options, descriptors1, descriptors2);
    BOOST_CHECK_EQUAL(num_matches1, 100);

    match_options.cross_check = true;
    const size_t num_matches2 =
        TestFLANNvsBruteForce(match_options, descriptors1, descriptors2);
    BOOST_CHECK_EQUAL(num_matches2, 98);
  }
}

BOOST_AUTO_TEST_CASE(TestMatchGuidedSiftFeaturesCPU) {
  FeatureKeypoints empty_keypoints(0);
  FeatureKeypoints keypoints1(2);
  keypoints1[0].x = 1;
  keypoints1[1].x = 2;
  FeatureKeypoints keypoints2(2);
  keypoints2[0].x = 2;
  keypoints2[1].x = 1;
  const FeatureDescriptors empty_descriptors =
      CreateRandomFeatureDescriptors(0);
  const FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(2);
  const FeatureDescriptors descriptors2 = descriptors1.colwise().reverse();

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::PLANAR_OR_PANORAMIC;
  two_view_geometry.H = Eigen::Matrix3d::Identity();

  MatchGuidedSiftFeaturesCPU(SiftMatchingOptions(), keypoints1, keypoints2,
                             descriptors1, descriptors2, &two_view_geometry);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 2);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx1, 1);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx2, 0);

  keypoints1[0].x = 100;
  MatchGuidedSiftFeaturesCPU(SiftMatchingOptions(), keypoints1, keypoints2,
                             descriptors1, descriptors2, &two_view_geometry);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 1);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx1, 1);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx2, 0);

  MatchGuidedSiftFeaturesCPU(SiftMatchingOptions(), empty_keypoints, keypoints2,
                             empty_descriptors, descriptors2,
                             &two_view_geometry);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 0);
  MatchGuidedSiftFeaturesCPU(SiftMatchingOptions(), keypoints1, empty_keypoints,
                             descriptors1, empty_descriptors,
                             &two_view_geometry);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 0);
  MatchGuidedSiftFeaturesCPU(SiftMatchingOptions(), empty_keypoints,
                             empty_keypoints, empty_descriptors,
                             empty_descriptors, &two_view_geometry);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 0);
}

BOOST_AUTO_TEST_CASE(TestMatchSiftFeaturesGPU) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  if (!OpenGLContextManager::HasOpenGL()) {
    return;
  }

  class TestThread : public Thread {
   private:
    void Run() {
      opengl_context_.MakeCurrent();
      SiftMatchGPU sift_match_gpu;
      SiftMatchingOptions match_options;
      match_options.max_num_matches = 1000;
      BOOST_CHECK(CreateSiftGPUMatcher(match_options, &sift_match_gpu));

      const FeatureDescriptors empty_descriptors =
          CreateRandomFeatureDescriptors(0);
      const FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(2);
      const FeatureDescriptors descriptors2 = descriptors1.colwise().reverse();

      FeatureMatches matches;

      MatchSiftFeaturesGPU(SiftMatchingOptions(), &descriptors1, &descriptors2,
                           &sift_match_gpu, &matches);
      BOOST_CHECK_EQUAL(matches.size(), 2);
      BOOST_CHECK_EQUAL(matches[0].point2D_idx1, 0);
      BOOST_CHECK_EQUAL(matches[0].point2D_idx2, 1);
      BOOST_CHECK_EQUAL(matches[1].point2D_idx1, 1);
      BOOST_CHECK_EQUAL(matches[1].point2D_idx2, 0);

      MatchSiftFeaturesGPU(SiftMatchingOptions(), nullptr, nullptr,
                           &sift_match_gpu, &matches);
      BOOST_CHECK_EQUAL(matches.size(), 2);
      BOOST_CHECK_EQUAL(matches[0].point2D_idx1, 0);
      BOOST_CHECK_EQUAL(matches[0].point2D_idx2, 1);
      BOOST_CHECK_EQUAL(matches[1].point2D_idx1, 1);
      BOOST_CHECK_EQUAL(matches[1].point2D_idx2, 0);

      MatchSiftFeaturesGPU(SiftMatchingOptions(), &descriptors1, nullptr,
                           &sift_match_gpu, &matches);
      BOOST_CHECK_EQUAL(matches.size(), 2);
      BOOST_CHECK_EQUAL(matches[0].point2D_idx1, 0);
      BOOST_CHECK_EQUAL(matches[0].point2D_idx2, 1);
      BOOST_CHECK_EQUAL(matches[1].point2D_idx1, 1);
      BOOST_CHECK_EQUAL(matches[1].point2D_idx2, 0);

      MatchSiftFeaturesGPU(SiftMatchingOptions(), nullptr, &descriptors2,
                           &sift_match_gpu, &matches);
      BOOST_CHECK_EQUAL(matches.size(), 2);
      BOOST_CHECK_EQUAL(matches[0].point2D_idx1, 0);
      BOOST_CHECK_EQUAL(matches[0].point2D_idx2, 1);
      BOOST_CHECK_EQUAL(matches[1].point2D_idx1, 1);
      BOOST_CHECK_EQUAL(matches[1].point2D_idx2, 0);

      MatchSiftFeaturesGPU(SiftMatchingOptions(), &empty_descriptors,
                           &descriptors2, &sift_match_gpu, &matches);
      BOOST_CHECK_EQUAL(matches.size(), 0);
      MatchSiftFeaturesGPU(SiftMatchingOptions(), &descriptors1,
                           &empty_descriptors, &sift_match_gpu, &matches);
      BOOST_CHECK_EQUAL(matches.size(), 0);
      MatchSiftFeaturesGPU(SiftMatchingOptions(), &empty_descriptors,
                           &empty_descriptors, &sift_match_gpu, &matches);
      BOOST_CHECK_EQUAL(matches.size(), 0);
    }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  RunThreadWithOpenGLContext(&thread);
}

BOOST_AUTO_TEST_CASE(TestMatchSiftFeaturesCPUvsGPU) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  if (!OpenGLContextManager::HasOpenGL()) {
    return;
  }

  class TestThread : public Thread {
   private:
    void Run() {
      opengl_context_.MakeCurrent();
      SiftMatchGPU sift_match_gpu;
      SiftMatchingOptions match_options;
      match_options.max_num_matches = 1000;
      BOOST_CHECK(CreateSiftGPUMatcher(match_options, &sift_match_gpu));

      auto TestCPUvsGPU = [&sift_match_gpu](
                              const SiftMatchingOptions& options,
                              const FeatureDescriptors& descriptors1,
                              const FeatureDescriptors& descriptors2) {
        FeatureMatches matches_cpu;
        FeatureMatches matches_gpu;

        MatchSiftFeaturesCPU(options, descriptors1, descriptors2, &matches_cpu);
        MatchSiftFeaturesGPU(options, &descriptors1, &descriptors2,
                             &sift_match_gpu, &matches_gpu);
        CheckEqualMatches(matches_cpu, matches_gpu);

        const size_t num_matches = matches_cpu.size();

        const FeatureDescriptors empty_descriptors =
            CreateRandomFeatureDescriptors(0);

        MatchSiftFeaturesCPU(options, empty_descriptors, descriptors2,
                             &matches_cpu);
        MatchSiftFeaturesGPU(options, &empty_descriptors, &descriptors2,
                             &sift_match_gpu, &matches_gpu);
        CheckEqualMatches(matches_cpu, matches_gpu);

        MatchSiftFeaturesCPU(options, descriptors1, empty_descriptors,
                             &matches_cpu);
        MatchSiftFeaturesGPU(options, &descriptors1, &empty_descriptors,
                             &sift_match_gpu, &matches_gpu);
        CheckEqualMatches(matches_cpu, matches_gpu);

        MatchSiftFeaturesCPU(options, empty_descriptors, empty_descriptors,
                             &matches_cpu);
        MatchSiftFeaturesGPU(options, &empty_descriptors, &empty_descriptors,
                             &sift_match_gpu, &matches_gpu);
        CheckEqualMatches(matches_cpu, matches_gpu);

        return num_matches;
      };

      {
        const FeatureDescriptors descriptors1 =
            CreateRandomFeatureDescriptors(100);
        const FeatureDescriptors descriptors2 =
            CreateRandomFeatureDescriptors(100);
        SiftMatchingOptions match_options;
        TestCPUvsGPU(match_options, descriptors1, descriptors2);
      }

      {
        const FeatureDescriptors descriptors1 =
            CreateRandomFeatureDescriptors(100);
        const FeatureDescriptors descriptors2 =
            descriptors1.colwise().reverse();
        SiftMatchingOptions match_options;
        const size_t num_matches =
            TestCPUvsGPU(match_options, descriptors1, descriptors2);
        BOOST_CHECK_EQUAL(num_matches, 100);
      }

      // Check the ratio test.
      {
        FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(100);
        FeatureDescriptors descriptors2 = descriptors1;

        SiftMatchingOptions match_options;
        const size_t num_matches1 =
            TestCPUvsGPU(match_options, descriptors1, descriptors2);
        BOOST_CHECK_EQUAL(num_matches1, 100);

        descriptors2.row(99) = descriptors2.row(0);
        descriptors2(0, 0) += 50.0f;
        descriptors2.row(0) = FeatureDescriptorsToUnsignedByte(
            L2NormalizeFeatureDescriptors(descriptors2.row(0).cast<float>()));
        descriptors2(99, 0) += 100.0f;
        descriptors2.row(99) = FeatureDescriptorsToUnsignedByte(
            L2NormalizeFeatureDescriptors(descriptors2.row(99).cast<float>()));

        match_options.max_ratio = 0.4;
        const size_t num_matches2 =
            TestCPUvsGPU(match_options, descriptors1.topRows(99), descriptors2);
        BOOST_CHECK_EQUAL(num_matches2, 98);

        match_options.max_ratio = 0.5;
        const size_t num_matches3 =
            TestCPUvsGPU(match_options, descriptors1, descriptors2);
        BOOST_CHECK_EQUAL(num_matches3, 99);
      }

      // Check the cross check.
      {
        FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(100);
        FeatureDescriptors descriptors2 = descriptors1;
        descriptors1.row(0) = descriptors1.row(1);

        SiftMatchingOptions match_options;

        match_options.cross_check = false;
        const size_t num_matches1 =
            TestCPUvsGPU(match_options, descriptors1, descriptors2);
        BOOST_CHECK_EQUAL(num_matches1, 100);

        match_options.cross_check = true;
        const size_t num_matches2 =
            TestCPUvsGPU(match_options, descriptors1, descriptors2);
        BOOST_CHECK_EQUAL(num_matches2, 98);
      }
    }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  RunThreadWithOpenGLContext(&thread);
}

BOOST_AUTO_TEST_CASE(TestMatchGuidedSiftFeaturesGPU) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  if (!OpenGLContextManager::HasOpenGL()) {
    return;
  }

  class TestThread : public Thread {
   private:
    void Run() {
      opengl_context_.MakeCurrent();
      SiftMatchGPU sift_match_gpu;
      SiftMatchingOptions match_options;
      match_options.max_num_matches = 1000;
      BOOST_CHECK(CreateSiftGPUMatcher(match_options, &sift_match_gpu));

      FeatureKeypoints empty_keypoints(0);
      FeatureKeypoints keypoints1(2);
      keypoints1[0].x = 1;
      keypoints1[1].x = 2;
      FeatureKeypoints keypoints2(2);
      keypoints2[0].x = 2;
      keypoints2[1].x = 1;
      const FeatureDescriptors empty_descriptors =
          CreateRandomFeatureDescriptors(0);
      const FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(2);
      const FeatureDescriptors descriptors2 = descriptors1.colwise().reverse();

      TwoViewGeometry two_view_geometry;
      two_view_geometry.config = TwoViewGeometry::PLANAR_OR_PANORAMIC;
      two_view_geometry.H = Eigen::Matrix3d::Identity();

      MatchGuidedSiftFeaturesGPU(SiftMatchingOptions(), &keypoints1,
                                 &keypoints2, &descriptors1, &descriptors2,
                                 &sift_match_gpu, &two_view_geometry);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 2);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx1, 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx2, 0);

      MatchGuidedSiftFeaturesGPU(SiftMatchingOptions(), nullptr, nullptr,
                                 nullptr, nullptr, &sift_match_gpu,
                                 &two_view_geometry);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 2);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx1, 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx2, 0);

      MatchGuidedSiftFeaturesGPU(SiftMatchingOptions(), &keypoints1, nullptr,
                                 &descriptors1, nullptr, &sift_match_gpu,
                                 &two_view_geometry);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 2);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx1, 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx2, 0);

      MatchGuidedSiftFeaturesGPU(SiftMatchingOptions(), nullptr, &keypoints2,
                                 nullptr, &descriptors2, &sift_match_gpu,
                                 &two_view_geometry);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 2);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx1, 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx2, 0);

      keypoints1[0].x = 100;
      MatchGuidedSiftFeaturesGPU(SiftMatchingOptions(), &keypoints1,
                                 &keypoints2, &descriptors1, &descriptors2,
                                 &sift_match_gpu, &two_view_geometry);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx1, 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx2, 0);

      MatchGuidedSiftFeaturesGPU(SiftMatchingOptions(), &empty_keypoints,
                                 &keypoints2, &empty_descriptors, &descriptors2,
                                 &sift_match_gpu, &two_view_geometry);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 0);
      MatchGuidedSiftFeaturesGPU(
          SiftMatchingOptions(), &keypoints1, &empty_keypoints, &descriptors1,
          &empty_descriptors, &sift_match_gpu, &two_view_geometry);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 0);
      MatchGuidedSiftFeaturesGPU(SiftMatchingOptions(), &empty_keypoints,
                                 &empty_keypoints, &empty_descriptors,
                                 &empty_descriptors, &sift_match_gpu,
                                 &two_view_geometry);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 0);
    }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  RunThreadWithOpenGLContext(&thread);
}

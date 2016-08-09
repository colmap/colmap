// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "base/feature_matching_test"
#include <boost/test/unit_test.hpp>

#include <QApplication>

#include "base/feature_matching.h"

using namespace colmap;

FeatureDescriptors CreateRandomFeatureDescriptors(const size_t num_features) {
  const auto descriptors_float =
      L2NormalizeFeatureDescriptors(Eigen::MatrixXf::Random(num_features, 128) +
                                    Eigen::MatrixXf::Ones(num_features, 128));
  return FeatureDescriptorsToUnsignedByte(descriptors_float);
}

BOOST_AUTO_TEST_CASE(TestCreateSiftGPUMatcherOpenGL) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  class TestThread : public Thread {
   private:
    void Run() {
      opengl_context_.MakeCurrent();
      SiftMatchGPU sift_match_gpu;
      BOOST_CHECK(CreateSiftGPUMatcher(SiftMatchOptions(), &sift_match_gpu));
    }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  RunThreadWithOpenGLContext(&app, &thread);
}

BOOST_AUTO_TEST_CASE(TestCreateSiftGPUMatcherCUDA) {
#ifdef CUDA_ENABLED
  SiftMatchGPU sift_match_gpu;
  SiftMatchOptions match_options;
  match_options.gpu_index = 0;
  BOOST_CHECK(CreateSiftGPUMatcher(match_options, &sift_match_gpu));
#endif
}

BOOST_AUTO_TEST_CASE(TestMatchSiftFeaturesGPU) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  class TestThread : public Thread {
   private:
    void Run() {
      opengl_context_.MakeCurrent();
      SiftMatchGPU sift_match_gpu;
      BOOST_CHECK(CreateSiftGPUMatcher(SiftMatchOptions(), &sift_match_gpu));

      const FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(2);
      const FeatureDescriptors descriptors2 = descriptors1.colwise().reverse();

      FeatureMatches matches;

      MatchSiftFeaturesGPU(SiftMatchOptions(), &descriptors1, &descriptors2,
                           &sift_match_gpu, &matches);
      BOOST_CHECK_EQUAL(matches.size(), 2);
      BOOST_CHECK_EQUAL(matches[0].point2D_idx1, 0);
      BOOST_CHECK_EQUAL(matches[0].point2D_idx2, 1);
      BOOST_CHECK_EQUAL(matches[1].point2D_idx1, 1);
      BOOST_CHECK_EQUAL(matches[1].point2D_idx2, 0);

      MatchSiftFeaturesGPU(SiftMatchOptions(), nullptr, nullptr,
                           &sift_match_gpu, &matches);
      BOOST_CHECK_EQUAL(matches.size(), 2);
      BOOST_CHECK_EQUAL(matches[0].point2D_idx1, 0);
      BOOST_CHECK_EQUAL(matches[0].point2D_idx2, 1);
      BOOST_CHECK_EQUAL(matches[1].point2D_idx1, 1);
      BOOST_CHECK_EQUAL(matches[1].point2D_idx2, 0);

      MatchSiftFeaturesGPU(SiftMatchOptions(), &descriptors1, nullptr,
                           &sift_match_gpu, &matches);
      BOOST_CHECK_EQUAL(matches.size(), 2);
      BOOST_CHECK_EQUAL(matches[0].point2D_idx1, 0);
      BOOST_CHECK_EQUAL(matches[0].point2D_idx2, 1);
      BOOST_CHECK_EQUAL(matches[1].point2D_idx1, 1);
      BOOST_CHECK_EQUAL(matches[1].point2D_idx2, 0);

      MatchSiftFeaturesGPU(SiftMatchOptions(), nullptr, &descriptors2,
                           &sift_match_gpu, &matches);
      BOOST_CHECK_EQUAL(matches.size(), 2);
      BOOST_CHECK_EQUAL(matches[0].point2D_idx1, 0);
      BOOST_CHECK_EQUAL(matches[0].point2D_idx2, 1);
      BOOST_CHECK_EQUAL(matches[1].point2D_idx1, 1);
      BOOST_CHECK_EQUAL(matches[1].point2D_idx2, 0);
    }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  RunThreadWithOpenGLContext(&app, &thread);
}

BOOST_AUTO_TEST_CASE(TestMatchGuidedSiftFeaturesGPU) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  class TestThread : public Thread {
   private:
    void Run() {
      opengl_context_.MakeCurrent();
      SiftMatchGPU sift_match_gpu;
      BOOST_CHECK(CreateSiftGPUMatcher(SiftMatchOptions(), &sift_match_gpu));

      FeatureKeypoints keypoints1(2);
      keypoints1[0].x = 1;
      keypoints1[1].x = 2;
      FeatureKeypoints keypoints2(2);
      keypoints2[0].x = 2;
      keypoints2[1].x = 1;
      const FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(2);
      const FeatureDescriptors descriptors2 = descriptors1.colwise().reverse();

      TwoViewGeometry two_view_geometry;
      two_view_geometry.config = TwoViewGeometry::PLANAR_OR_PANORAMIC;
      two_view_geometry.H = Eigen::Matrix3d::Identity();

      MatchGuidedSiftFeaturesGPU(SiftMatchOptions(), &keypoints1, &keypoints2,
                                 &descriptors1, &descriptors2, &sift_match_gpu,
                                 &two_view_geometry);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 2);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx1, 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx2, 0);

      MatchGuidedSiftFeaturesGPU(SiftMatchOptions(), nullptr, nullptr, nullptr,
                                 nullptr, &sift_match_gpu, &two_view_geometry);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 2);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx1, 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx2, 0);

      MatchGuidedSiftFeaturesGPU(SiftMatchOptions(), &keypoints1, nullptr,
                                 &descriptors1, nullptr, &sift_match_gpu,
                                 &two_view_geometry);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 2);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx1, 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx2, 0);

      MatchGuidedSiftFeaturesGPU(SiftMatchOptions(), nullptr, &keypoints2,
                                 nullptr, &descriptors2, &sift_match_gpu,
                                 &two_view_geometry);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 2);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx1, 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx2, 0);

      keypoints1[0].x = 100;
      MatchGuidedSiftFeaturesGPU(SiftMatchOptions(), &keypoints1, &keypoints2,
                                 &descriptors1, &descriptors2, &sift_match_gpu,
                                 &two_view_geometry);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx1, 1);
      BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx2, 0);
    }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  RunThreadWithOpenGLContext(&app, &thread);
}

// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#define TEST_NAME "base/feature_extraction_test"
#include "util/testing.h"

#include <QApplication>

#include "base/feature_extraction.h"
#include "ext/SiftGPU/SiftGPU.h"

using namespace colmap;

void CreateImageWithSquare(const int size, Bitmap* bitmap) {
  bitmap->Allocate(size, size, false);
  bitmap->Fill(BitmapColor<uint8_t>(0, 0, 0));
  for (int r = size / 2 - size / 8; r < size / 2 + size / 8; ++r) {
    for (int c = size / 2 - size / 8; c < size / 2 + size / 8; ++c) {
      bitmap->SetPixel(r, c, BitmapColor<uint8_t>(255, 255, 255));
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
    BOOST_CHECK_GT(keypoints[i].scale, 0);
    BOOST_CHECK_GT(keypoints[i].orientation, 0);
    BOOST_CHECK_LT(keypoints[i].orientation, 2 * M_PI);
  }

  BOOST_CHECK_EQUAL(descriptors.rows(), 22);
  for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
    BOOST_CHECK_LT(std::abs(descriptors.row(i).cast<float>().norm() - 512), 1);
  }
}

BOOST_AUTO_TEST_CASE(TestExtractSiftFeaturesGPU) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  if (!OpenGLContextManager::HasMachineDisplay()) {
    return;
  }

  class TestThread : public Thread {
   private:
    void Run() {
      opengl_context_.MakeCurrent();

      Bitmap bitmap;
      CreateImageWithSquare(256, &bitmap);

      SiftGPU sift_gpu;
      BOOST_CHECK(
          CreateSiftGPUExtractor(SiftExtractionOptions(), -1, &sift_gpu));

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
        BOOST_CHECK_GT(keypoints[i].scale, 0);
        BOOST_CHECK_GT(keypoints[i].orientation, 0);
        BOOST_CHECK_LT(keypoints[i].orientation, 2 * M_PI);
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

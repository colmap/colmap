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

#define TEST_NAME "feature/utils"
#include "util/testing.h"

#include "feature/utils.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestFeatureKeypointsToPointsVector) {
  FeatureKeypoints keypoints(2);
  keypoints[1].x = 0.1;
  keypoints[1].y = 0.2;
  const std::vector<Eigen::Vector2d> points =
      FeatureKeypointsToPointsVector(keypoints);
  BOOST_CHECK_EQUAL(points[0], Eigen::Vector2d(0, 0));
  BOOST_CHECK_EQUAL(points[1].cast<float>(), Eigen::Vector2f(0.1, 0.2));
}

BOOST_AUTO_TEST_CASE(TestL2NormalizeFeatureDescriptors) {
  Eigen::MatrixXf descriptors = Eigen::MatrixXf::Random(100, 128);
  descriptors.array() += 1.0f;
  const Eigen::MatrixXf descriptors_normalized =
      L2NormalizeFeatureDescriptors(descriptors);
  for (Eigen::MatrixXf::Index r = 0; r < descriptors.rows(); ++r) {
    BOOST_CHECK(std::abs(descriptors_normalized.row(r).norm() - 1.0f) < 1e-6);
  }
}

BOOST_AUTO_TEST_CASE(TestL1RootNormalizeFeatureDescriptors) {
  Eigen::MatrixXf descriptors = Eigen::MatrixXf::Random(100, 128);
  descriptors.array() += 1.0f;
  const Eigen::MatrixXf descriptors_normalized =
      L1RootNormalizeFeatureDescriptors(descriptors);
  for (Eigen::MatrixXf::Index r = 0; r < descriptors.rows(); ++r) {
    BOOST_CHECK(std::abs(descriptors_normalized.row(r).norm() - 1.0f) < 1e-6);
  }
}

BOOST_AUTO_TEST_CASE(TestFeatureDescriptorsToUnsignedByte) {
  Eigen::MatrixXf descriptors = Eigen::MatrixXf::Random(100, 128);
  descriptors.array() += 1.0f;
  const FeatureDescriptors descriptors_uint8 =
      FeatureDescriptorsToUnsignedByte(descriptors);
  for (Eigen::MatrixXf::Index r = 0; r < descriptors.rows(); ++r) {
    for (Eigen::MatrixXf::Index c = 0; c < descriptors.cols(); ++c) {
      BOOST_CHECK_EQUAL(static_cast<uint8_t>(std::min(
                            255.0f, std::round(512.0f * descriptors(r, c)))),
                        descriptors_uint8(r, c));
    }
  }
}

BOOST_AUTO_TEST_CASE(TestExtractTopScaleFeatures) {
  FeatureKeypoints keypoints(5);
  keypoints[0].Rescale(3);
  keypoints[1].Rescale(4);
  keypoints[2].Rescale(1);
  keypoints[3].Rescale(5);
  keypoints[4].Rescale(2);
  const FeatureDescriptors descriptors = FeatureDescriptors::Random(5, 128);

  auto top_keypoints2 = keypoints;
  auto top_descriptors2 = descriptors;
  ExtractTopScaleFeatures(&top_keypoints2, &top_descriptors2, 2);
  BOOST_CHECK_EQUAL(top_keypoints2.size(), 2);
  BOOST_CHECK_EQUAL(top_keypoints2[0].ComputeScale(),
                    keypoints[3].ComputeScale());
  BOOST_CHECK_EQUAL(top_keypoints2[1].ComputeScale(),
                    keypoints[1].ComputeScale());
  BOOST_CHECK_EQUAL(top_descriptors2.rows(), 2);
  BOOST_CHECK_EQUAL(top_descriptors2.row(0), descriptors.row(3));
  BOOST_CHECK_EQUAL(top_descriptors2.row(1), descriptors.row(1));

  auto top_keypoints5 = keypoints;
  auto top_descriptors5 = descriptors;
  ExtractTopScaleFeatures(&top_keypoints5, &top_descriptors5, 5);
  BOOST_CHECK_EQUAL(top_keypoints5.size(), 5);
  BOOST_CHECK_EQUAL(top_descriptors5.rows(), 5);
  BOOST_CHECK_EQUAL(top_descriptors5, descriptors);

  auto top_keypoints6 = keypoints;
  auto top_descriptors6 = descriptors;
  ExtractTopScaleFeatures(&top_keypoints6, &top_descriptors6, 6);
  BOOST_CHECK_EQUAL(top_keypoints5.size(), 5);
  BOOST_CHECK_EQUAL(top_descriptors6.rows(), 5);
  BOOST_CHECK_EQUAL(top_descriptors6, descriptors);
}

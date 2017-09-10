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

// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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
#define BOOST_TEST_MODULE "base/feature"
#include <boost/test/unit_test.hpp>

#include "base/feature.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestFeatureKeypointsToPointsVector) {
  FeatureKeypoints keypoints(2);
  keypoints[1].x = 0.1;
  keypoints[1].y = 0.2;
  keypoints[1].scale = 0.3;
  keypoints[1].orientation = 0.4;
  const std::vector<Eigen::Vector2d> points =
      FeatureKeypointsToPointsVector(keypoints);
  BOOST_CHECK_EQUAL(points[0], Eigen::Vector2d(0, 0));
  BOOST_CHECK_EQUAL(points[1].cast<float>(), Eigen::Vector2f(0.1, 0.2));
}

BOOST_AUTO_TEST_CASE(TestFeatureKeypoints) {
  FeatureKeypoint keypoint;
  BOOST_CHECK_EQUAL(keypoint.x, 0.0f);
  BOOST_CHECK_EQUAL(keypoint.y, 0.0f);
  BOOST_CHECK_EQUAL(keypoint.scale, 0.0f);
  BOOST_CHECK_EQUAL(keypoint.orientation, 0.0f);
  FeatureKeypoints keypoints(1);
  BOOST_CHECK_EQUAL(keypoints.size(), 1);
  BOOST_CHECK_EQUAL(keypoints[0].x, 0.0f);
  BOOST_CHECK_EQUAL(keypoints[0].y, 0.0f);
  BOOST_CHECK_EQUAL(keypoints[0].scale, 0.0f);
  BOOST_CHECK_EQUAL(keypoints[0].orientation, 0.0f);
}

BOOST_AUTO_TEST_CASE(TestFeatureDescriptors) {
  FeatureDescriptors descriptors = FeatureDescriptors::Random(2, 3);
  BOOST_CHECK_EQUAL(descriptors.rows(), 2);
  BOOST_CHECK_EQUAL(descriptors.cols(), 3);
  BOOST_CHECK_EQUAL(descriptors(0, 0), descriptors.data()[0]);
  BOOST_CHECK_EQUAL(descriptors(0, 1), descriptors.data()[1]);
  BOOST_CHECK_EQUAL(descriptors(0, 2), descriptors.data()[2]);
  BOOST_CHECK_EQUAL(descriptors(1, 0), descriptors.data()[3]);
  BOOST_CHECK_EQUAL(descriptors(1, 1), descriptors.data()[4]);
  BOOST_CHECK_EQUAL(descriptors(1, 2), descriptors.data()[5]);
}

BOOST_AUTO_TEST_CASE(TestFeatureMatches) {
  FeatureMatch match;
  BOOST_CHECK_EQUAL(match.point2D_idx1, kInvalidPoint2DIdx);
  BOOST_CHECK_EQUAL(match.point2D_idx2, kInvalidPoint2DIdx);
  FeatureMatches matches(1);
  BOOST_CHECK_EQUAL(matches.size(), 1);
  BOOST_CHECK_EQUAL(matches[0].point2D_idx1, kInvalidPoint2DIdx);
  BOOST_CHECK_EQUAL(matches[0].point2D_idx2, kInvalidPoint2DIdx);
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

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

#define TEST_NAME "feature/types"
#include "util/testing.h"

#include "feature/types.h"
#include "util/math.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestFeatureKeypoints) {
  FeatureKeypoint keypoint;
  BOOST_CHECK_EQUAL(keypoint.x, 0.0f);
  BOOST_CHECK_EQUAL(keypoint.y, 0.0f);
  BOOST_CHECK_EQUAL(keypoint.a11, 1.0f);
  BOOST_CHECK_EQUAL(keypoint.a12, 0.0f);
  BOOST_CHECK_EQUAL(keypoint.a21, 0.0f);
  BOOST_CHECK_EQUAL(keypoint.a22, 1.0f);

  FeatureKeypoints keypoints(1);
  BOOST_CHECK_EQUAL(keypoints.size(), 1);
  BOOST_CHECK_EQUAL(keypoints[0].x, 0.0f);
  BOOST_CHECK_EQUAL(keypoints[0].y, 0.0f);
  BOOST_CHECK_EQUAL(keypoints[0].a11, 1.0f);
  BOOST_CHECK_EQUAL(keypoints[0].a12, 0.0f);
  BOOST_CHECK_EQUAL(keypoints[0].a21, 0.0f);
  BOOST_CHECK_EQUAL(keypoints[0].a22, 1.0f);

  keypoint = FeatureKeypoint(1, 2);
  BOOST_CHECK_EQUAL(keypoint.x, 1.0f);
  BOOST_CHECK_EQUAL(keypoint.y, 2.0f);
  BOOST_CHECK_EQUAL(keypoint.a11, 1.0f);
  BOOST_CHECK_EQUAL(keypoint.a12, 0.0f);
  BOOST_CHECK_EQUAL(keypoint.a21, 0.0f);
  BOOST_CHECK_EQUAL(keypoint.a22, 1.0f);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScale() - 1.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleX() - 1.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleY() - 1.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeOrientation() - 0.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeShear() - 0.0f), 1e-6);

  keypoint = FeatureKeypoint(1, 2, 0, 0);
  BOOST_CHECK_EQUAL(keypoint.x, 1.0f);
  BOOST_CHECK_EQUAL(keypoint.y, 2.0f);
  BOOST_CHECK_EQUAL(keypoint.a11, 0.0f);
  BOOST_CHECK_EQUAL(keypoint.a12, 0.0f);
  BOOST_CHECK_EQUAL(keypoint.a21, 0.0f);
  BOOST_CHECK_EQUAL(keypoint.a22, 0.0f);

  keypoint = FeatureKeypoint(1, 2, 1, 0);
  BOOST_CHECK_EQUAL(keypoint.x, 1.0f);
  BOOST_CHECK_EQUAL(keypoint.y, 2.0f);
  BOOST_CHECK_EQUAL(keypoint.a11, 1.0f);
  BOOST_CHECK_EQUAL(keypoint.a12, 0.0f);
  BOOST_CHECK_EQUAL(keypoint.a21, 0.0f);
  BOOST_CHECK_EQUAL(keypoint.a22, 1.0f);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScale() - 1.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleX() - 1.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleY() - 1.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeOrientation() - 0.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeShear() - 0.0f), 1e-6);

  keypoint = FeatureKeypoint(1, 2, 1, M_PI / 2);
  BOOST_CHECK_EQUAL(keypoint.x, 1.0f);
  BOOST_CHECK_EQUAL(keypoint.y, 2.0f);
  BOOST_CHECK_LT(std::abs(keypoint.a11 - 0.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a12 - -1.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a21 - 1.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a22 - 0.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScale() - 1.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleX() - 1.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleY() - 1.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeOrientation() - M_PI / 2), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeShear() - 0.0f), 1e-6);

  keypoint = FeatureKeypoint(1, 2, 2, M_PI / 2);
  BOOST_CHECK_EQUAL(keypoint.x, 1.0f);
  BOOST_CHECK_EQUAL(keypoint.y, 2.0f);
  BOOST_CHECK_LT(std::abs(keypoint.a11 - 0.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a12 - -2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a21 - 2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a22 - 0.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScale() - 2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleX() - 2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleY() - 2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeOrientation() - M_PI / 2), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeShear() - 0.0f), 1e-6);

  keypoint = FeatureKeypoint(1, 2, 2, M_PI);
  BOOST_CHECK_EQUAL(keypoint.x, 1.0f);
  BOOST_CHECK_EQUAL(keypoint.y, 2.0f);
  BOOST_CHECK_LT(std::abs(keypoint.a11 - -2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a12 - 0.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a21 - 0.0), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a22 - -2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScale() - 2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleX() - 2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleY() - 2.0f), 1e-6);
  BOOST_CHECK(std::abs(keypoint.ComputeOrientation() - M_PI) < 1e-6 ||
              std::abs(keypoint.ComputeOrientation() + M_PI) < 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeShear() - 0.0f), 1e-6);

  keypoint = FeatureKeypoint::FromParameters(1, 2, 2, 2, M_PI, 0);
  BOOST_CHECK_EQUAL(keypoint.x, 1.0f);
  BOOST_CHECK_EQUAL(keypoint.y, 2.0f);
  BOOST_CHECK_LT(std::abs(keypoint.a11 - -2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a12 - 0.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a21 - 0.0), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a22 - -2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScale() - 2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleX() - 2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleY() - 2.0f), 1e-6);
  BOOST_CHECK(std::abs(keypoint.ComputeOrientation() - M_PI) < 1e-6 ||
              std::abs(keypoint.ComputeOrientation() + M_PI) < 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeShear() - 0.0f), 1e-6);

  keypoint = FeatureKeypoint::FromParameters(1, 2, 2, 3, M_PI, 0);
  BOOST_CHECK_EQUAL(keypoint.x, 1.0f);
  BOOST_CHECK_EQUAL(keypoint.y, 2.0f);
  BOOST_CHECK_LT(std::abs(keypoint.a11 - -2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a12 - 0.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a21 - 0.0), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a22 - -3.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScale() - 2.5f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleX() - 2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleY() - 3.0f), 1e-6);
  BOOST_CHECK(std::abs(keypoint.ComputeOrientation() - M_PI) < 1e-6 ||
              std::abs(keypoint.ComputeOrientation() + M_PI) < 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeShear() - 0.0f), 1e-6);

  keypoint = FeatureKeypoint::FromParameters(1, 2, 2, 3, -M_PI / 2, M_PI / 4);
  BOOST_CHECK_EQUAL(keypoint.x, 1.0f);
  BOOST_CHECK_EQUAL(keypoint.y, 2.0f);
  BOOST_CHECK_LT(std::abs(keypoint.a11 - 0.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a12 - 2.12132025f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a21 - -2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a22 - 2.12132025f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScale() - 2.5f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleX() - 2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleY() - 3.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeOrientation() - -M_PI / 2), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeShear() - M_PI / 4), 1e-6);

  keypoint = FeatureKeypoint::FromParameters(1, 2, 2, 3, M_PI / 2, M_PI / 4);
  BOOST_CHECK_EQUAL(keypoint.x, 1.0f);
  BOOST_CHECK_EQUAL(keypoint.y, 2.0f);
  BOOST_CHECK_LT(std::abs(keypoint.a11 - 0.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a12 - -2.12132025f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a21 - 2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a22 - -2.12132025f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScale() - 2.5f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleX() - 2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleY() - 3.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeOrientation() - M_PI / 2), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeShear() - M_PI / 4), 1e-6);

  keypoint.Rescale(2, 2);
  BOOST_CHECK_EQUAL(keypoint.x, 2.0f);
  BOOST_CHECK_EQUAL(keypoint.y, 4.0f);
  BOOST_CHECK_LT(std::abs(keypoint.a11 - 2 * 0.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a12 - 2 * -2.12132025f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a21 - 2 * 2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a22 - 2 * -2.12132025f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScale() - 2 * 2.5f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleX() - 2 * 2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleY() - 2 * 3.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeOrientation() - M_PI / 2), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeShear() - M_PI / 4), 1e-6);

  keypoint.Rescale(1, 0.5);
  BOOST_CHECK_EQUAL(keypoint.x, 2.0f);
  BOOST_CHECK_EQUAL(keypoint.y, 2.0f);
  BOOST_CHECK_LT(std::abs(keypoint.a11 - 2 * 0.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a12 - -2.12132025f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a21 - 2 * 2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.a22 - -2.12132025f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScale() - 3.5f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleX() - 2 * 2.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeScaleY() - 3.0f), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeOrientation() - M_PI / 2), 1e-6);
  BOOST_CHECK_LT(std::abs(keypoint.ComputeShear() - M_PI / 4), 1e-6);
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

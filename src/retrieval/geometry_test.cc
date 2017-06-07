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

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "retrieval/geometry"
#include <boost/test/unit_test.hpp>

#include <Eigen/Dense>
#include <iostream>

#include "retrieval/geometry.h"

using namespace colmap::retrieval;

BOOST_AUTO_TEST_CASE(TestIdentity) {
  for (float x = 0; x < 3; ++x) {
    for (float y = 0; y < 3; ++y) {
      for (float scale = 1; scale < 5; ++scale) {
        for (float orientation = 0; orientation < 3; ++orientation) {
          FeatureGeometry feature1;
          feature1.x = x;
          feature1.y = y;
          feature1.scale = scale;
          feature1.orientation = orientation;
          FeatureGeometry feature2;
          feature2.x = x;
          feature2.y = y;
          feature2.scale = scale;
          feature2.orientation = orientation;
          const auto tform_matrix =
              FeatureGeometry::TransformMatrixFromMatch(feature1, feature2);
          BOOST_CHECK(
              tform_matrix.isApprox(Eigen::Matrix<float, 2, 3>::Identity()));
          const auto tform =
              FeatureGeometry::TransformFromMatch(feature1, feature2);
          BOOST_CHECK_CLOSE(tform.scale, 1, 1e-6);
          BOOST_CHECK_CLOSE(tform.angle, 0, 1e-6);
          BOOST_CHECK_CLOSE(tform.tx, 0, 1e-6);
          BOOST_CHECK_CLOSE(tform.ty, 0, 1e-6);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(TestTranslation) {
  for (float x = 0; x < 3; ++x) {
    for (float y = 0; y < 3; ++y) {
      FeatureGeometry feature1;
      feature1.scale = 1;
      FeatureGeometry feature2;
      feature2.x = x;
      feature2.y = y;
      feature2.scale = 1;
      feature2.orientation = 0;
      const auto tform_matrix =
          FeatureGeometry::TransformMatrixFromMatch(feature1, feature2);
      BOOST_CHECK(
          tform_matrix.leftCols<2>().isApprox(Eigen::Matrix2f::Identity()));
      BOOST_CHECK(tform_matrix.rightCols<1>().isApprox(Eigen::Vector2f(x, y)));
      const auto tform =
          FeatureGeometry::TransformFromMatch(feature1, feature2);
      BOOST_CHECK_CLOSE(tform.scale, 1, 1e-6);
      BOOST_CHECK_CLOSE(tform.angle, 0, 1e-6);
      BOOST_CHECK_CLOSE(tform.tx, x, 1e-6);
      BOOST_CHECK_CLOSE(tform.ty, y, 1e-6);
    }
  }
}

BOOST_AUTO_TEST_CASE(TestScale) {
  for (float scale = 1; scale < 5; ++scale) {
    FeatureGeometry feature1;
    feature1.scale = 1;
    FeatureGeometry feature2;
    feature2.scale = scale;
    feature2.orientation = 0;
    const auto tform_matrix =
        FeatureGeometry::TransformMatrixFromMatch(feature1, feature2);
    BOOST_CHECK(tform_matrix.leftCols<2>().isApprox(
        scale * Eigen::Matrix2f::Identity()));
    BOOST_CHECK(tform_matrix.rightCols<1>().isApprox(Eigen::Vector2f(0, 0)));
    const auto tform = FeatureGeometry::TransformFromMatch(feature1, feature2);
    BOOST_CHECK_CLOSE(tform.scale, scale, 1e-6);
    BOOST_CHECK_CLOSE(tform.angle, 0, 1e-6);
    BOOST_CHECK_CLOSE(tform.tx, 0, 1e-6);
    BOOST_CHECK_CLOSE(tform.ty, 0, 1e-6);
  }
}

BOOST_AUTO_TEST_CASE(TestOrientation) {
  for (float orientation = 0; orientation < 3; ++orientation) {
    FeatureGeometry feature1;
    feature1.scale = 1;
    feature1.orientation = 0;
    FeatureGeometry feature2;
    feature2.scale = 1;
    feature2.orientation = orientation;
    const auto tform_matrix =
        FeatureGeometry::TransformMatrixFromMatch(feature1, feature2);
    BOOST_CHECK_CLOSE(tform_matrix.leftCols<2>().determinant(), 1, 1e-5);
    BOOST_CHECK(tform_matrix.rightCols<1>().isApprox(Eigen::Vector2f(0, 0)));
    const auto tform = FeatureGeometry::TransformFromMatch(feature1, feature2);
    BOOST_CHECK_CLOSE(tform.scale, 1, 1e-6);
    BOOST_CHECK_CLOSE(tform.angle, orientation, 1e-6);
    BOOST_CHECK_CLOSE(tform.tx, 0, 1e-6);
    BOOST_CHECK_CLOSE(tform.ty, 0, 1e-6);
  }
}

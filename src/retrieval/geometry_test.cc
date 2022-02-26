// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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

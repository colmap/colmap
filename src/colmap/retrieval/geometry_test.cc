// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "retrieval/geometry"
#include "colmap/retrieval/geometry.h"

#include "colmap/util/eigen_alignment.h"

#include <iostream>

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace colmap {
namespace retrieval {
namespace {

TEST(FeatureGeometry, Identity) {
  for (int x = 0; x < 3; ++x) {
    for (int y = 0; y < 3; ++y) {
      for (int scale = 1; scale < 5; ++scale) {
        for (int orientation = 0; orientation < 3; ++orientation) {
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
          EXPECT_TRUE(
              tform_matrix.isApprox(Eigen::Matrix<float, 2, 3>::Identity()));
          const auto tform =
              FeatureGeometry::TransformFromMatch(feature1, feature2);
          EXPECT_NEAR(tform.scale, 1, 1e-6);
          EXPECT_NEAR(tform.angle, 0, 1e-6);
          EXPECT_NEAR(tform.tx, 0, 1e-6);
          EXPECT_NEAR(tform.ty, 0, 1e-6);
        }
      }
    }
  }
}

TEST(FeatureGeometry, Translation) {
  for (int x = 0; x < 3; ++x) {
    for (int y = 0; y < 3; ++y) {
      FeatureGeometry feature1;
      feature1.scale = 1;
      FeatureGeometry feature2;
      feature2.x = x;
      feature2.y = y;
      feature2.scale = 1;
      feature2.orientation = 0;
      const auto tform_matrix =
          FeatureGeometry::TransformMatrixFromMatch(feature1, feature2);
      EXPECT_TRUE(
          tform_matrix.leftCols<2>().isApprox(Eigen::Matrix2f::Identity()));
      EXPECT_TRUE(tform_matrix.rightCols<1>().isApprox(Eigen::Vector2f(x, y)));
      const auto tform =
          FeatureGeometry::TransformFromMatch(feature1, feature2);
      EXPECT_NEAR(tform.scale, 1, 1e-6);
      EXPECT_NEAR(tform.angle, 0, 1e-6);
      EXPECT_NEAR(tform.tx, x, 1e-6);
      EXPECT_NEAR(tform.ty, y, 1e-6);
    }
  }
}

TEST(FeatureGeometry, Scale) {
  for (int scale = 1; scale < 5; ++scale) {
    FeatureGeometry feature1;
    feature1.scale = 1;
    FeatureGeometry feature2;
    feature2.scale = scale;
    feature2.orientation = 0;
    const auto tform_matrix =
        FeatureGeometry::TransformMatrixFromMatch(feature1, feature2);
    EXPECT_TRUE(tform_matrix.leftCols<2>().isApprox(
        scale * Eigen::Matrix2f::Identity()));
    EXPECT_TRUE(tform_matrix.rightCols<1>().isApprox(Eigen::Vector2f(0, 0)));
    const auto tform = FeatureGeometry::TransformFromMatch(feature1, feature2);
    EXPECT_NEAR(tform.scale, scale, 1e-6);
    EXPECT_NEAR(tform.angle, 0, 1e-6);
    EXPECT_NEAR(tform.tx, 0, 1e-6);
    EXPECT_NEAR(tform.ty, 0, 1e-6);
  }
}

TEST(FeatureGeometry, Orientation) {
  for (int orientation = 0; orientation < 3; ++orientation) {
    FeatureGeometry feature1;
    feature1.scale = 1;
    feature1.orientation = 0;
    FeatureGeometry feature2;
    feature2.scale = 1;
    feature2.orientation = orientation;
    const auto tform_matrix =
        FeatureGeometry::TransformMatrixFromMatch(feature1, feature2);
    EXPECT_NEAR(tform_matrix.leftCols<2>().determinant(), 1, 1e-5);
    EXPECT_TRUE(tform_matrix.rightCols<1>().isApprox(Eigen::Vector2f(0, 0)));
    const auto tform = FeatureGeometry::TransformFromMatch(feature1, feature2);
    EXPECT_NEAR(tform.scale, 1, 1e-6);
    EXPECT_NEAR(tform.angle, orientation, 1e-6);
    EXPECT_NEAR(tform.tx, 0, 1e-6);
    EXPECT_NEAR(tform.ty, 0, 1e-6);
  }
}

}  // namespace
}  // namespace retrieval
}  // namespace colmap

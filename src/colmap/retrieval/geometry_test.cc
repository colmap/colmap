// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/retrieval/geometry.h"

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/eigen_matchers.h"

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
          EXPECT_THAT(tform_matrix,
                      EigenMatrixNear(Eigen::Matrix<float, 2, 3>(
                          Eigen::Matrix<float, 2, 3>::Identity())));
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
      EXPECT_THAT(
          tform_matrix.leftCols<2>(),
          EigenMatrixNear<Eigen::Matrix2f>(Eigen::Matrix2f::Identity()));
      EXPECT_THAT(tform_matrix.rightCols<1>(),
                  EigenMatrixNear(Eigen::Vector2f(x, y)));
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
    EXPECT_THAT(
        tform_matrix.leftCols<2>(),
        EigenMatrixNear<Eigen::Matrix2f>(scale * Eigen::Matrix2f::Identity()));
    EXPECT_THAT(tform_matrix.rightCols<1>(),
                EigenMatrixNear(Eigen::Vector2f(0, 0)));
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
    EXPECT_THAT(tform_matrix.rightCols<1>(),
                EigenMatrixNear(Eigen::Vector2f(0, 0)));
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

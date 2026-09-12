// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/two_view_geometry.h"

#include "colmap/util/eigen_matchers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(TwoViewGeometry, Default) {
  TwoViewGeometry two_view_geometry;
  EXPECT_EQ(two_view_geometry.config, TwoViewGeometry::UNDEFINED);
  EXPECT_FALSE(two_view_geometry.F.has_value());
  EXPECT_FALSE(two_view_geometry.E.has_value());
  EXPECT_FALSE(two_view_geometry.H.has_value());
  EXPECT_FALSE(two_view_geometry.cam2_from_cam1.has_value());
  EXPECT_TRUE(two_view_geometry.inlier_matches.empty());
}

TEST(TwoViewGeometry, Invert) {
  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  two_view_geometry.F = Eigen::Matrix3d::Identity();
  two_view_geometry.E = Eigen::Matrix3d::Identity();
  two_view_geometry.H = Eigen::Matrix3d::Identity();
  two_view_geometry.cam2_from_cam1 =
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(0, 1, 2));
  two_view_geometry.inlier_matches.resize(2);
  two_view_geometry.inlier_matches[0] = FeatureMatch(0, 1);
  two_view_geometry.inlier_matches[1] = FeatureMatch(2, 3);

  two_view_geometry.Invert();
  EXPECT_EQ(two_view_geometry.config, TwoViewGeometry::CALIBRATED);
  EXPECT_THAT(two_view_geometry.F.value(),
              EigenMatrixNear<Eigen::Matrix3d>(Eigen::Matrix3d::Identity()));
  EXPECT_THAT(two_view_geometry.E.value(),
              EigenMatrixNear<Eigen::Matrix3d>(Eigen::Matrix3d::Identity()));
  EXPECT_THAT(two_view_geometry.H.value(),
              EigenMatrixNear<Eigen::Matrix3d>(Eigen::Matrix3d::Identity()));
  EXPECT_TRUE(two_view_geometry.cam2_from_cam1.has_value());
  EXPECT_THAT(two_view_geometry.cam2_from_cam1->rotation().coeffs(),
              EigenMatrixNear(Eigen::Quaterniond::Identity().coeffs()));
  EXPECT_THAT(two_view_geometry.cam2_from_cam1->translation(),
              EigenMatrixNear(Eigen::Vector3d(-0, -1, -2)));
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 0);
  EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx1, 3);
  EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx2, 2);

  two_view_geometry.Invert();
  EXPECT_EQ(two_view_geometry.config, TwoViewGeometry::CALIBRATED);
  EXPECT_THAT(two_view_geometry.F.value(),
              EigenMatrixNear<Eigen::Matrix3d>(Eigen::Matrix3d::Identity()));
  EXPECT_THAT(two_view_geometry.E.value(),
              EigenMatrixNear<Eigen::Matrix3d>(Eigen::Matrix3d::Identity()));
  EXPECT_THAT(two_view_geometry.H.value(),
              EigenMatrixNear<Eigen::Matrix3d>(Eigen::Matrix3d::Identity()));
  EXPECT_TRUE(two_view_geometry.cam2_from_cam1.has_value());
  EXPECT_THAT(two_view_geometry.cam2_from_cam1->rotation().coeffs(),
              EigenMatrixNear(Eigen::Quaterniond::Identity().coeffs()));
  EXPECT_THAT(two_view_geometry.cam2_from_cam1->translation(),
              EigenMatrixNear(Eigen::Vector3d(0, 1, 2)));
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx1, 2);
  EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx2, 3);
}

}  // namespace
}  // namespace colmap

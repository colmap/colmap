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

#include "colmap/scene/two_view_geometry.h"

#include "colmap/geometry/pose.h"
#include "colmap/util/eigen_matchers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(TwoViewGeometry, Default) {
  TwoViewGeometry two_view_geometry;
  EXPECT_EQ(two_view_geometry.config, TwoViewGeometry::UNDEFINED);
  EXPECT_EQ(two_view_geometry.F, Eigen::Matrix3d::Zero());
  EXPECT_EQ(two_view_geometry.E, Eigen::Matrix3d::Zero());
  EXPECT_EQ(two_view_geometry.H, Eigen::Matrix3d::Zero());
  EXPECT_EQ(two_view_geometry.cam2_from_cam1.rotation.coeffs(),
            Eigen::Quaterniond::Identity().coeffs());
  EXPECT_EQ(two_view_geometry.cam2_from_cam1.translation,
            Eigen::Vector3d::Zero());
  EXPECT_TRUE(two_view_geometry.inlier_matches.empty());
}

TEST(TwoViewGeometry, Invert) {
  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  two_view_geometry.F = two_view_geometry.E = two_view_geometry.H =
      Eigen::Matrix3d::Identity();
  two_view_geometry.cam2_from_cam1 =
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(0, 1, 2));
  two_view_geometry.inlier_matches.resize(2);
  two_view_geometry.inlier_matches[0] = FeatureMatch(0, 1);
  two_view_geometry.inlier_matches[1] = FeatureMatch(2, 3);

  two_view_geometry.Invert();
  EXPECT_EQ(two_view_geometry.config, TwoViewGeometry::CALIBRATED);
  EXPECT_THAT(two_view_geometry.F,
              EigenMatrixNear<Eigen::Matrix3d>(Eigen::Matrix3d::Identity()));
  EXPECT_THAT(two_view_geometry.E,
              EigenMatrixNear<Eigen::Matrix3d>(Eigen::Matrix3d::Identity()));
  EXPECT_THAT(two_view_geometry.H,
              EigenMatrixNear<Eigen::Matrix3d>(Eigen::Matrix3d::Identity()));
  EXPECT_THAT(two_view_geometry.cam2_from_cam1.rotation.coeffs(),
              EigenMatrixNear(Eigen::Quaterniond::Identity().coeffs()));
  EXPECT_THAT(two_view_geometry.cam2_from_cam1.translation,
              EigenMatrixNear(Eigen::Vector3d(-0, -1, -2)));
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 0);
  EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx1, 3);
  EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx2, 2);

  two_view_geometry.Invert();
  EXPECT_EQ(two_view_geometry.config, TwoViewGeometry::CALIBRATED);
  EXPECT_THAT(two_view_geometry.F,
              EigenMatrixNear<Eigen::Matrix3d>(Eigen::Matrix3d::Identity()));
  EXPECT_THAT(two_view_geometry.E,
              EigenMatrixNear<Eigen::Matrix3d>(Eigen::Matrix3d::Identity()));
  EXPECT_THAT(two_view_geometry.H,
              EigenMatrixNear<Eigen::Matrix3d>(Eigen::Matrix3d::Identity()));
  EXPECT_THAT(two_view_geometry.cam2_from_cam1.rotation.coeffs(),
              EigenMatrixNear(Eigen::Quaterniond::Identity().coeffs()));
  EXPECT_THAT(two_view_geometry.cam2_from_cam1.translation,
              EigenMatrixNear(Eigen::Vector3d(0, 1, 2)));
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx1, 2);
  EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx2, 3);
}

}  // namespace
}  // namespace colmap

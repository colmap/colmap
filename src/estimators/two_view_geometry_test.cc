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

#define TEST_NAME "estimators/two_view_geometry"
#include "util/testing.h"

#include "base/pose.h"
#include "estimators/two_view_geometry.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestDefault) {
  TwoViewGeometry two_view_geometry;
  BOOST_CHECK_EQUAL(two_view_geometry.config, TwoViewGeometry::UNDEFINED);
  BOOST_CHECK_EQUAL(two_view_geometry.F, Eigen::Matrix3d::Zero());
  BOOST_CHECK_EQUAL(two_view_geometry.E, Eigen::Matrix3d::Zero());
  BOOST_CHECK_EQUAL(two_view_geometry.H, Eigen::Matrix3d::Zero());
  BOOST_CHECK_EQUAL(two_view_geometry.qvec, Eigen::Vector4d::Zero());
  BOOST_CHECK_EQUAL(two_view_geometry.tvec, Eigen::Vector3d::Zero());
  BOOST_CHECK(two_view_geometry.inlier_matches.empty());
}

BOOST_AUTO_TEST_CASE(TestInvert) {
  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  two_view_geometry.F = two_view_geometry.E = two_view_geometry.H =
      Eigen::Matrix3d::Identity();
  two_view_geometry.qvec = ComposeIdentityQuaternion();
  two_view_geometry.tvec = Eigen::Vector3d(0, 1, 2);
  two_view_geometry.inlier_matches.resize(2);
  two_view_geometry.inlier_matches[0] = FeatureMatch(0, 1);
  two_view_geometry.inlier_matches[1] = FeatureMatch(2, 3);

  two_view_geometry.Invert();
  BOOST_CHECK_EQUAL(two_view_geometry.config, TwoViewGeometry::CALIBRATED);
  BOOST_CHECK(two_view_geometry.F.isApprox(Eigen::Matrix3d::Identity()));
  BOOST_CHECK(two_view_geometry.E.isApprox(Eigen::Matrix3d::Identity()));
  BOOST_CHECK(two_view_geometry.H.isApprox(Eigen::Matrix3d::Identity()));
  BOOST_CHECK(two_view_geometry.qvec.isApprox(ComposeIdentityQuaternion()));
  BOOST_CHECK(two_view_geometry.tvec.isApprox(Eigen::Vector3d(-0, -1, -2)));
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx1, 1);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx2, 0);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx1, 3);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx2, 2);

  two_view_geometry.Invert();
  BOOST_CHECK_EQUAL(two_view_geometry.config, TwoViewGeometry::CALIBRATED);
  BOOST_CHECK(two_view_geometry.F.isApprox(Eigen::Matrix3d::Identity()));
  BOOST_CHECK(two_view_geometry.E.isApprox(Eigen::Matrix3d::Identity()));
  BOOST_CHECK(two_view_geometry.H.isApprox(Eigen::Matrix3d::Identity()));
  BOOST_CHECK(two_view_geometry.qvec.isApprox(ComposeIdentityQuaternion()));
  BOOST_CHECK(two_view_geometry.tvec.isApprox(Eigen::Vector3d(0, 1, 2)));
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx1, 2);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx2, 3);
}

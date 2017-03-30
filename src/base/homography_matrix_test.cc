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

#define TEST_NAME "base/homography_matrix"
#include "util/testing.h"

#include <Eigen/Geometry>

#include "base/homography_matrix.h"

using namespace colmap;

// Note that the test case values are obtained from OpenCV.
BOOST_AUTO_TEST_CASE(TestDecomposeHomographyMatrix) {
  Eigen::Matrix3d H;
  H << 2.649157564634028, 4.583875997496426, 70.694447785121326,
      -1.072756858861583, 3.533262150437228, 1513.656999614321649,
      0.001303887589576, 0.003042206876298, 1;
  H *= 3;

  Eigen::Matrix3d K;
  K << 640, 0, 320, 0, 640, 240, 0, 0, 1;

  std::vector<Eigen::Matrix3d> R;
  std::vector<Eigen::Vector3d> t;
  std::vector<Eigen::Vector3d> n;
  DecomposeHomographyMatrix(H, K, K, &R, &t, &n);

  BOOST_CHECK_EQUAL(R.size(), 4);
  BOOST_CHECK_EQUAL(t.size(), 4);
  BOOST_CHECK_EQUAL(n.size(), 4);

  Eigen::Matrix3d R_ref;
  R_ref << 0.43307983549125, 0.545749113549648, -0.717356090899523,
      -0.85630229674426, 0.497582023798831, -0.138414255706431,
      0.281404038139784, 0.67421809131173, 0.682818960388909;
  const Eigen::Vector3d t_ref(1.826751712278038, 1.264718492450820,
                              0.195080809998819);
  const Eigen::Vector3d n_ref(-0.244875830334816, -0.480857890778889,
                              -0.841909446789566);

  bool ref_solution_exists = false;
  for (size_t i = 0; i < 4; ++i) {
    const double kEps = 1e-6;
    if ((R[i] - R_ref).norm() < kEps && (t[i] - t_ref).norm() < kEps &&
        (n[i] - n_ref).norm() < kEps) {
      ref_solution_exists = true;
    }
  }
  BOOST_CHECK(ref_solution_exists);
}

BOOST_AUTO_TEST_CASE(TestPoseFromHomographyMatrix) {
  const Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d K2 = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d R_ref = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t_ref(1, 0, 0);
  const Eigen::Vector3d n_ref(-1, 0, 0);
  const double d_ref = 1;
  const Eigen::Matrix3d H =
      HomographyMatrixFromPose(K1, K2, R_ref, t_ref, n_ref, d_ref);

  std::vector<Eigen::Vector2d> points1;
  points1.emplace_back(0.1, 0.4);
  points1.emplace_back(0.2, 0.3);
  points1.emplace_back(0.3, 0.2);
  points1.emplace_back(0.4, 0.1);

  std::vector<Eigen::Vector2d> points2;
  for (const auto& point1 : points1) {
    const Eigen::Vector3d point2 = H * point1.homogeneous();
    points2.push_back(point2.hnormalized());
  }

  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  Eigen::Vector3d n;
  std::vector<Eigen::Vector3d> points3D;
  PoseFromHomographyMatrix(H, K1, K2, points1, points2, &R, &t, &n, &points3D);

  BOOST_CHECK_EQUAL(R, R_ref);
  BOOST_CHECK_EQUAL(t, t_ref);
  BOOST_CHECK_EQUAL(n, n_ref);
  BOOST_CHECK_EQUAL(points3D.size(), points1.size());
}

BOOST_AUTO_TEST_CASE(TestHomographyMatrixFromPosePureRotation) {
  const Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d K2 = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t(0, 0, 0);
  const Eigen::Vector3d n(-1, 0, 0);
  const double d = 1;
  const Eigen::Matrix3d H = HomographyMatrixFromPose(K1, K2, R, t, n, d);
  BOOST_CHECK_EQUAL(H, Eigen::Matrix3d::Identity());
}

BOOST_AUTO_TEST_CASE(TestHomographyMatrixFromPosePlanarScene) {
  const Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d K2 = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t(1, 0, 0);
  const Eigen::Vector3d n(-1, 0, 0);
  const double d = 1;
  const Eigen::Matrix3d H = HomographyMatrixFromPose(K1, K2, R, t, n, d);
  Eigen::Matrix3d H_ref;
  H_ref << 2, 0, 0, 0, 1, 0, 0, 0, 1;
  BOOST_CHECK_EQUAL(H, H_ref);
}

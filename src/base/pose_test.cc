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
#define BOOST_TEST_MODULE "base/pose"
#include <boost/test/unit_test.hpp>

#include <Eigen/Core>

#include "base/pose.h"
#include "base/projection.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestEulerAnglesX) {
  const double rx = 0.3;
  const double ry = 0;
  const double rz = 0;
  double rxx, ryy, rzz;

  RotationMatrixToEulerAngles(EulerAnglesToRotationMatrix(rx, ry, rz), &rxx,
                              &ryy, &rzz);

  BOOST_CHECK_CLOSE(rx, rxx, 1e-6);
  BOOST_CHECK_CLOSE(ry, ryy, 1e-6);
  BOOST_CHECK_CLOSE(rz, rzz, 1e-6);
}

BOOST_AUTO_TEST_CASE(TestEulerAnglesY) {
  const double rx = 0;
  const double ry = 0.3;
  const double rz = 0;
  double rxx, ryy, rzz;

  RotationMatrixToEulerAngles(EulerAnglesToRotationMatrix(rx, ry, rz), &rxx,
                              &ryy, &rzz);

  BOOST_CHECK_CLOSE(rx, rxx, 1e-6);
  BOOST_CHECK_CLOSE(ry, ryy, 1e-6);
  BOOST_CHECK_CLOSE(rz, rzz, 1e-6);
}

BOOST_AUTO_TEST_CASE(TestEulerAnglesZ) {
  const double rx = 0;
  const double ry = 0;
  const double rz = 0.3;
  double rxx, ryy, rzz;

  RotationMatrixToEulerAngles(EulerAnglesToRotationMatrix(rx, ry, rz), &rxx,
                              &ryy, &rzz);

  BOOST_CHECK_CLOSE(rx, rxx, 1e-6);
  BOOST_CHECK_CLOSE(ry, ryy, 1e-6);
  BOOST_CHECK_CLOSE(rz, rzz, 1e-6);
}

BOOST_AUTO_TEST_CASE(TestQuaternionToRotationMatrix) {
  const double rx = 0;
  const double ry = 0;
  const double rz = 0.3;
  const Eigen::Matrix3d rot_mat0 = EulerAnglesToRotationMatrix(rx, ry, rz);
  const Eigen::Matrix3d rot_mat1 =
      QuaternionToRotationMatrix(RotationMatrixToQuaternion(rot_mat0));
  BOOST_CHECK_LT((rot_mat0 - rot_mat1).norm(), 1e-8);
}

BOOST_AUTO_TEST_CASE(TestNormalizeQuaternion) {
  BOOST_CHECK_EQUAL(NormalizeQuaternion(Eigen::Vector4d(1, 0, 0, 0)),
                    Eigen::Vector4d(1, 0, 0, 0));
  BOOST_CHECK_EQUAL(NormalizeQuaternion(Eigen::Vector4d(2, 0, 0, 0)),
                    Eigen::Vector4d(1, 0, 0, 0));
  BOOST_CHECK_EQUAL(NormalizeQuaternion(Eigen::Vector4d(0.5, 0, 0, 0)),
                    Eigen::Vector4d(1, 0, 0, 0));
  BOOST_CHECK_EQUAL(NormalizeQuaternion(Eigen::Vector4d(0, 0, 0, 0)),
                    Eigen::Vector4d(1, 0, 0, 0));
  BOOST_CHECK_LT((NormalizeQuaternion(Eigen::Vector4d(1, 1, 0, 0)) -
                  Eigen::Vector4d(std::sqrt(2) / 2, std::sqrt(2) / 2, 0, 0))
                     .norm(),
                 1e-10);
  BOOST_CHECK_LT((NormalizeQuaternion(Eigen::Vector4d(0.5, 0.5, 0, 0)) -
                  Eigen::Vector4d(std::sqrt(2) / 2, std::sqrt(2) / 2, 0, 0))
                     .norm(),
                 1e-10);
}

BOOST_AUTO_TEST_CASE(TestInvertQuaternion) {
  BOOST_CHECK_EQUAL(InvertQuaternion(Eigen::Vector4d(1, 0, 0, 0)),
                    Eigen::Vector4d(1, -0, -0, -0));
  BOOST_CHECK_EQUAL(InvertQuaternion(Eigen::Vector4d(2, 0, 0, 0)),
                    Eigen::Vector4d(1, -0, -0, -0));
  BOOST_CHECK_LT(
      (InvertQuaternion(InvertQuaternion(Eigen::Vector4d(0.1, 0.2, 0.3, 0.4))) -
       NormalizeQuaternion(Eigen::Vector4d(0.1, 0.2, 0.3, 0.4)))
          .norm(),
      1e-10);
}

BOOST_AUTO_TEST_CASE(TestConcatenateQuaternions) {
  BOOST_CHECK_EQUAL(ConcatenateQuaternions(Eigen::Vector4d(1, 0, 0, 0),
                                           Eigen::Vector4d(1, 0, 0, 0)),
                    Eigen::Vector4d(1, 0, 0, 0));
  BOOST_CHECK_EQUAL(ConcatenateQuaternions(Eigen::Vector4d(2, 0, 0, 0),
                                           Eigen::Vector4d(1, 0, 0, 0)),
                    Eigen::Vector4d(1, 0, 0, 0));
  BOOST_CHECK_EQUAL(ConcatenateQuaternions(Eigen::Vector4d(1, 0, 0, 0),
                                           Eigen::Vector4d(2, 0, 0, 0)),
                    Eigen::Vector4d(1, 0, 0, 0));
  BOOST_CHECK_LT((ConcatenateQuaternions(
                      Eigen::Vector4d(0.1, 0.2, 0.3, 0.4),
                      InvertQuaternion(Eigen::Vector4d(0.1, 0.2, 0.3, 0.4))) -
                  Eigen::Vector4d(1, 0, 0, 0))
                     .norm(),
                 1e-10);
  BOOST_CHECK_LT((ConcatenateQuaternions(
                      InvertQuaternion(Eigen::Vector4d(0.1, 0.2, 0.3, 0.4)),
                      Eigen::Vector4d(0.1, 0.2, 0.3, 0.4)) -
                  Eigen::Vector4d(1, 0, 0, 0))
                     .norm(),
                 1e-10);
}

BOOST_AUTO_TEST_CASE(TestQuaternionRotatePoint) {
  BOOST_CHECK_EQUAL(QuaternionRotatePoint(Eigen::Vector4d(1, 0, 0, 0),
                                          Eigen::Vector3d(0, 0, 0)),
                    Eigen::Vector3d(0, 0, 0));
  BOOST_CHECK_EQUAL(QuaternionRotatePoint(Eigen::Vector4d(0.1, 0, 0, 0),
                                          Eigen::Vector3d(0, 0, 0)),
                    Eigen::Vector3d(0, 0, 0));
  BOOST_CHECK_EQUAL(QuaternionRotatePoint(Eigen::Vector4d(1, 0, 0, 0),
                                          Eigen::Vector3d(1, 1, 0)),
                    Eigen::Vector3d(1, 1, 0));
  BOOST_CHECK_EQUAL(QuaternionRotatePoint(Eigen::Vector4d(0.1, 0, 0, 0),
                                          Eigen::Vector3d(1, 1, 0)),
                    Eigen::Vector3d(1, 1, 0));
  BOOST_CHECK_LT(
      (QuaternionRotatePoint(
           RotationMatrixToQuaternion(EulerAnglesToRotationMatrix(M_PI, 0, 0)),
           Eigen::Vector3d(1, 1, 0)) -
       Eigen::Vector3d(1, -1, 0))
          .norm(),
      1e-10);
}

BOOST_AUTO_TEST_CASE(TestPoseFromProjectionMatrix) {
  const Eigen::Vector4d qvec = Eigen::Vector4d::Random().normalized();
  const Eigen::Vector3d tvec(3, 4, 5);
  const Eigen::Matrix3x4d proj_matrix = ComposeProjectionMatrix(qvec, tvec);
  const Eigen::Matrix3x4d inv_proj_matrix = InvertProjectionMatrix(proj_matrix);
  const Eigen::Vector3d pose = ProjectionCenterFromMatrix(proj_matrix);
  BOOST_CHECK_CLOSE((inv_proj_matrix.rightCols<1>() - pose).norm(), 0, 1e-6);
}

BOOST_AUTO_TEST_CASE(TestPoseFromProjectionParameters) {
  const Eigen::Vector4d qvec = Eigen::Vector4d::Random().normalized();
  const Eigen::Vector3d tvec(3, 4, 5);
  const Eigen::Matrix3x4d proj_matrix = ComposeProjectionMatrix(qvec, tvec);
  const Eigen::Matrix3x4d inv_proj_matrix = InvertProjectionMatrix(proj_matrix);
  const Eigen::Vector3d pose = ProjectionCenterFromParameters(qvec, tvec);
  BOOST_CHECK((inv_proj_matrix.rightCols<1>() - pose).norm() < 1e-6);
}

BOOST_AUTO_TEST_CASE(TestInterpolatePose) {
  const Eigen::Vector4d qvec1 = Eigen::Vector4d::Random().normalized();
  const Eigen::Vector3d tvec1 = Eigen::Vector3d::Random();
  const Eigen::Vector4d qvec2 = Eigen::Vector4d::Random().normalized();
  const Eigen::Vector3d tvec2 = Eigen::Vector3d::Random();

  Eigen::Vector4d qveci;
  Eigen::Vector3d tveci;

  InterpolatePose(qvec1, tvec1, qvec2, tvec2, 0, &qveci, &tveci);
  BOOST_CHECK_LT((tvec1 - tveci).norm(), 1e-6);

  InterpolatePose(qvec1, tvec1, qvec2, tvec2, 1, &qveci, &tveci);
  BOOST_CHECK_LT((tvec2 - tveci).norm(), 1e-6);

  InterpolatePose(qvec1, tvec1, qvec2, tvec2, 0.5, &qveci, &tveci);
  BOOST_CHECK_LT(((tvec1 + tvec2) / 2 - tveci).norm(), 1e-6);
}

BOOST_AUTO_TEST_CASE(TestCalculateBaseline) {
  Eigen::Vector4d qvec1(1, 0, 0, 0);
  Eigen::Vector4d qvec2(1, 0, 0, 0);

  Eigen::Vector3d tvec1(0, 0, 0);
  Eigen::Vector3d tvec2(0, 0, 1);

  const double baseline1 = CalculateBaseline(qvec1, tvec1, qvec2, tvec2).norm();
  BOOST_CHECK_CLOSE(baseline1, 1, 1e-10);

  tvec2(2) = 2;

  const double baseline2 = CalculateBaseline(qvec1, tvec1, qvec2, tvec2).norm();
  BOOST_CHECK_CLOSE(baseline2, 2, 1e-10);
}

BOOST_AUTO_TEST_CASE(TestCheckCheirality) {
  const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t(1, 0, 0);

  std::vector<Eigen::Vector3d> points1;
  std::vector<Eigen::Vector3d> points2;
  std::vector<Eigen::Vector3d> points3D;

  points1.emplace_back(0, 0, 1);
  points2.emplace_back(0.1, 0, 1);
  BOOST_CHECK(CheckCheirality(R, t, points1, points2, &points3D));
  BOOST_CHECK_EQUAL(points3D.size(), 1);

  points1.emplace_back(0, 0, 1);
  points2.emplace_back(-0.1, 0, 1);
  BOOST_CHECK(CheckCheirality(R, t, points1, points2, &points3D));
  BOOST_CHECK_EQUAL(points3D.size(), 1);

  points2[1][0] = 0.2;
  BOOST_CHECK(CheckCheirality(R, t, points1, points2, &points3D));
  BOOST_CHECK_EQUAL(points3D.size(), 2);

  points2[0][0] = -0.2;
  points2[1][0] = -0.2;
  BOOST_CHECK(!CheckCheirality(R, t, points1, points2, &points3D));
  BOOST_CHECK_EQUAL(points3D.size(), 0);
}

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

#define TEST_NAME "base/essential_matrix"
#include "util/testing.h"

#include <Eigen/Geometry>

#include "base/essential_matrix.h"
#include "base/pose.h"
#include "base/projection.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestDecomposeEssentialMatrix) {
  const Eigen::Matrix3d R = EulerAnglesToRotationMatrix(0, 1, 1);
  const Eigen::Vector3d t = Eigen::Vector3d(0.5, 1, 1).normalized();
  const Eigen::Matrix3d E = EssentialMatrixFromPose(R, t);

  Eigen::Matrix3d R1;
  Eigen::Matrix3d R2;
  Eigen::Vector3d tt;
  DecomposeEssentialMatrix(E, &R1, &R2, &tt);

  BOOST_CHECK((R1 - R).norm() < 1e-10 || (R2 - R).norm() < 1e-10);
  BOOST_CHECK((tt - t).norm() < 1e-10 || (tt + t).norm() < 1e-10);
}

BOOST_AUTO_TEST_CASE(TestEssentialMatrixFromPose) {
  BOOST_CHECK_EQUAL(
      EssentialMatrixFromPose(EulerAnglesToRotationMatrix(0, 0, 0),
                              Eigen::Vector3d(0, 0, 1)),
      (Eigen::MatrixXd(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 0).finished());
  BOOST_CHECK_EQUAL(
      EssentialMatrixFromPose(EulerAnglesToRotationMatrix(0, 0, 0),
                              Eigen::Vector3d(0, 0, 2)),
      (Eigen::MatrixXd(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 0).finished());
}

BOOST_AUTO_TEST_CASE(TestEssentialMatrixFromPoses) {
  const Eigen::Matrix3d R1 = EulerAnglesToRotationMatrix(0, 0, 0);
  const Eigen::Matrix3d R2 = EulerAnglesToRotationMatrix(0, 1, 2);
  const Eigen::Vector3d t1(0, 0, 0);
  const Eigen::Vector3d t2 = Eigen::Vector3d(0.5, 1, 1).normalized();

  const Eigen::Matrix3d E1 = EssentialMatrixFromPose(R2, t2);
  const Eigen::Matrix3d E2 = EssentialMatrixFromAbsolutePoses(
      ComposeProjectionMatrix(R1, t1), ComposeProjectionMatrix(R2, t2));

  BOOST_CHECK_CLOSE((E1 - E2).norm(), 0, 1e-6);
}

BOOST_AUTO_TEST_CASE(TestPoseFromEssentialMatrix) {
  const Eigen::Matrix3d R = EulerAnglesToRotationMatrix(0, 0, 0);
  const Eigen::Vector3d t = Eigen::Vector3d(1, 0, 0).normalized();
  const Eigen::Matrix3d E = EssentialMatrixFromPose(R, t);

  const Eigen::Matrix3x4d proj_matrix1 = Eigen::Matrix3x4d::Identity();
  const Eigen::Matrix3x4d proj_matrix2 = ComposeProjectionMatrix(R, t);

  std::vector<Eigen::Vector3d> points3D(4);
  points3D[0] = Eigen::Vector3d(0, 0, 1);
  points3D[1] = Eigen::Vector3d(0, 0.1, 1);
  points3D[2] = Eigen::Vector3d(0.1, 0, 1);
  points3D[3] = Eigen::Vector3d(0.1, 0.1, 1);

  std::vector<Eigen::Vector2d> points1(4);
  std::vector<Eigen::Vector2d> points2(4);
  for (size_t i = 0; i < points3D.size(); ++i) {
    const Eigen::Vector3d point1 = proj_matrix1 * points3D[i].homogeneous();
    points1[i] = point1.hnormalized();
    const Eigen::Vector3d point2 = proj_matrix2 * points3D[i].homogeneous();
    points2[i] = point2.hnormalized();
  }

  points3D.clear();

  Eigen::Matrix3d RR;
  Eigen::Vector3d tt;
  PoseFromEssentialMatrix(E, points1, points2, &RR, &tt, &points3D);

  BOOST_CHECK_EQUAL(points3D.size(), 4);

  BOOST_CHECK(RR.isApprox(R));
  BOOST_CHECK(tt.isApprox(t));
}

BOOST_AUTO_TEST_CASE(TestFindOptimalImageObservations) {
  const Eigen::Matrix3d R = EulerAnglesToRotationMatrix(0, 0, 0);
  const Eigen::Vector3d t = Eigen::Vector3d(1, 0, 0).normalized();
  const Eigen::Matrix3d E = EssentialMatrixFromPose(R, t);

  const Eigen::Matrix3x4d proj_matrix1 = Eigen::Matrix3x4d::Identity();
  const Eigen::Matrix3x4d proj_matrix2 = ComposeProjectionMatrix(R, t);

  std::vector<Eigen::Vector3d> points3D(4);
  points3D[0] = Eigen::Vector3d(0, 0, 1);
  points3D[1] = Eigen::Vector3d(0, 0.1, 1);
  points3D[2] = Eigen::Vector3d(0.1, 0, 1);
  points3D[3] = Eigen::Vector3d(0.1, 0.1, 1);

  // Test if perfect projection is equivalent to optimal image observations.
  for (size_t i = 0; i < points3D.size(); ++i) {
    const Eigen::Vector3d point1_homogeneous =
        proj_matrix1 * points3D[i].homogeneous();
    const Eigen::Vector2d point1 = point1_homogeneous.hnormalized();
    const Eigen::Vector3d point2_homogeneous =
        proj_matrix2 * points3D[i].homogeneous();
    const Eigen::Vector2d point2 = point2_homogeneous.hnormalized();
    Eigen::Vector2d optimal_point1;
    Eigen::Vector2d optimal_point2;
    FindOptimalImageObservations(E, point1, point2, &optimal_point1,
                                 &optimal_point2);
    BOOST_CHECK(point1.isApprox(optimal_point1));
    BOOST_CHECK(point2.isApprox(optimal_point2));
  }
}

BOOST_AUTO_TEST_CASE(TestEpipoleFromEssentialMatrix) {
  const Eigen::Matrix3d R = EulerAnglesToRotationMatrix(0, 0, 0);
  const Eigen::Vector3d t = Eigen::Vector3d(0, 0, -1).normalized();
  const Eigen::Matrix3d E = EssentialMatrixFromPose(R, t);

  const Eigen::Vector3d left_epipole = EpipoleFromEssentialMatrix(E, true);
  const Eigen::Vector3d right_epipole = EpipoleFromEssentialMatrix(E, false);
  BOOST_CHECK(left_epipole.isApprox(Eigen::Vector3d(0, 0, 1)));
  BOOST_CHECK(right_epipole.isApprox(Eigen::Vector3d(0, 0, 1)));
}

BOOST_AUTO_TEST_CASE(TestInvertEssentialMatrix) {
  for (size_t i = 1; i < 10; ++i) {
    const Eigen::Matrix3d R = EulerAnglesToRotationMatrix(0, 0.1, 0);
    const Eigen::Vector3d t = Eigen::Vector3d(0, 0, i).normalized();
    const Eigen::Matrix3d E = EssentialMatrixFromPose(R, t);
    const Eigen::Matrix3d inv_inv_E =
        InvertEssentialMatrix(InvertEssentialMatrix(E));
    BOOST_CHECK(E.isApprox(inv_inv_E));
  }
}

BOOST_AUTO_TEST_CASE(TestRefineEssentialMatrix) {
  const Eigen::Matrix3d R = EulerAnglesToRotationMatrix(0, 0, 0);
  const Eigen::Vector3d t = Eigen::Vector3d(1, 0, 0).normalized();
  const Eigen::Matrix3d E = EssentialMatrixFromPose(R, t);

  const Eigen::Matrix3x4d proj_matrix1 = Eigen::Matrix3x4d::Identity();
  const Eigen::Matrix3x4d proj_matrix2 = ComposeProjectionMatrix(R, t);

  std::vector<Eigen::Vector3d> points3D(150);
  for (size_t i = 0; i < points3D.size() / 3; ++i) {
    points3D[3 * i + 0] = Eigen::Vector3d(i * 0.01, 0, 1);
    points3D[3 * i + 1] = Eigen::Vector3d(0, i * 0.01, 1);
    points3D[3 * i + 2] = Eigen::Vector3d(i * 0.01, i * 0.01, 1);
  }

  std::vector<Eigen::Vector2d> points1(points3D.size());
  std::vector<Eigen::Vector2d> points2(points3D.size());
  for (size_t i = 0; i < points3D.size(); ++i) {
    const Eigen::Vector3d point1 = proj_matrix1 * points3D[i].homogeneous();
    points1[i] = point1.hnormalized();
    const Eigen::Vector3d point2 = proj_matrix2 * points3D[i].homogeneous();
    points2[i] = point2.hnormalized();
  }

  const Eigen::Matrix3d R_pertubated = EulerAnglesToRotationMatrix(0, 0, 0);
  const Eigen::Vector3d t_pertubated =
      Eigen::Vector3d(1.02, 0.02, 0.02).normalized();
  const Eigen::Matrix3d E_pertubated =
      EssentialMatrixFromPose(R_pertubated, t_pertubated);

  Eigen::Matrix3d E_refined = E_pertubated;
  ceres::Solver::Options options;
  RefineEssentialMatrix(options, points1, points2,
                        std::vector<char>(points1.size(), true), &E_refined);

  BOOST_CHECK_LE((E - E_refined).norm(), (E - E_pertubated).norm());
}

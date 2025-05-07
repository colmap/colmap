// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#include "colmap/geometry/essential_matrix.h"

#include "colmap/geometry/pose.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/eigen_matchers.h"

#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(DecomposeEssentialMatrix, Nominal) {
  const Rigid3d cam2_from_cam1(Eigen::Quaterniond::UnitRandom(),
                               Eigen::Vector3d(0.5, 1, 1).normalized());
  const Eigen::Matrix3d cam2_from_cam1_rot_mat =
      cam2_from_cam1.rotation.toRotationMatrix();
  const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);

  Eigen::Matrix3d R1;
  Eigen::Matrix3d R2;
  Eigen::Vector3d t;
  DecomposeEssentialMatrix(E, &R1, &R2, &t);

  EXPECT_TRUE((R1 - cam2_from_cam1_rot_mat).norm() < 1e-10 ||
              (R2 - cam2_from_cam1_rot_mat).norm() < 1e-10);
  EXPECT_TRUE((t - cam2_from_cam1.translation).norm() < 1e-10 ||
              (t + cam2_from_cam1.translation).norm() < 1e-10);
}

TEST(EssentialMatrixFromPose, Nominal) {
  EXPECT_EQ(EssentialMatrixFromPose(Rigid3d(Eigen::Quaterniond::Identity(),
                                            Eigen::Vector3d(0, 0, 1))),
            (Eigen::MatrixXd(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 0).finished());
  EXPECT_EQ(EssentialMatrixFromPose(Rigid3d(Eigen::Quaterniond::Identity(),
                                            Eigen::Vector3d(0, 0, 2))),
            (Eigen::MatrixXd(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 0).finished());
}

TEST(PoseFromEssentialMatrix, Nominal) {
  const Rigid3d cam1_from_world;
  const Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                                Eigen::Vector3d(1, 0, 0).normalized());
  const Rigid3d cam2_from_cam1 = cam2_from_world * Inverse(cam1_from_world);
  const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);

  std::vector<Eigen::Vector3d> points3D(4);
  points3D[0] = Eigen::Vector3d(0, 0, 1);
  points3D[1] = Eigen::Vector3d(0, 0.1, 1);
  points3D[2] = Eigen::Vector3d(0.1, 0, 1);
  points3D[3] = Eigen::Vector3d(0.1, 0.1, 1);

  std::vector<Eigen::Vector2d> points1(4);
  std::vector<Eigen::Vector2d> points2(4);
  for (size_t i = 0; i < points3D.size(); ++i) {
    points1[i] = (cam1_from_world * points3D[i]).hnormalized();
    points2[i] = (cam2_from_world * points3D[i]).hnormalized();
  }

  points3D.clear();

  Rigid3d cam2_from_cam1_est;
  PoseFromEssentialMatrix(E, points1, points2, &cam2_from_cam1_est, &points3D);

  EXPECT_EQ(points3D.size(), 4);

  EXPECT_THAT(cam2_from_cam1_est, Rigid3dNear(cam2_from_cam1));
}

TEST(FindOptimalImageObservations, Nominal) {
  const Rigid3d cam1_from_world;
  const Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                                Eigen::Vector3d(1, 0, 0).normalized());
  const Eigen::Matrix3d E =
      EssentialMatrixFromPose(cam2_from_world * Inverse(cam1_from_world));

  std::vector<Eigen::Vector3d> points3D(4);
  points3D[0] = Eigen::Vector3d(0, 0, 1);
  points3D[1] = Eigen::Vector3d(0, 0.1, 1);
  points3D[2] = Eigen::Vector3d(0.1, 0, 1);
  points3D[3] = Eigen::Vector3d(0.1, 0.1, 1);

  // Test if perfect projection is equivalent to optimal image observations.
  for (size_t i = 0; i < points3D.size(); ++i) {
    const Eigen::Vector2d point1 =
        (cam1_from_world * points3D[i]).hnormalized();
    const Eigen::Vector2d point2 =
        (cam2_from_world * points3D[i]).hnormalized();
    Eigen::Vector2d optimal_point1;
    Eigen::Vector2d optimal_point2;
    FindOptimalImageObservations(
        E, point1, point2, &optimal_point1, &optimal_point2);
    EXPECT_THAT(point1, EigenMatrixNear(optimal_point1));
    EXPECT_THAT(point2, EigenMatrixNear(optimal_point2));
  }
}

TEST(EpipoleFromEssentialMatrix, Nominal) {
  const Rigid3d cam2_from_cam1(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d(0, 0, -1).normalized());
  const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);

  const Eigen::Vector3d left_epipole = EpipoleFromEssentialMatrix(E, true);
  const Eigen::Vector3d right_epipole = EpipoleFromEssentialMatrix(E, false);
  EXPECT_THAT(left_epipole, EigenMatrixNear(Eigen::Vector3d(0, 0, 1)));
  EXPECT_THAT(right_epipole, EigenMatrixNear(Eigen::Vector3d(0, 0, 1)));
}

TEST(InvertEssentialMatrix, Nominal) {
  for (size_t i = 1; i < 10; ++i) {
    const Rigid3d cam2_from_cam1(
        Eigen::Quaterniond(EulerAnglesToRotationMatrix(0, 0.1, 0)),
        Eigen::Vector3d(0, 0, i).normalized());
    const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);
    const Eigen::Matrix3d inv_inv_E =
        InvertEssentialMatrix(InvertEssentialMatrix(E));
    EXPECT_THAT(E, EigenMatrixNear(inv_inv_E));
  }
}

TEST(FundamentalFromEssentialMatrix, Nominal) {
  const Eigen::Matrix3d E = EssentialMatrixFromPose(
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random()));
  const Eigen::Matrix3d K1 =
      (Eigen::Matrix3d() << 2, 0, 1, 0, 3, 2, 0, 0, 1).finished();
  const Eigen::Matrix3d K2 =
      (Eigen::Matrix3d() << 3, 0, 2, 0, 4, 1, 0, 0, 1).finished();
  const Eigen::Matrix3d F = FundamentalFromEssentialMatrix(K2, E, K1);
  const Eigen::Vector3d x(3, 2, 1);
  EXPECT_THAT(K2.transpose().inverse() * E * x,
              EigenMatrixNear(Eigen::Vector3d(F * K1 * x)));
  EXPECT_THAT(E * K1.inverse() * x,
              EigenMatrixNear(Eigen::Vector3d(K2.transpose() * F * x)));
}

TEST(EssentialFromFundamentalMatrix, Nominal) {
  const Eigen::Matrix3d E = EssentialMatrixFromPose(
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random()));
  const Eigen::Matrix3d K1 =
      (Eigen::Matrix3d() << 2, 0, 1, 0, 3, 2, 0, 0, 1).finished();
  const Eigen::Matrix3d K2 =
      (Eigen::Matrix3d() << 3, 0, 2, 0, 4, 1, 0, 0, 1).finished();
  const Eigen::Matrix3d F = FundamentalFromEssentialMatrix(K2, E, K1);
  EXPECT_THAT(EssentialFromFundamentalMatrix(K2, F, K1),
              EigenMatrixNear(E, 1e-6));
}

}  // namespace
}  // namespace colmap

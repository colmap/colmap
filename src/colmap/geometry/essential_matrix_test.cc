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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "colmap/geometry/essential_matrix.h"

#include "colmap/geometry/pose.h"
#include "colmap/geometry/projection.h"

#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {

TEST(DecomposeEssentialMatrix, Nominal) {
  const Eigen::Matrix3d R = EulerAnglesToRotationMatrix(0, 1, 1);
  const Eigen::Vector3d t = Eigen::Vector3d(0.5, 1, 1).normalized();
  const Eigen::Matrix3d E = EssentialMatrixFromPose(R, t);

  Eigen::Matrix3d R1;
  Eigen::Matrix3d R2;
  Eigen::Vector3d tt;
  DecomposeEssentialMatrix(E, &R1, &R2, &tt);

  EXPECT_TRUE((R1 - R).norm() < 1e-10 || (R2 - R).norm() < 1e-10);
  EXPECT_TRUE((tt - t).norm() < 1e-10 || (tt + t).norm() < 1e-10);
}

TEST(EssentialMatrixFromPose, Nominal) {
  EXPECT_EQ(EssentialMatrixFromPose(EulerAnglesToRotationMatrix(0, 0, 0),
                                    Eigen::Vector3d(0, 0, 1)),
            (Eigen::MatrixXd(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 0).finished());
  EXPECT_EQ(EssentialMatrixFromPose(EulerAnglesToRotationMatrix(0, 0, 0),
                                    Eigen::Vector3d(0, 0, 2)),
            (Eigen::MatrixXd(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 0).finished());
}

TEST(EssentialMatrixFromPoses, Nominal) {
  const Eigen::Matrix3d R1 = EulerAnglesToRotationMatrix(0, 0, 0);
  const Eigen::Matrix3d R2 = EulerAnglesToRotationMatrix(0, 1, 2);
  const Eigen::Vector3d t1(0, 0, 0);
  const Eigen::Vector3d t2 = Eigen::Vector3d(0.5, 1, 1).normalized();

  const Eigen::Matrix3d E1 = EssentialMatrixFromPose(R2, t2);
  const Eigen::Matrix3d E2 = EssentialMatrixFromAbsolutePoses(
      ComposeProjectionMatrix(R1, t1), ComposeProjectionMatrix(R2, t2));

  EXPECT_NEAR((E1 - E2).norm(), 0, 1e-6);
}

TEST(PoseFromEssentialMatrix, Nominal) {
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

  EXPECT_EQ(points3D.size(), 4);

  EXPECT_TRUE(RR.isApprox(R));
  EXPECT_TRUE(tt.isApprox(t));
}

TEST(FindOptimalImageObservations, Nominal) {
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
    FindOptimalImageObservations(
        E, point1, point2, &optimal_point1, &optimal_point2);
    EXPECT_TRUE(point1.isApprox(optimal_point1));
    EXPECT_TRUE(point2.isApprox(optimal_point2));
  }
}

TEST(EpipoleFromEssentialMatrix, Nominal) {
  const Eigen::Matrix3d R = EulerAnglesToRotationMatrix(0, 0, 0);
  const Eigen::Vector3d t = Eigen::Vector3d(0, 0, -1).normalized();
  const Eigen::Matrix3d E = EssentialMatrixFromPose(R, t);

  const Eigen::Vector3d left_epipole = EpipoleFromEssentialMatrix(E, true);
  const Eigen::Vector3d right_epipole = EpipoleFromEssentialMatrix(E, false);
  EXPECT_TRUE(left_epipole.isApprox(Eigen::Vector3d(0, 0, 1)));
  EXPECT_TRUE(right_epipole.isApprox(Eigen::Vector3d(0, 0, 1)));
}

TEST(InvertEssentialMatrix, Nominal) {
  for (size_t i = 1; i < 10; ++i) {
    const Eigen::Matrix3d R = EulerAnglesToRotationMatrix(0, 0.1, 0);
    const Eigen::Vector3d t = Eigen::Vector3d(0, 0, i).normalized();
    const Eigen::Matrix3d E = EssentialMatrixFromPose(R, t);
    const Eigen::Matrix3d inv_inv_E =
        InvertEssentialMatrix(InvertEssentialMatrix(E));
    EXPECT_TRUE(E.isApprox(inv_inv_E));
  }
}

TEST(RefineEssentialMatrix, Nominal) {
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
  RefineEssentialMatrix(options,
                        points1,
                        points2,
                        std::vector<char>(points1.size(), true),
                        &E_refined);

  EXPECT_LE((E - E_refined).norm(), (E - E_pertubated).norm());
}

}  // namespace colmap

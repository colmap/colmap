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

#include "colmap/geometry/pose.h"

#include "colmap/math/math.h"
#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(ComputeClosestRotationMatrix, Nominal) {
  const Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
  EXPECT_LT((ComputeClosestRotationMatrix(A) - A).norm(), 1e-6);
  EXPECT_LT((ComputeClosestRotationMatrix(2 * A) - A).norm(), 1e-6);
}

TEST(DecomposeProjectionMatrix, Nominal) {
  for (int i = 1; i < 100; ++i) {
    Eigen::Matrix3d ref_K = i * Eigen::Matrix3d::Identity();
    ref_K(0, 2) = i;
    ref_K(1, 2) = 2 * i;
    const Rigid3d cam_from_world(Eigen::Quaterniond::UnitRandom(),
                                 Eigen::Vector3d::Random());
    const Eigen::Matrix3x4d P = ref_K * cam_from_world.ToMatrix();
    Eigen::Matrix3d K;
    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    DecomposeProjectionMatrix(P, &K, &R, &T);
    EXPECT_TRUE(ref_K.isApprox(K, 1e-6));
    EXPECT_TRUE(cam_from_world.rotation.toRotationMatrix().isApprox(R, 1e-6));
    EXPECT_TRUE(cam_from_world.translation.isApprox(T, 1e-6));
  }
}

TEST(CrossProductMatrix, Nominal) {
  EXPECT_EQ(CrossProductMatrix(Eigen::Vector3d(0, 0, 0)),
            Eigen::Matrix3d::Zero());
  Eigen::Matrix3d ref_matrix;
  ref_matrix << 0, -3, 2, 3, 0, -1, -2, 1, 0;
  EXPECT_EQ(CrossProductMatrix(Eigen::Vector3d(1, 2, 3)), ref_matrix);
}

TEST(EulerAngles, X) {
  const double rx = 0.3;
  const double ry = 0;
  const double rz = 0;
  double rxx, ryy, rzz;

  RotationMatrixToEulerAngles(
      EulerAnglesToRotationMatrix(rx, ry, rz), &rxx, &ryy, &rzz);

  EXPECT_NEAR(rx, rxx, 1e-6);
  EXPECT_NEAR(ry, ryy, 1e-6);
  EXPECT_NEAR(rz, rzz, 1e-6);
}

TEST(EulerAngles, Y) {
  const double rx = 0;
  const double ry = 0.3;
  const double rz = 0;
  double rxx, ryy, rzz;

  RotationMatrixToEulerAngles(
      EulerAnglesToRotationMatrix(rx, ry, rz), &rxx, &ryy, &rzz);

  EXPECT_NEAR(rx, rxx, 1e-6);
  EXPECT_NEAR(ry, ryy, 1e-6);
  EXPECT_NEAR(rz, rzz, 1e-6);
}

TEST(EulerAngles, Z) {
  const double rx = 0;
  const double ry = 0;
  const double rz = 0.3;
  double rxx, ryy, rzz;

  RotationMatrixToEulerAngles(
      EulerAnglesToRotationMatrix(rx, ry, rz), &rxx, &ryy, &rzz);

  EXPECT_NEAR(rx, rxx, 1e-6);
  EXPECT_NEAR(ry, ryy, 1e-6);
  EXPECT_NEAR(rz, rzz, 1e-6);
}

TEST(EulerAngles, XYZ) {
  const double rx = 0.1;
  const double ry = 0.2;
  const double rz = 0.3;
  double rxx, ryy, rzz;

  RotationMatrixToEulerAngles(
      EulerAnglesToRotationMatrix(rx, ry, rz), &rxx, &ryy, &rzz);

  EXPECT_NEAR(rx, rxx, 1e-6);
  EXPECT_NEAR(ry, ryy, 1e-6);
  EXPECT_NEAR(rz, rzz, 1e-6);
}

TEST(AverageQuaternions, Nominal) {
  std::vector<Eigen::Quaterniond> quats;
  std::vector<double> weights;

  quats = {{Eigen::Quaterniond::Identity()}};
  weights = {1.0};
  EXPECT_EQ(AverageQuaternions(quats, weights).coeffs(),
            Eigen::Quaterniond::Identity().coeffs());

  quats = {Eigen::Quaterniond::Identity()};
  weights = {2.0};
  EXPECT_EQ(AverageQuaternions(quats, weights).coeffs(),
            Eigen::Quaterniond::Identity().coeffs());

  quats = {Eigen::Quaterniond::Identity(), Eigen::Quaterniond::Identity()};
  weights = {1.0, 1.0};
  EXPECT_EQ(AverageQuaternions(quats, weights).coeffs(),
            Eigen::Quaterniond::Identity().coeffs());

  quats = {Eigen::Quaterniond::Identity(), Eigen::Quaterniond::Identity()};
  weights = {1.0, 2.0};
  EXPECT_EQ(AverageQuaternions(quats, weights).coeffs(),
            Eigen::Quaterniond::Identity().coeffs());

  quats = {Eigen::Quaterniond::Identity(), Eigen::Quaterniond(2, 0, 0, 0)};
  weights = {1.0, 2.0};
  EXPECT_EQ(AverageQuaternions(quats, weights).coeffs(),
            Eigen::Quaterniond::Identity().coeffs());

  quats = {Eigen::Quaterniond::Identity(), Eigen::Quaterniond(1, 1, 0, 0)};
  weights = {1.0, 1.0};
  EXPECT_TRUE(AverageQuaternions(quats, weights)
                  .isApprox(Eigen::Quaterniond(0.92388, 0.382683, 0, 0), 1e-6));

  quats = {Eigen::Quaterniond::Identity(), Eigen::Quaterniond(1, 1, 0, 0)};
  weights = {1.0, 2.0};
  EXPECT_TRUE(
      AverageQuaternions(quats, weights)
          .isApprox(Eigen::Quaterniond(0.850651, 0.525731, 0, 0), 1e-6));
}

TEST(InterpolateCameraPoses, Nominal) {
  const Rigid3d cam_from_world1(Eigen::Quaterniond::UnitRandom(),
                                Eigen::Vector3d::Random());
  const Rigid3d cam_from_world2(Eigen::Quaterniond::UnitRandom(),
                                Eigen::Vector3d::Random());

  const Rigid3d interp_cam_from_world1 =
      InterpolateCameraPoses(cam_from_world1, cam_from_world2, 0);
  EXPECT_TRUE(
      interp_cam_from_world1.translation.isApprox(cam_from_world1.translation));

  const Rigid3d interp_cam_from_world2 =
      InterpolateCameraPoses(cam_from_world1, cam_from_world2, 1);
  EXPECT_TRUE(
      interp_cam_from_world2.translation.isApprox(cam_from_world2.translation));

  const Rigid3d interp_cam_from_world3 =
      InterpolateCameraPoses(cam_from_world1, cam_from_world2, 0.5);
  EXPECT_TRUE(interp_cam_from_world3.translation.isApprox(
      (cam_from_world1.translation + cam_from_world2.translation) / 2));
}

TEST(CheckCheirality, Nominal) {
  const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t(1, 0, 0);

  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  std::vector<Eigen::Vector3d> points3D;

  points1.emplace_back(0, 0);
  points2.emplace_back(0.1, 0);
  EXPECT_TRUE(CheckCheirality(R, t, points1, points2, &points3D));
  EXPECT_EQ(points3D.size(), 1);

  points1.emplace_back(0, 0);
  points2.emplace_back(-0.1, 0);
  EXPECT_TRUE(CheckCheirality(R, t, points1, points2, &points3D));
  EXPECT_EQ(points3D.size(), 1);

  points2[1][0] = 0.2;
  EXPECT_TRUE(CheckCheirality(R, t, points1, points2, &points3D));
  EXPECT_EQ(points3D.size(), 2);

  points2[0][0] = -0.2;
  points2[1][0] = -0.2;
  EXPECT_FALSE(CheckCheirality(R, t, points1, points2, &points3D));
  EXPECT_EQ(points3D.size(), 0);
}

}  // namespace
}  // namespace colmap

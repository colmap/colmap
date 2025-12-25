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

#include "colmap/geometry/pose.h"

#include "colmap/math/math.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/eigen_matchers.h"

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(ComputeClosestRotationMatrix, Nominal) {
  const Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
  EXPECT_THAT(ComputeClosestRotationMatrix(A), EigenMatrixNear(A, 1e-6));
  EXPECT_THAT(ComputeClosestRotationMatrix(2 * A), EigenMatrixNear(A, 1e-6));
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
    EXPECT_THAT(ref_K, EigenMatrixNear(K, 1e-6));
    EXPECT_THAT(cam_from_world.rotation.toRotationMatrix(),
                EigenMatrixNear(R, 1e-6));
    EXPECT_THAT(cam_from_world.translation, EigenMatrixNear(T, 1e-6));
  }
}

TEST(RotationMatrixToAngleAxis, Roundtrip) {
  const Eigen::Matrix3d R = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
  EXPECT_THAT(AngleAxisToRotationMatrix(RotationMatrixToAngleAxis(R)),
              EigenMatrixNear(R, 1e-6));
}

TEST(AngleAxisToRotationMatrix, Roundtrip) {
  const Eigen::AngleAxisd aa(Eigen::Quaterniond::UnitRandom());
  const Eigen::Vector3d w = aa.angle() * aa.axis();
  EXPECT_THAT(RotationMatrixToAngleAxis(AngleAxisToRotationMatrix(w)),
              EigenMatrixNear(w, 1e-6));
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
  EXPECT_THAT(AverageQuaternions(quats, weights).coeffs(),
              EigenMatrixNear(
                  Eigen::Quaterniond(0.92388, 0.382683, 0, 0).coeffs(), 1e-6));

  quats = {Eigen::Quaterniond::Identity(), Eigen::Quaterniond(1, 1, 0, 0)};
  weights = {1.0, 2.0};
  EXPECT_THAT(AverageQuaternions(quats, weights).coeffs(),
              EigenMatrixNear(
                  Eigen::Quaterniond(0.850651, 0.525731, 0, 0).coeffs(), 1e-6));
}

TEST(InterpolateCameraPoses, Nominal) {
  const Rigid3d cam_from_world1(Eigen::Quaterniond::UnitRandom(),
                                Eigen::Vector3d::Random());
  const Rigid3d cam_from_world2(Eigen::Quaterniond::UnitRandom(),
                                Eigen::Vector3d::Random());

  const Rigid3d interp_cam_from_world1 =
      InterpolateCameraPoses(cam_from_world1, cam_from_world2, 0);
  EXPECT_THAT(interp_cam_from_world1.translation,
              EigenMatrixNear(cam_from_world1.translation));

  const Rigid3d interp_cam_from_world2 =
      InterpolateCameraPoses(cam_from_world1, cam_from_world2, 1);
  EXPECT_THAT(interp_cam_from_world2.translation,
              EigenMatrixNear(cam_from_world2.translation));

  const Rigid3d interp_cam_from_world3 =
      InterpolateCameraPoses(cam_from_world1, cam_from_world2, 0.5);
  EXPECT_THAT(
      interp_cam_from_world3.translation,
      EigenMatrixNear(Eigen::Vector3d(
          (cam_from_world1.translation + cam_from_world2.translation) / 2)));
}

TEST(CheckCheirality, Nominal) {
  const Rigid3d cam2_from_cam1(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d(1, 0, 0));

  std::vector<Eigen::Vector3d> rays1;
  std::vector<Eigen::Vector3d> rays2;
  std::vector<Eigen::Vector3d> points3D;

  rays1.push_back(Eigen::Vector3d(0, 0, 1).normalized());
  rays2.push_back(Eigen::Vector3d(0.1, 0, 1).normalized());
  EXPECT_TRUE(CheckCheirality(cam2_from_cam1, rays1, rays2, &points3D));
  EXPECT_EQ(points3D.size(), 1);

  rays1.push_back(Eigen::Vector3d(0, 0, 1).normalized());
  rays2.push_back(Eigen::Vector3d(-0.1, 0, 1).normalized());
  EXPECT_TRUE(CheckCheirality(cam2_from_cam1, rays1, rays2, &points3D));
  EXPECT_EQ(points3D.size(), 1);

  rays2[1][0] = 0.2;
  EXPECT_TRUE(CheckCheirality(cam2_from_cam1, rays1, rays2, &points3D));
  EXPECT_EQ(points3D.size(), 2);

  rays2[0][0] = -0.2;
  rays2[1][0] = -0.2;
  EXPECT_FALSE(CheckCheirality(cam2_from_cam1, rays1, rays2, &points3D));
  EXPECT_EQ(points3D.size(), 0);
}

TEST(AverageUnitVectors, Nominal) {
  Eigen::Matrix3Xd vectors(3, 1);
  vectors.col(0) = Eigen::Vector3d::UnitX();
  Eigen::VectorXd avg = AverageUnitVectors(vectors);
  EXPECT_THAT(Eigen::Vector3d(avg),
              EigenMatrixNear(Eigen::Vector3d::UnitX().eval(), 1e-6));

  vectors.resize(3, 2);
  vectors.col(0) = Eigen::Vector3d::UnitZ();
  vectors.col(1) = Eigen::Vector3d::UnitZ();
  avg = AverageUnitVectors(vectors);
  EXPECT_THAT(Eigen::Vector3d(avg),
              EigenMatrixNear(Eigen::Vector3d::UnitZ().eval(), 1e-6));

  vectors.col(0) = Eigen::Vector3d::UnitX();
  vectors.col(1) = Eigen::Vector3d::UnitY();
  avg = AverageUnitVectors(vectors);
  EXPECT_EQ(avg.size(), 3);
  EXPECT_NEAR(avg.norm(), 1.0, 1e-6);

  std::vector<double> weights = {3.0, 1.0};
  avg = AverageUnitVectors(vectors, weights);
  EXPECT_GT(avg(0), avg(1));

  vectors.resize(3, 4);
  vectors.col(0) = Eigen::Vector3d(1, 0.1, 0).normalized();
  vectors.col(1) = Eigen::Vector3d(1, -0.1, 0).normalized();
  vectors.col(2) = Eigen::Vector3d(1, 0, 0.1).normalized();
  vectors.col(3) = Eigen::Vector3d(-1, 0, 0);
  avg = AverageUnitVectors(vectors);
  EXPECT_GT(avg(0), 0);

  Eigen::Matrix4Xd vectors4d(4, 2);
  vectors4d.col(0) = Eigen::Vector4d(1, 0, 0, 0);
  vectors4d.col(1) = Eigen::Vector4d(0, 1, 0, 0);
  avg = AverageUnitVectors(vectors4d);
  EXPECT_EQ(avg.size(), 4);
  EXPECT_NEAR(avg.norm(), 1.0, 1e-6);
}

TEST(AverageDirections, Nominal) {
  std::vector<Eigen::Vector3d> vectors = {Eigen::Vector3d(1, 0.1, 0),
                                          Eigen::Vector3d(1, -0.1, 0)};
  Eigen::Vector3d avg = AverageDirections(vectors);
  EXPECT_THAT(avg,
              EigenMatrixNear(Eigen::Vector3d(Eigen::Vector3d::UnitX()), 1e-6));

  std::vector<double> weights = {1.0, 1.0};
  avg = AverageDirections(vectors, weights);
  EXPECT_THAT(avg,
              EigenMatrixNear(Eigen::Vector3d(Eigen::Vector3d::UnitX()), 1e-6));
}

TEST(GravityAlignedRotation, Nominal) {
  Eigen::Vector3d gravity = Eigen::Vector3d(0.5, 0.5, 0.5).normalized();
  Eigen::Matrix3d R = GravityAlignedRotation(gravity);
  EXPECT_THAT(Eigen::Vector3d(R.col(1)), EigenMatrixNear(gravity, 1e-6));

  gravity = Eigen::Vector3d::UnitY();
  R = GravityAlignedRotation(gravity);
  EXPECT_THAT(Eigen::Vector3d(R.col(1)),
              EigenMatrixNear(Eigen::Vector3d(Eigen::Vector3d::UnitY()), 1e-6));
  EXPECT_NEAR(R.determinant(), 1.0, 1e-6);

  for (int i = 0; i < 100; ++i) {
    gravity = Eigen::Vector3d::Random().normalized();
    R = GravityAlignedRotation(gravity);
    EXPECT_THAT(
        Eigen::Matrix3d(R.transpose() * R),
        EigenMatrixNear(Eigen::Matrix3d(Eigen::Matrix3d::Identity()), 1e-6));
    EXPECT_NEAR(R.determinant(), 1.0, 1e-6);
  }
}

TEST(YAxisAngleFromRotation, Roundtrip) {
  for (int i = 0; i < 100; ++i) {
    const double angle =
        (static_cast<double>(rand()) / RAND_MAX - 0.5) * 2 * M_PI;
    const Eigen::Matrix3d R = RotationFromYAxisAngle(angle);
    const double recovered_angle = YAxisAngleFromRotation(R);
    EXPECT_THAT(RotationFromYAxisAngle(recovered_angle),
                EigenMatrixNear(R, 1e-6));
  }
}

TEST(RotationFromYAxisAngle, Nominal) {
  Eigen::Matrix3d R = RotationFromYAxisAngle(0);
  EXPECT_THAT(
      R, EigenMatrixNear(Eigen::Matrix3d(Eigen::Matrix3d::Identity()), 1e-6));
  EXPECT_NEAR(YAxisAngleFromRotation(R), 0.0, 1e-6);

  R = RotationFromYAxisAngle(M_PI / 2);
  EXPECT_THAT(
      Eigen::Vector3d(R * Eigen::Vector3d::UnitX()),
      EigenMatrixNear(Eigen::Vector3d(-Eigen::Vector3d::UnitZ()), 1e-6));
}

}  // namespace
}  // namespace colmap

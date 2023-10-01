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

#include "colmap/scene/projection.h"

#include "colmap/geometry/pose.h"
#include "colmap/math/math.h"
#include "colmap/sensor/models.h"

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(CalculateSquaredReprojectionError, Nominal) {
  const Rigid3d cam_from_world(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d::Zero());
  const Eigen::Matrix3x4d cam_from_world_mat = cam_from_world.ToMatrix();

  const Eigen::Vector3d point3D = Eigen::Vector3d::Random().cwiseAbs();
  const Eigen::Vector3d point2D_h = cam_from_world_mat * point3D.homogeneous();
  const Eigen::Vector2d point2D = point2D_h.hnormalized();

  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1, 0, 0);

  EXPECT_NEAR(CalculateSquaredReprojectionError(
                  point2D, point3D, cam_from_world, camera),
              0,
              1e-6);
  EXPECT_NEAR(CalculateSquaredReprojectionError(
                  point2D, point3D, cam_from_world_mat, camera),
              0,
              1e-6);

  EXPECT_NEAR(CalculateSquaredReprojectionError(
                  point2D.array() + 1, point3D, cam_from_world, camera),
              2,
              1e-6);
  EXPECT_NEAR(CalculateSquaredReprojectionError(
                  point2D.array() + 1, point3D, cam_from_world_mat, camera),
              2,
              1e-6);
}

TEST(CalculateAngularError, Nominal) {
  const Rigid3d cam_from_world(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d::Zero());
  const Eigen::Matrix3x4d cam_from_world_mat = cam_from_world.ToMatrix();

  Camera camera;
  camera.SetModelId(SimplePinholeCameraModel::model_id);
  camera.Params() = {1, 0, 0};

  const double error1 = CalculateAngularError(Eigen::Vector2d(0, 0),
                                              Eigen::Vector3d(0, 0, 1),
                                              cam_from_world_mat,
                                              camera);
  EXPECT_NEAR(error1, 0, 1e-6);

  const double error2 = CalculateAngularError(Eigen::Vector2d(0, 0),
                                              Eigen::Vector3d(0, 1, 1),
                                              cam_from_world_mat,
                                              camera);
  EXPECT_NEAR(error2, M_PI / 4, 1e-6);

  const double error3 = CalculateAngularError(Eigen::Vector2d(0, 0),
                                              Eigen::Vector3d(0, 5, 5),
                                              cam_from_world_mat,
                                              camera);
  EXPECT_NEAR(error3, M_PI / 4, 1e-6);

  const double error4 = CalculateAngularError(Eigen::Vector2d(1, 0),
                                              Eigen::Vector3d(0, 0, 1),
                                              cam_from_world_mat,
                                              camera);
  EXPECT_NEAR(error4, M_PI / 4, 1e-6);

  const double error5 = CalculateAngularError(Eigen::Vector2d(2, 0),
                                              Eigen::Vector3d(0, 0, 1),
                                              cam_from_world_mat,
                                              camera);
  EXPECT_NEAR(error5, 1.10714872, 1e-6);

  const double error6 = CalculateAngularError(Eigen::Vector2d(2, 0),
                                              Eigen::Vector3d(1, 0, 1),
                                              cam_from_world_mat,
                                              camera);
  EXPECT_NEAR(error6, 1.10714872 - M_PI / 4, 1e-6);

  const double error7 = CalculateAngularError(Eigen::Vector2d(2, 0),
                                              Eigen::Vector3d(5, 0, 5),
                                              cam_from_world_mat,
                                              camera);
  EXPECT_NEAR(error7, 1.10714872 - M_PI / 4, 1e-6);

  const double error8 = CalculateAngularError(Eigen::Vector2d(1, 0),
                                              Eigen::Vector3d(-1, 0, 1),
                                              cam_from_world_mat,
                                              camera);
  EXPECT_NEAR(error8, M_PI / 2, 1e-6);

  const double error9 = CalculateAngularError(Eigen::Vector2d(1, 0),
                                              Eigen::Vector3d(-1, 0, 0),
                                              cam_from_world_mat,
                                              camera);
  EXPECT_NEAR(error9, M_PI * 3 / 4, 1e-6);

  const double error10 = CalculateAngularError(Eigen::Vector2d(1, 0),
                                               Eigen::Vector3d(-1, 0, -1),
                                               cam_from_world_mat,
                                               camera);
  EXPECT_NEAR(error10, M_PI, 1e-6);

  const double error11 = CalculateAngularError(Eigen::Vector2d(1, 0),
                                               Eigen::Vector3d(0, 0, -1),
                                               cam_from_world_mat,
                                               camera);
  EXPECT_NEAR(error11, M_PI * 3 / 4, 1e-6);
}

TEST(HasPointPositiveDepth, Nominal) {
  const Rigid3d cam_from_world(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d::Zero());
  const Eigen::Matrix3x4d cam_from_world_mat = cam_from_world.ToMatrix();

  // In the image plane
  const bool check1 =
      HasPointPositiveDepth(cam_from_world_mat, Eigen::Vector3d(0, 0, 0));
  EXPECT_FALSE(check1);
  const bool check2 =
      HasPointPositiveDepth(cam_from_world_mat, Eigen::Vector3d(0, 2, 0));
  EXPECT_FALSE(check2);

  // Infront of camera
  const bool check3 =
      HasPointPositiveDepth(cam_from_world_mat, Eigen::Vector3d(0, 0, 1));
  EXPECT_TRUE(check3);

  // Behind camera
  const bool check4 =
      HasPointPositiveDepth(cam_from_world_mat, Eigen::Vector3d(0, 0, -1));
  EXPECT_FALSE(check4);
}

}  // namespace
}  // namespace colmap

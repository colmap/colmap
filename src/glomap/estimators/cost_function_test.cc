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

#include "glomap/estimators/cost_function.h"

#include "colmap/util/eigen_matchers.h"

#include <gtest/gtest.h>

namespace glomap {
namespace {

TEST(BATAPairwiseDirectionCostFunctor, ZeroResidual) {
  const Eigen::Vector3d pos1(1, 2, 3);
  const Eigen::Vector3d pos2(2, 3, 4);
  const double scale = 1.0;
  const Eigen::Vector3d direction = pos2 - pos1;

  BATAPairwiseDirectionCostFunctor cost_function(direction);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(
      cost_function(pos1.data(), pos2.data(), &scale, residuals.data()));

  EXPECT_THAT(residuals,
              colmap::EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-10));
}

TEST(BATAPairwiseDirectionCostFunctor, NonZeroResidual) {
  const Eigen::Vector3d pos1(1, 2, 3);
  const Eigen::Vector3d pos2(4, 5, 6);
  const double scale = 2.0;
  const Eigen::Vector3d direction(1, 1, 1);

  BATAPairwiseDirectionCostFunctor cost_function(direction);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(
      cost_function(pos1.data(), pos2.data(), &scale, residuals.data()));

  const Eigen::Vector3d expected_residuals = direction - scale * (pos2 - pos1);
  EXPECT_THAT(residuals, colmap::EigenMatrixNear(expected_residuals, 1e-10));
}

TEST(BATAPairwiseDirectionCostFunctor, DifferentScale) {
  const Eigen::Vector3d pos1(1, 2, 3);
  const Eigen::Vector3d pos2(2, 4, 6);
  const double scale = 0.5;
  const Eigen::Vector3d direction = scale * (pos2 - pos1);

  BATAPairwiseDirectionCostFunctor cost_function(direction);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(
      cost_function(pos1.data(), pos2.data(), &scale, residuals.data()));

  EXPECT_THAT(residuals,
              colmap::EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-10));
}

TEST(BATAPairwiseDirectionCostFunctor, CreateCostFunction) {
  const Eigen::Vector3d direction(1, 0, 0);
  std::unique_ptr<ceres::CostFunction> cost_function(
      BATAPairwiseDirectionCostFunctor::Create(direction));
  ASSERT_NE(cost_function, nullptr);
}

TEST(RigBATAPairwiseDirectionCostFunctor, ZeroResidual) {
  const Eigen::Vector3d point3D(1, 2, 3);
  const Eigen::Vector3d rig_in_world(3, 2, 1);
  const double scale = 1.5;
  const double rig_scale = 2.0;
  const Eigen::Vector3d cam_from_rig_dir(0.25, 0.5, 0.75);
  const Eigen::Vector3d cam_from_point3D_dir =
      scale * (point3D - rig_in_world + rig_scale * cam_from_rig_dir);

  RigBATAPairwiseDirectionCostFunctor cost_function(cam_from_point3D_dir,
                                                    cam_from_rig_dir);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_function(point3D.data(),
                            rig_in_world.data(),
                            &scale,
                            &rig_scale,
                            residuals.data()));

  EXPECT_THAT(residuals,
              colmap::EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-10));
}

TEST(RigBATAPairwiseDirectionCostFunctor, NonZeroResidual) {
  const Eigen::Vector3d point3D(3, 4, 5);
  const Eigen::Vector3d rig_in_world(1, 2, 3);
  const double scale = 2.0;
  const double rig_scale = 1.5;
  const Eigen::Vector3d cam_from_rig_dir(0.1, 0.2, 0.3);
  const Eigen::Vector3d cam_from_point3D_dir(1, 1, 1);

  RigBATAPairwiseDirectionCostFunctor cost_function(cam_from_point3D_dir,
                                                    cam_from_rig_dir);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_function(point3D.data(),
                            rig_in_world.data(),
                            &scale,
                            &rig_scale,
                            residuals.data()));

  const Eigen::Vector3d expected_residuals =
      cam_from_point3D_dir -
      scale * (point3D - rig_in_world + rig_scale * cam_from_rig_dir);
  EXPECT_THAT(residuals, colmap::EigenMatrixNear(expected_residuals, 1e-10));
}

TEST(RigBATAPairwiseDirectionCostFunctor, CreateCostFunction) {
  const Eigen::Vector3d cam_from_point3D_dir(1, 0, 0);
  const Eigen::Vector3d cam_from_rig_dir(0, 1, 0);
  std::unique_ptr<ceres::CostFunction> cost_function(
      RigBATAPairwiseDirectionCostFunctor::Create(cam_from_point3D_dir,
                                                  cam_from_rig_dir));
  ASSERT_NE(cost_function, nullptr);
}

TEST(RigUnknownBATAPairwiseDirectionCostFunctor, ZeroResidual) {
  const Eigen::Vector3d point3D(5, 5, 5);
  const Eigen::Vector3d rig_in_world(1, 1, 1);
  const Eigen::Vector3d cam_in_rig(0.5, 0.5, 0.5);
  const double scale = 1.0;
  const Eigen::Quaterniond rig_from_world_rot = Eigen::Quaterniond::Identity();
  const Eigen::Vector3d cam_from_rig_dir =
      rig_from_world_rot.inverse() * cam_in_rig;
  const Eigen::Vector3d cam_from_point3D_dir =
      scale * (point3D - rig_in_world - cam_from_rig_dir);

  RigUnknownBATAPairwiseDirectionCostFunctor cost_function(cam_from_point3D_dir,
                                                           rig_from_world_rot);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_function(point3D.data(),
                            rig_in_world.data(),
                            cam_in_rig.data(),
                            &scale,
                            residuals.data()));

  EXPECT_THAT(residuals,
              colmap::EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-10));
}

TEST(RigUnknownBATAPairwiseDirectionCostFunctor, NonZeroResidual) {
  const Eigen::Vector3d point3D(3, 4, 5);
  const Eigen::Vector3d rig_in_world(1, 2, 3);
  const Eigen::Vector3d cam_in_rig(0.2, 0.3, 0.4);
  const double scale = 2.0;
  const Eigen::Quaterniond rig_from_world_rot =
      Eigen::Quaterniond(0.707, 0.707, 0, 0).normalized();
  const Eigen::Vector3d cam_from_point3D_dir(1, 1, 1);

  RigUnknownBATAPairwiseDirectionCostFunctor cost_function(cam_from_point3D_dir,
                                                           rig_from_world_rot);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_function(point3D.data(),
                            rig_in_world.data(),
                            cam_in_rig.data(),
                            &scale,
                            residuals.data()));

  const Eigen::Vector3d cam_from_rig_dir =
      rig_from_world_rot.toRotationMatrix().transpose() * cam_in_rig;
  const Eigen::Vector3d expected_residuals =
      cam_from_point3D_dir -
      scale * (point3D - rig_in_world - cam_from_rig_dir);
  EXPECT_THAT(residuals, colmap::EigenMatrixNear(expected_residuals, 1e-10));
}

TEST(RigUnknownBATAPairwiseDirectionCostFunctor, CreateCostFunction) {
  const Eigen::Vector3d cam_from_point3D_dir(1, 0, 0);
  const Eigen::Quaterniond rig_from_world_rot = Eigen::Quaterniond::Identity();
  std::unique_ptr<ceres::CostFunction> cost_function(
      RigUnknownBATAPairwiseDirectionCostFunctor::Create(cam_from_point3D_dir,
                                                         rig_from_world_rot));
  ASSERT_NE(cost_function, nullptr);
}

// TODO(jsch): Add meaningful tests for FetzerFocalLengthCostFunctor.

TEST(FetzerFocalLengthCostFunctor, CreateCostFunction) {
  Eigen::Matrix3d F;
  F << 0, 0, 0.1, 0, 0, 0.2, -0.1, -0.2, 0;
  const Eigen::Vector2d pp0(320, 240);
  const Eigen::Vector2d pp1(320, 240);

  std::unique_ptr<ceres::CostFunction> cost_function(
      FetzerFocalLengthCostFunctor::Create(F, pp0, pp1));
  ASSERT_NE(cost_function, nullptr);
}

// TODO(jsch): Add meaningful tests for FetzerFocalLengthSameCameraCostFunctor.

TEST(FetzerFocalLengthSameCameraCostFunctor, CreateCostFunction) {
  Eigen::Matrix3d F;
  F << 0, 0, 0.1, 0, 0, 0.2, -0.1, -0.2, 0;
  const Eigen::Vector2d pp(320, 240);

  std::unique_ptr<ceres::CostFunction> cost_function(
      FetzerFocalLengthSameCameraCostFunctor::Create(F, pp));
  ASSERT_NE(cost_function, nullptr);
}

TEST(GravityCostFunctor, ZeroResidual) {
  const Eigen::Vector3d measured_gravity =
      Eigen::Vector3d::Random().normalized();
  const Eigen::Vector3d& gravity = measured_gravity;

  GravityCostFunctor cost_function(measured_gravity);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_function(gravity.data(), residuals.data()));

  EXPECT_THAT(residuals,
              colmap::EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-10));
}

TEST(GravityCostFunctor, NonZeroResidual) {
  const Eigen::Vector3d measured_gravity =
      Eigen::Vector3d::Random().normalized();
  const Eigen::Vector3d gravity = Eigen::Vector3d::Random().normalized();

  GravityCostFunctor cost_function(measured_gravity);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_function(gravity.data(), residuals.data()));

  const Eigen::Vector3d expected_residuals = gravity - measured_gravity;
  EXPECT_THAT(residuals, colmap::EigenMatrixNear(expected_residuals, 1e-10));
}

TEST(GravityCostFunctor, CreateCostFunction) {
  const Eigen::Vector3d measured_gravity(0, 0, 1);
  std::unique_ptr<ceres::CostFunction> cost_function(
      GravityCostFunctor::Create(measured_gravity));
  ASSERT_NE(cost_function, nullptr);
}

}  // namespace
}  // namespace glomap

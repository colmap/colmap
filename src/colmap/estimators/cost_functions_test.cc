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

#include "colmap/estimators/cost_functions.h"

#include "colmap/geometry/pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/sensor/models.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(BundleAdjustment, AbsolutePose) {
  using CostFunction = ReprojErrorCostFunction<SimplePinholeCameraModel>;
  std::unique_ptr<ceres::CostFunction> cost_function(
      CostFunction::Create(Eigen::Vector2d::Zero()));
  double cam_from_world_rotation[4] = {0, 0, 0, 1};
  double cam_from_world_translation[3] = {0, 0, 0};
  double point3D[3] = {0, 0, 1};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[4] = {cam_from_world_rotation,
                                 cam_from_world_translation,
                                 point3D,
                                 camera_params};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);

  point3D[1] = 1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 1);

  camera_params[0] = 2;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 2);

  point3D[0] = -1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], -2);
  EXPECT_EQ(residuals[1], 2);

  std::unique_ptr<ceres::CostFunction> cost_function_with_noise(
      IsotropicNoiseCostFunctionWrapper<CostFunction>::Create(
          2.0, Eigen::Vector2d::Zero()));
  EXPECT_TRUE(
      cost_function_with_noise->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], -1);
  EXPECT_EQ(residuals[1], 1);
}

TEST(BundleAdjustment, ConstantPoseAbsolutePose) {
  Rigid3d cam_from_world;
  std::unique_ptr<ceres::CostFunction> cost_function(
      ReprojErrorConstantPoseCostFunction<SimplePinholeCameraModel>::Create(
          cam_from_world, Eigen::Vector2d::Zero()));
  double point3D[3] = {0, 0, 1};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[2] = {point3D, camera_params};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);

  point3D[1] = 1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 1);

  camera_params[0] = 2;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 2);

  point3D[0] = -1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], -2);
  EXPECT_EQ(residuals[1], 2);
}

TEST(BundleAdjustment, ConstantPoint3DAbsolutePose) {
  Eigen::Vector2d point2D = Eigen::Vector2d::Zero();
  Eigen::Vector3d point3D;
  point3D << 0, 0, 1;

  double cam_from_world_rotation[4] = {0, 0, 0, 1};
  double cam_from_world_translation[3] = {0, 0, 0};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[3] = {
      cam_from_world_rotation, cam_from_world_translation, camera_params};

  {
    std::unique_ptr<ceres::CostFunction> cost_function(
        ReprojErrorConstantPoint3DCostFunction<
            SimplePinholeCameraModel>::Create(point2D, point3D));
    EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
    EXPECT_EQ(residuals[0], 0);
    EXPECT_EQ(residuals[1], 0);
  }

  {
    point3D[1] = 1;
    std::unique_ptr<ceres::CostFunction> cost_function(
        ReprojErrorConstantPoint3DCostFunction<
            SimplePinholeCameraModel>::Create(point2D, point3D));
    EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
    EXPECT_EQ(residuals[0], 0);
    EXPECT_EQ(residuals[1], 1);
  }

  {
    camera_params[0] = 2;
    std::unique_ptr<ceres::CostFunction> cost_function(
        ReprojErrorConstantPoint3DCostFunction<
            SimplePinholeCameraModel>::Create(point2D, point3D));
    EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
    EXPECT_EQ(residuals[0], 0);
    EXPECT_EQ(residuals[1], 2);
  }

  {
    point3D[0] = -1;
    std::unique_ptr<ceres::CostFunction> cost_function(
        ReprojErrorConstantPoint3DCostFunction<
            SimplePinholeCameraModel>::Create(point2D, point3D));
    EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
    EXPECT_EQ(residuals[0], -2);
    EXPECT_EQ(residuals[1], 2);
  }
}

TEST(BundleAdjustment, Rig) {
  std::unique_ptr<ceres::CostFunction> cost_function(
      RigReprojErrorCostFunction<SimplePinholeCameraModel>::Create(
          Eigen::Vector2d::Zero()));
  double cam_from_rig_rotation[4] = {0, 0, 0, 1};
  double cam_from_rig_translation[3] = {0, 0, -1};
  double rig_from_world_rotation[4] = {0, 0, 0, 1};
  double rig_from_world_translation[3] = {0, 0, 1};
  double point3D[3] = {0, 0, 1};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[6] = {cam_from_rig_rotation,
                                 cam_from_rig_translation,
                                 rig_from_world_rotation,
                                 rig_from_world_translation,
                                 point3D,
                                 camera_params};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);

  point3D[1] = 1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 1);

  camera_params[0] = 2;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 2);

  point3D[0] = -1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], -2);
  EXPECT_EQ(residuals[1], 2);
}

TEST(BundleAdjustment, ConstantRig) {
  Rigid3d cam_from_rig;
  cam_from_rig.translation << 0, 0, -1;
  std::unique_ptr<ceres::CostFunction> cost_function(
      RigReprojErrorConstantRigCostFunction<SimplePinholeCameraModel>::Create(
          cam_from_rig, Eigen::Vector2d::Zero()));

  double rig_from_world_rotation[4] = {0, 0, 0, 1};
  double rig_from_world_translation[3] = {0, 0, 1};
  double point3D[3] = {0, 0, 1};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[4] = {rig_from_world_rotation,
                                 rig_from_world_translation,
                                 point3D,
                                 camera_params};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);

  point3D[1] = 1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 1);

  camera_params[0] = 2;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 2);

  point3D[0] = -1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], -2);
  EXPECT_EQ(residuals[1], 2);
}

TEST(BundleAdjustment, RelativePose) {
  std::unique_ptr<ceres::CostFunction> cost_function(
      SampsonErrorCostFunction::Create(Eigen::Vector2d(0, 0),
                                       Eigen::Vector2d(0, 0)));
  double cam_from_world_rotation[4] = {1, 0, 0, 0};
  double cam_from_world_translation[3] = {0, 1, 0};
  double residuals[1];
  const double* parameters[2] = {cam_from_world_rotation,
                                 cam_from_world_translation};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);

  cost_function.reset(SampsonErrorCostFunction::Create(Eigen::Vector2d(0, 0),
                                                       Eigen::Vector2d(1, 0)));
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0.5);

  cost_function.reset(SampsonErrorCostFunction::Create(Eigen::Vector2d(0, 0),
                                                       Eigen::Vector2d(1, 1)));
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0.5);
}

TEST(PoseGraphOptimization, AbsolutePose) {
  const Rigid3d mes_cam_from_world;
  EigenMatrix6d covariance_cam = EigenMatrix6d::Identity();
  covariance_cam(5, 5) = 4;
  std::unique_ptr<ceres::CostFunction> cost_function(
      AbsolutePoseErrorCostFunction::Create(mes_cam_from_world,
                                            covariance_cam));

  double cam_from_world_rotation[4] = {0, 0, 0, 1};
  double cam_from_world_translation[3] = {0, 0, 0};
  double residuals[6];
  const double* parameters[2] = {cam_from_world_rotation,
                                 cam_from_world_translation};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  EXPECT_EQ(residuals[2], 0);
  EXPECT_EQ(residuals[3], 0);
  EXPECT_EQ(residuals[4], 0);
  EXPECT_EQ(residuals[5], 0);

  cam_from_world_translation[0] = 1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  EXPECT_EQ(residuals[2], 0);
  EXPECT_EQ(residuals[3], 1);
  EXPECT_EQ(residuals[4], 0);
  EXPECT_EQ(residuals[5], 0);

  // Rotation by 90 degrees around the Y axis.
  Eigen::Matrix3d rotation_matrix;
  rotation_matrix << 0, 0, 1, 0, 1, 0, -1, 0, 0;
  Eigen::Map<Eigen::Quaterniond>(
      static_cast<double*>(cam_from_world_rotation)) = rotation_matrix;
  cam_from_world_translation[1] = 2;
  cam_from_world_translation[2] = 3;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_NEAR(residuals[0], 0, 1e-6);
  EXPECT_NEAR(residuals[1], DegToRad(90.0), 1e-6);
  EXPECT_NEAR(residuals[2], 0, 1e-6);
  EXPECT_NEAR(residuals[3], 1, 1e-6);
  EXPECT_NEAR(residuals[4], 2, 1e-6);
  EXPECT_NEAR(residuals[5], 1.5, 1e-6);
}

TEST(PoseGraphOptimization, RelativePose) {
  Rigid3d i_from_j;
  i_from_j.translation << 0, 0, -1;
  EigenMatrix6d covariance_j = EigenMatrix6d::Identity();
  covariance_j(5, 5) = 4;
  std::unique_ptr<ceres::CostFunction> cost_function(
      MetricRelativePoseErrorCostFunction::Create(i_from_j, covariance_j));

  double i_from_world_rotation[4] = {0, 0, 0, 1};
  double i_from_world_translation[3] = {0, 0, 0};
  double j_from_world_rotation[4] = {0, 0, 0, 1};
  double j_from_world_translation[3] = {0, 0, 1};
  double residuals[6];
  const double* parameters[4] = {i_from_world_rotation,
                                 i_from_world_translation,
                                 j_from_world_rotation,
                                 j_from_world_translation};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  EXPECT_EQ(residuals[2], 0);
  EXPECT_EQ(residuals[3], 0);
  EXPECT_EQ(residuals[4], 0);
  EXPECT_EQ(residuals[5], 0);

  i_from_world_translation[2] = 4;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  EXPECT_EQ(residuals[2], 0);
  EXPECT_EQ(residuals[3], 0);
  EXPECT_EQ(residuals[4], 0);
  EXPECT_EQ(residuals[5], -2);

  j_from_world_translation[0] = 2;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  EXPECT_EQ(residuals[2], 0);
  EXPECT_EQ(residuals[3], 2);
  EXPECT_EQ(residuals[4], 0);
  EXPECT_EQ(residuals[5], -2);

  // Rotation by 90 degrees around the Y axis.
  Eigen::Matrix3d rotation_matrix;
  rotation_matrix << 0, 0, 1, 0, 1, 0, -1, 0, 0;
  Eigen::Map<Eigen::Quaterniond>(static_cast<double*>(j_from_world_rotation)) =
      rotation_matrix;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_NEAR(residuals[0], 0, 1e-6);
  EXPECT_NEAR(residuals[1], DegToRad(90.0), 1e-6);
  EXPECT_NEAR(residuals[2], 0, 1e-6);
  EXPECT_NEAR(residuals[3], -3, 1e-6);
  EXPECT_NEAR(residuals[4], 0, 1e-6);
  EXPECT_NEAR(residuals[5], 0.5, 1e-6);
}

}  // namespace
}  // namespace colmap

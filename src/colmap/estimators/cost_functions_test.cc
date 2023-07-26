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

#include "colmap/estimators/cost_functions.h"

#include "colmap/geometry/pose.h"
#include "colmap/sensor/models.h"

#include <gtest/gtest.h>

namespace colmap {

TEST(BundleAdjustment, AbsolutePose) {
  std::unique_ptr<ceres::CostFunction> cost_function(
      ReprojErrorCostFunction<SimplePinholeCameraModel>::Create(
          Eigen::Vector2d::Zero()));
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
}

TEST(BundleAdjustment, ConstantAbsolutePose) {
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

}  // namespace colmap

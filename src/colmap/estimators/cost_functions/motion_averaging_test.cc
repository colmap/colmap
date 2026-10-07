// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/cost_functions/motion_averaging.h"

#include "colmap/util/eigen_matchers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(RelativeRotationCostFunctor, Create) {
  const Eigen::Quaterniond sensor1_from_world(
      Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()));
  const Eigen::Quaterniond sensor2_from_world(
      Eigen::AngleAxisd(-0.4, Eigen::Vector3d::UnitY()));
  const Eigen::Quaterniond sensor2_from_sensor1 =
      sensor2_from_world * sensor1_from_world.conjugate();
  std::unique_ptr<ceres::CostFunction> cost_function(
      RelativeRotationCostFunctor::Create(sensor2_from_sensor1));
  const double* parameters[] = {sensor1_from_world.coeffs().data(),
                                sensor2_from_world.coeffs().data()};
  Eigen::Vector3d residuals;
  ASSERT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-12));

  parameters[1] = parameters[0];
  ASSERT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_NEAR(
      residuals.norm(), Eigen::AngleAxisd(sensor2_from_sensor1).angle(), 1e-12);
}

TEST(RelativeRotationCostFunctor, RigParameterBlocks) {
  const Eigen::Quaterniond rig1_from_world(
      Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()));
  const Eigen::Quaterniond rig2_from_world(
      Eigen::AngleAxisd(-0.4, Eigen::Vector3d::UnitY()));
  const Eigen::Quaterniond sensor1_from_rig(
      Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()));
  const Eigen::Quaterniond sensor2_from_rig(
      Eigen::AngleAxisd(-0.1, Eigen::Vector3d::UnitX()));
  const double* parameters[] = {rig1_from_world.coeffs().data(),
                                rig2_from_world.coeffs().data(),
                                sensor1_from_rig.coeffs().data(),
                                sensor2_from_rig.coeffs().data()};
  for (const bool same_frame : {false, true}) {
    const Eigen::Quaterniond sensor1_from_world =
        sensor1_from_rig * rig1_from_world;
    const Eigen::Quaterniond sensor2_from_world =
        sensor2_from_rig * (same_frame ? rig1_from_world : rig2_from_world);
    const int sensor_offset = same_frame ? 0 : 2;
    const RelativeRotationCostFunctor cost_functor{
        sensor2_from_world * sensor1_from_world.conjugate(),
        sensor_offset,
        sensor_offset + 1,
        same_frame};
    Eigen::Vector3d residuals;
    ASSERT_TRUE(cost_functor(same_frame ? parameters + 2 : parameters,
                             residuals.data()));
    EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-12));
  }
}

TEST(BATAPairwiseDirectionCostFunctor, ZeroResidual) {
  const Eigen::Vector3d pos1(1, 2, 3);
  const Eigen::Vector3d pos2(2, 3, 4);
  const double scale = 1.0;
  const Eigen::Vector3d direction = pos2 - pos1;

  BATAPairwiseDirectionCostFunctor cost_functor(direction);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_functor(pos1.data(), pos2.data(), &scale, residuals.data()));

  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-10));
}

TEST(BATAPairwiseDirectionCostFunctor, NonZeroResidual) {
  const Eigen::Vector3d pos1(1, 2, 3);
  const Eigen::Vector3d pos2(4, 5, 6);
  const double scale = 2.0;
  const Eigen::Vector3d direction(1, 1, 1);

  BATAPairwiseDirectionCostFunctor cost_functor(direction);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_functor(pos1.data(), pos2.data(), &scale, residuals.data()));

  const Eigen::Vector3d expected_residuals = direction - scale * (pos2 - pos1);
  EXPECT_THAT(residuals, EigenMatrixNear(expected_residuals, 1e-10));
}

TEST(BATAPairwiseDirectionCostFunctor, DifferentScale) {
  const Eigen::Vector3d pos1(1, 2, 3);
  const Eigen::Vector3d pos2(2, 4, 6);
  const double scale = 0.5;
  const Eigen::Vector3d direction = scale * (pos2 - pos1);

  BATAPairwiseDirectionCostFunctor cost_functor(direction);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_functor(pos1.data(), pos2.data(), &scale, residuals.data()));

  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-10));
}

TEST(BATAPairwiseDirectionCostFunctor, Create) {
  const Eigen::Vector3d direction(1, 0, 0);
  std::unique_ptr<ceres::CostFunction> cost_function(
      BATAPairwiseDirectionCostFunctor::Create(direction));
  ASSERT_NE(cost_function, nullptr);
}

TEST(RigBATAPairwiseDirectionConstantRigCostFunctor, ZeroResidual) {
  const Eigen::Vector3d point3D(1, 2, 3);
  const Eigen::Vector3d rig_in_world(3, 2, 1);
  const double scale = 1.5;
  const Eigen::Vector3d cam_from_rig_dir(0.25, 0.5, 0.75);
  const Eigen::Vector3d cam_from_point3D_dir =
      scale * (point3D - rig_in_world + cam_from_rig_dir);

  RigBATAPairwiseDirectionConstantRigCostFunctor cost_functor(
      cam_from_point3D_dir, cam_from_rig_dir);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_functor(
      point3D.data(), rig_in_world.data(), &scale, residuals.data()));

  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-10));
}

TEST(RigBATAPairwiseDirectionConstantRigCostFunctor, NonZeroResidual) {
  const Eigen::Vector3d point3D(3, 4, 5);
  const Eigen::Vector3d rig_in_world(1, 2, 3);
  const double scale = 2.0;
  const Eigen::Vector3d cam_from_rig_dir(0.1, 0.2, 0.3);
  const Eigen::Vector3d cam_from_point3D_dir(1, 1, 1);

  RigBATAPairwiseDirectionConstantRigCostFunctor cost_functor(
      cam_from_point3D_dir, cam_from_rig_dir);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_functor(
      point3D.data(), rig_in_world.data(), &scale, residuals.data()));

  const Eigen::Vector3d expected_residuals =
      cam_from_point3D_dir -
      scale * (point3D - rig_in_world + cam_from_rig_dir);
  EXPECT_THAT(residuals, EigenMatrixNear(expected_residuals, 1e-10));
}

TEST(RigBATAPairwiseDirectionConstantRigCostFunctor, Create) {
  const Eigen::Vector3d cam_from_point3D_dir(1, 0, 0);
  const Eigen::Vector3d cam_from_rig_dir(0, 1, 0);
  std::unique_ptr<ceres::CostFunction> cost_function(
      RigBATAPairwiseDirectionConstantRigCostFunctor::Create(
          cam_from_point3D_dir, cam_from_rig_dir));
  ASSERT_NE(cost_function, nullptr);
}

TEST(RigBATAPairwiseDirectionCostFunctor, ZeroResidual) {
  const Eigen::Vector3d point3D(5, 5, 5);
  const Eigen::Vector3d rig_in_world(1, 1, 1);
  const Eigen::Vector3d cam_in_rig(0.5, 0.5, 0.5);
  const double scale = 1.0;
  const Eigen::Quaterniond rig_from_world_rot = Eigen::Quaterniond::Identity();
  const Eigen::Vector3d cam_from_rig_dir =
      rig_from_world_rot.inverse() * cam_in_rig;
  const Eigen::Vector3d cam_from_point3D_dir =
      scale * (point3D - rig_in_world - cam_from_rig_dir);

  RigBATAPairwiseDirectionCostFunctor cost_functor(cam_from_point3D_dir,
                                                   rig_from_world_rot);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_functor(point3D.data(),
                           rig_in_world.data(),
                           cam_in_rig.data(),
                           &scale,
                           residuals.data()));

  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-10));
}

TEST(RigBATAPairwiseDirectionCostFunctor, NonZeroResidual) {
  const Eigen::Vector3d point3D(3, 4, 5);
  const Eigen::Vector3d rig_in_world(1, 2, 3);
  const Eigen::Vector3d cam_in_rig(0.2, 0.3, 0.4);
  const double scale = 2.0;
  const Eigen::Quaterniond rig_from_world_rot =
      Eigen::Quaterniond(0.707, 0.707, 0, 0).normalized();
  const Eigen::Vector3d cam_from_point3D_dir(1, 1, 1);

  RigBATAPairwiseDirectionCostFunctor cost_functor(cam_from_point3D_dir,
                                                   rig_from_world_rot);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_functor(point3D.data(),
                           rig_in_world.data(),
                           cam_in_rig.data(),
                           &scale,
                           residuals.data()));

  const Eigen::Vector3d cam_from_rig_dir =
      rig_from_world_rot.toRotationMatrix().transpose() * cam_in_rig;
  const Eigen::Vector3d expected_residuals =
      cam_from_point3D_dir -
      scale * (point3D - rig_in_world - cam_from_rig_dir);
  EXPECT_THAT(residuals, EigenMatrixNear(expected_residuals, 1e-10));
}

TEST(RigBATAPairwiseDirectionCostFunctor, Create) {
  const Eigen::Vector3d cam_from_point3D_dir(1, 0, 0);
  const Eigen::Quaterniond rig_from_world_rot = Eigen::Quaterniond::Identity();
  std::unique_ptr<ceres::CostFunction> cost_function(
      RigBATAPairwiseDirectionCostFunctor::Create(cam_from_point3D_dir,
                                                  rig_from_world_rot));
  ASSERT_NE(cost_function, nullptr);
}

}  // namespace
}  // namespace colmap

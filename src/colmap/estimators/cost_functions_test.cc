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
#include "colmap/geometry/sim3.h"
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/sensor/models.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(ReprojErrorCostFunctor, Nominal) {
  using CostFunctor = ReprojErrorCostFunctor<SimplePinholeCameraModel>;
  std::unique_ptr<ceres::CostFunction> cost_function(
      CostFunctor::Create(Eigen::Vector2d::Zero()));
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

TEST(ReprojErrorConstantPoseCostFunctor, Nominal) {
  Rigid3d cam_from_world;
  std::unique_ptr<ceres::CostFunction> cost_function(
      ReprojErrorConstantPoseCostFunctor<SimplePinholeCameraModel>::Create(
          Eigen::Vector2d::Zero(), cam_from_world));
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

TEST(ReprojErrorConstantPoint3DCostFunctor, Nominal) {
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
        ReprojErrorConstantPoint3DCostFunctor<SimplePinholeCameraModel>::Create(
            point2D, point3D));
    EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
    EXPECT_EQ(residuals[0], 0);
    EXPECT_EQ(residuals[1], 0);
  }

  {
    point3D[1] = 1;
    std::unique_ptr<ceres::CostFunction> cost_function(
        ReprojErrorConstantPoint3DCostFunctor<SimplePinholeCameraModel>::Create(
            point2D, point3D));
    EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
    EXPECT_EQ(residuals[0], 0);
    EXPECT_EQ(residuals[1], 1);
  }

  {
    camera_params[0] = 2;
    std::unique_ptr<ceres::CostFunction> cost_function(
        ReprojErrorConstantPoint3DCostFunctor<SimplePinholeCameraModel>::Create(
            point2D, point3D));
    EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
    EXPECT_EQ(residuals[0], 0);
    EXPECT_EQ(residuals[1], 2);
  }

  {
    point3D[0] = -1;
    std::unique_ptr<ceres::CostFunction> cost_function(
        ReprojErrorConstantPoint3DCostFunctor<SimplePinholeCameraModel>::Create(
            point2D, point3D));
    EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
    EXPECT_EQ(residuals[0], -2);
    EXPECT_EQ(residuals[1], 2);
  }
}

TEST(RigReprojErrorCostFunctor, Nominal) {
  std::unique_ptr<ceres::CostFunction> cost_function(
      RigReprojErrorCostFunctor<SimplePinholeCameraModel>::Create(
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

TEST(RigReprojErrorConstantRigCostFunctor, Nominal) {
  Rigid3d cam_from_rig;
  cam_from_rig.translation << 0, 0, -1;
  std::unique_ptr<ceres::CostFunction> cost_function(
      RigReprojErrorConstantRigCostFunctor<SimplePinholeCameraModel>::Create(
          Eigen::Vector2d::Zero(), cam_from_rig));

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

TEST(SampsonErrorCostFunctor, Nominal) {
  std::unique_ptr<ceres::CostFunction> cost_function(
      SampsonErrorCostFunctor::Create(Eigen::Vector2d(0, 0),
                                      Eigen::Vector2d(0, 0)));
  double cam_from_world_rotation[4] = {1, 0, 0, 0};
  double cam_from_world_translation[3] = {0, 1, 0};
  double residuals[1];
  const double* parameters[2] = {cam_from_world_rotation,
                                 cam_from_world_translation};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);

  cost_function.reset(SampsonErrorCostFunctor::Create(Eigen::Vector2d(0, 0),
                                                      Eigen::Vector2d(1, 0)));
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0.5);

  cost_function.reset(SampsonErrorCostFunctor::Create(Eigen::Vector2d(0, 0),
                                                      Eigen::Vector2d(1, 1)));
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0.5);
}

TEST(AbsolutePosePositionPriorCostFunctor, Nominal) {
  std::unique_ptr<ceres::CostFunction> cost_function(
      AbsolutePosePositionPriorCostFunctor::Create(Eigen::Vector3d(0, 0, 0)));

  double cam_from_world_rotation[4] = {0, 0, 0, 1};
  double cam_from_world_translation[3] = {0, 0, 0};

  double residuals[3];
  const double* parameters[2] = {cam_from_world_rotation,
                                 cam_from_world_translation};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  EXPECT_EQ(residuals[2], 0);

  const Rigid3d cam_from_world(Eigen::Quaterniond::UnitRandom(),
                               Eigen::Vector3d::Random());
  const Rigid3d world_from_cam = Inverse(cam_from_world);

  cost_function.reset(
      AbsolutePosePositionPriorCostFunctor::Create(Eigen::Vector3d::Zero()));
  parameters[0] = cam_from_world.rotation.coeffs().data();
  parameters[1] = cam_from_world.translation.data();
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_NEAR(residuals[0], -world_from_cam.translation[0], 1e-6);
  EXPECT_NEAR(residuals[1], -world_from_cam.translation[1], 1e-6);
  EXPECT_NEAR(residuals[2], -world_from_cam.translation[2], 1e-6);

  cost_function.reset(
      AbsolutePosePositionPriorCostFunctor::Create(world_from_cam.translation));
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  EXPECT_EQ(residuals[2], 0);
}

TEST(AbsolutePosePriorCostFunctor, Nominal) {
  const Rigid3d cam_from_world_prior;
  std::unique_ptr<ceres::CostFunction> cost_function(
      AbsolutePosePriorCostFunctor::Create(cam_from_world_prior));

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
  EXPECT_NEAR(residuals[5], 3, 1e-6);
}

TEST(RelativePosePriorCostFunctor, Nominal) {
  Rigid3d i_from_j_prior(Eigen::Quaterniond::Identity(),
                         Eigen::Vector3d(0, 0, -1));
  std::unique_ptr<ceres::CostFunction> cost_function(
      RelativePosePriorCostFunctor::Create(i_from_j_prior));

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
  EXPECT_EQ(residuals[5], 4);

  j_from_world_translation[0] = 2;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  EXPECT_EQ(residuals[2], 0);
  EXPECT_EQ(residuals[3], -2);
  EXPECT_EQ(residuals[4], 0);
  EXPECT_EQ(residuals[5], 4);

  // Rotation by 90 degrees around the Y axis.
  Eigen::Matrix3d rotation_matrix;
  rotation_matrix << 0, 0, 1, 0, 1, 0, -1, 0, 0;
  Eigen::Map<Eigen::Quaterniond>(static_cast<double*>(j_from_world_rotation)) =
      rotation_matrix;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_NEAR(residuals[0], 0, 1e-6);
  EXPECT_NEAR(residuals[1], DegToRad(-90.0), 1e-6);
  EXPECT_NEAR(residuals[2], 0, 1e-6);
  EXPECT_NEAR(residuals[3], 0, 1e-6);
  EXPECT_NEAR(residuals[4], 0, 1e-6);
  EXPECT_NEAR(residuals[5], 2, 1e-6);
}

TEST(Point3DAlignmentCostFunctor, Nominal) {
  // generate a test transformation
  Sim3d tform = Sim3d(RandomUniformReal<double>(0.1, 10),
                      Eigen::Quaterniond::UnitRandom(),
                      Eigen::Vector3d::Random());
  // construct cost function and evaluate
  Eigen::Vector3d point_in_b_prior(1., 1., 1.);
  std::unique_ptr<ceres::CostFunction> cost_function(
      Point3DAlignmentCostFunctor::Create(point_in_b_prior));
  Eigen::Vector3d point(0., 0., 0.);
  const double* parameters[4] = {point.data(),
                                 tform.rotation.coeffs().data(),
                                 tform.translation.data(),
                                 &tform.scale};
  double residuals[3];
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));

  // test with reference computation from C++
  Eigen::Vector3d error = tform * point - point_in_b_prior;
  EXPECT_NEAR(residuals[0], error[0], 1e-6);
  EXPECT_NEAR(residuals[1], error[1], 1e-6);
  EXPECT_NEAR(residuals[2], error[2], 1e-6);
}

TEST(CovarianceWeightedCostFunctor, ReprojErrorCostFunctor) {
  using CostFunctor = ReprojErrorCostFunctor<SimplePinholeCameraModel>;
  double cam_from_world_rotation[4] = {0, 0, 0, 1};
  double cam_from_world_translation[3] = {0, 0, 0};
  double point3D[3] = {-1, 1, 1};
  double camera_params[3] = {2, 0, 0};
  double residuals[2];
  const double* parameters[4] = {cam_from_world_rotation,
                                 cam_from_world_translation,
                                 point3D,
                                 camera_params};

  std::unique_ptr<ceres::CostFunction> cost_function1(
      CovarianceWeightedCostFunctor<CostFunctor>::Create(
          Eigen::Matrix2d::Identity(), Eigen::Vector2d::Zero()));
  EXPECT_TRUE(cost_function1->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], -2);
  EXPECT_EQ(residuals[1], 2);

  std::unique_ptr<ceres::CostFunction> cost_function2(
      CovarianceWeightedCostFunctor<CostFunctor>::Create(
          4 * Eigen::Matrix2d::Identity(), Eigen::Vector2d::Zero()));
  EXPECT_TRUE(cost_function2->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], -1);
  EXPECT_EQ(residuals[1], 1);
}

TEST(CovarianceWeightedCostFunctor, AbsolutePosePositionPriorCostFunctor) {
  const Rigid3d cam_from_world(Eigen::Quaterniond::UnitRandom(),
                               Eigen::Vector3d::Random());
  const Rigid3d world_from_cam = Inverse(cam_from_world);

  double residuals[3];
  const double* parameters[2] = {cam_from_world.rotation.coeffs().data(),
                                 cam_from_world.translation.data()};

  std::unique_ptr<ceres::CostFunction> cost_function(
      CovarianceWeightedCostFunctor<AbsolutePosePositionPriorCostFunctor>::
          Create(2 * Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 0)));
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_NEAR(
      residuals[0], -0.5 * std::sqrt(2) * world_from_cam.translation[0], 1e-6);
  EXPECT_NEAR(
      residuals[1], -0.5 * std::sqrt(2) * world_from_cam.translation[1], 1e-6);
  EXPECT_NEAR(
      residuals[2], -0.5 * std::sqrt(2) * world_from_cam.translation[2], 1e-6);
}

}  // namespace
}  // namespace colmap

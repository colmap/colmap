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

#include "colmap/estimators/cost_functions/reprojection_error.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/math/random.h"
#include "colmap/sensor/models.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(ReprojErrorCostFunctor, Nominal) {
  using CostFunctor = ReprojErrorCostFunctor<SimplePinholeCameraModel>;
  std::unique_ptr<ceres::CostFunction> cost_function(
      CostFunctor::Create(Eigen::Vector2d::Zero()));
  double cam_from_world[7] = {0, 0, 0, 1, 0, 0, 0};
  double point3D[3] = {0, 0, 1};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[3] = {point3D, cam_from_world, camera_params};
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

  point3D[2] = -1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
}

TEST(ReprojErrorCostFunctor, AnalyticalVersusAutoDiff) {
  SetPRNGSeed(42);

  const Eigen::Vector2d kPoint2D(200, 300);
  auto analytical_cost_function = std::make_unique<
      AnalyticalReprojErrorCostFunction<SimpleRadialCameraModel>>(kPoint2D);
  std::unique_ptr<ceres::CostFunction> auto_diff_cost_function(
      ReprojErrorCostFunctor<SimpleRadialCameraModel>::Create(kPoint2D));

  for (const double x : {-1, 0, 1}) {
    for (const double y : {-1, 0, 1}) {
      for (const double z : {0, 1, 2, 3}) {
        Rigid3d cam_from_world(Eigen::Quaterniond(Eigen::AngleAxisd(
                                   RandomUniformReal<double>(0, 2 * EIGEN_PI),
                                   Eigen::Vector3d(0.1, -0.1, 1).normalized())),
                               Eigen::Vector3d(1, 2, 3));
        Eigen::Vector3d point3D(x, y, z);
        std::vector<double> simple_radial_params = {200, 100, 120, 0.1};

        // Ensure point is in front of camera.
        ASSERT_GT((cam_from_world * point3D).z(), 0);

        std::vector<double*> parameter_blocks{point3D.data(),
                                              cam_from_world.params.data(),
                                              simple_radial_params.data()};

        constexpr double kEps = 1e-9;

        Eigen::Vector2d auto_diff_residuals;
        EXPECT_TRUE(auto_diff_cost_function->Evaluate(
            parameter_blocks.data(), auto_diff_residuals.data(), nullptr));

        ceres::NumericDiffOptions numeric_diff_options;
        ceres::GradientChecker gradient_checker(
            analytical_cost_function.get(), nullptr, numeric_diff_options);
        ceres::GradientChecker::ProbeResults results;
        EXPECT_TRUE(
            gradient_checker.Probe(parameter_blocks.data(), kEps, &results));
        EXPECT_NEAR(results.residuals[0], auto_diff_residuals[0], kEps);
        EXPECT_NEAR(results.residuals[1], auto_diff_residuals[1], kEps);
      }
    }
  }
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

  double cam_from_world[7] = {0, 0, 0, 1, 0, 0, 0};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[2] = {cam_from_world, camera_params};

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

TEST(CovarianceWeightedCostFunctor, ReprojErrorCostFunctor) {
  using CostFunctor = ReprojErrorCostFunctor<SimplePinholeCameraModel>;
  double cam_from_world[7] = {0, 0, 0, 1, 0, 0, 0};
  double point3D[3] = {-1, 1, 1};
  double camera_params[3] = {2, 0, 0};
  double residuals[2];
  const double* parameters[3] = {point3D, cam_from_world, camera_params};

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

TEST(RigReprojErrorCostFunctor, Nominal) {
  std::unique_ptr<ceres::CostFunction> cost_function(
      RigReprojErrorCostFunctor<SimplePinholeCameraModel>::Create(
          Eigen::Vector2d::Zero()));
  double cam_from_rig[7] = {0, 0, 0, 1, 0, 0, -1};
  double rig_from_world[7] = {0, 0, 0, 1, 0, 0, 1};
  double point3D[3] = {0, 0, 1};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[4] = {
      point3D, cam_from_rig, rig_from_world, camera_params};
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
  cam_from_rig.translation() << 0, 0, -1;
  std::unique_ptr<ceres::CostFunction> cost_function(
      RigReprojErrorConstantRigCostFunctor<SimplePinholeCameraModel>::Create(
          Eigen::Vector2d::Zero(), cam_from_rig));

  double rig_from_world[7] = {0, 0, 0, 1, 0, 0, 1};
  double point3D[3] = {0, 0, 1};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[3] = {point3D, rig_from_world, camera_params};
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

}  // namespace
}  // namespace colmap

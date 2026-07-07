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

// Helper that constructs a Ceres GradientChecker across Ceres versions.
ceres::GradientChecker MakeGradientChecker(
    ceres::CostFunction* cost_function,
    const ceres::NumericDiffOptions& numeric_diff_options) {
  return ceres::GradientChecker(
      cost_function,
#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
      static_cast<const std::vector<const ceres::Manifold*>*>(nullptr),
#else
      static_cast<const std::vector<const ceres::LocalParameterization*>*>(
          nullptr),
#endif
      numeric_diff_options);
}

// Validates the fully-variable analytical reprojection error cost function
// against both the numeric Jacobian and the autodiff residual. The numeric
// (Ridders) Jacobian comparison uses a looser tolerance than the residual
// comparison, since finite differences on higher-order distortion models do
// not reach full double precision; the exact analytical Jacobians are verified
// separately against Ceres Jets in sensor/models_jacobian_test.
template <typename CameraModel>
void TestAnalyticalReprojError(const std::vector<double>& camera_params) {
  SetPRNGSeed(42);

  constexpr double kJacEps = 1e-4;
  constexpr double kResEps = 1e-9;

  const Eigen::Vector2d kPoint2D(200, 300);
  auto analytical_cost_function =
      std::make_unique<AnalyticalReprojErrorCostFunction<CameraModel>>(
          kPoint2D);
  std::unique_ptr<ceres::CostFunction> auto_diff_cost_function(
      ReprojErrorCostFunctor<CameraModel>::Create(kPoint2D));

  for (const double x : {-1, 0, 1}) {
    for (const double y : {-1, 0, 1}) {
      for (const double z : {0, 1, 2, 3}) {
        Rigid3d cam_from_world(Eigen::Quaterniond(Eigen::AngleAxisd(
                                   RandomUniformReal<double>(0, 2 * EIGEN_PI),
                                   Eigen::Vector3d(0.1, -0.1, 1).normalized())),
                               Eigen::Vector3d(1, 2, 3));
        Eigen::Vector3d point3D(x, y, z);
        std::vector<double> params = camera_params;

        // Restrict to a realistic field of view. The OpenCV/rational models are
        // only well-conditioned for moderate normalized radii, where the
        // finite-difference (Ridders) gradient check is reliable; the exact
        // analytical Jacobians are validated over the full range against Ceres
        // Jets in sensor/models_jacobian_test.
        const Eigen::Vector3d point3D_in_cam = cam_from_world * point3D;
        if (point3D_in_cam.z() < 0.5 ||
            point3D_in_cam.hnormalized().norm() > 0.8) {
          continue;
        }

        std::vector<double*> parameter_blocks{
            point3D.data(), cam_from_world.params.data(), params.data()};

        Eigen::Vector2d auto_diff_residuals;
        EXPECT_TRUE(auto_diff_cost_function->Evaluate(
            parameter_blocks.data(), auto_diff_residuals.data(), nullptr));

        ceres::NumericDiffOptions numeric_diff_options;
        ceres::GradientChecker gradient_checker = MakeGradientChecker(
            analytical_cost_function.get(), numeric_diff_options);
        ceres::GradientChecker::ProbeResults results;
        EXPECT_TRUE(
            gradient_checker.Probe(parameter_blocks.data(), kJacEps, &results))
            << results.error_log;
        EXPECT_NEAR(results.residuals[0], auto_diff_residuals[0], kResEps);
        EXPECT_NEAR(results.residuals[1], auto_diff_residuals[1], kResEps);
      }
    }
  }
}

// Validates the fixed-pose analytical reprojection error cost function against
// both the numeric Jacobian and the autodiff residual. See
// TestAnalyticalReprojError for the tolerance rationale.
template <typename CameraModel>
void TestAnalyticalReprojErrorConstantPose(
    const std::vector<double>& camera_params) {
  SetPRNGSeed(42);

  constexpr double kJacEps = 1e-4;
  constexpr double kResEps = 1e-9;

  const Eigen::Vector2d kPoint2D(200, 300);
  for (const double x : {-1, 0, 1}) {
    for (const double y : {-1, 0, 1}) {
      for (const double z : {0, 1, 2, 3}) {
        Rigid3d cam_from_world(Eigen::Quaterniond(Eigen::AngleAxisd(
                                   RandomUniformReal<double>(0, 2 * EIGEN_PI),
                                   Eigen::Vector3d(0.1, -0.1, 1).normalized())),
                               Eigen::Vector3d(1, 2, 3));
        Eigen::Vector3d point3D(x, y, z);
        std::vector<double> params = camera_params;

        // Restrict to a realistic field of view. The OpenCV/rational models are
        // only well-conditioned for moderate normalized radii, where the
        // finite-difference (Ridders) gradient check is reliable; the exact
        // analytical Jacobians are validated over the full range against Ceres
        // Jets in sensor/models_jacobian_test.
        const Eigen::Vector3d point3D_in_cam = cam_from_world * point3D;
        if (point3D_in_cam.z() < 0.5 ||
            point3D_in_cam.hnormalized().norm() > 0.8) {
          continue;
        }

        auto analytical_cost_function = std::make_unique<
            AnalyticalReprojErrorConstantPoseCostFunction<CameraModel>>(
            kPoint2D, cam_from_world);
        std::unique_ptr<ceres::CostFunction> auto_diff_cost_function(
            ReprojErrorConstantPoseCostFunctor<CameraModel>::Create(
                kPoint2D, cam_from_world));

        std::vector<double*> parameter_blocks{point3D.data(), params.data()};

        Eigen::Vector2d auto_diff_residuals;
        EXPECT_TRUE(auto_diff_cost_function->Evaluate(
            parameter_blocks.data(), auto_diff_residuals.data(), nullptr));

        ceres::NumericDiffOptions numeric_diff_options;
        ceres::GradientChecker gradient_checker = MakeGradientChecker(
            analytical_cost_function.get(), numeric_diff_options);
        ceres::GradientChecker::ProbeResults results;
        EXPECT_TRUE(
            gradient_checker.Probe(parameter_blocks.data(), kJacEps, &results))
            << results.error_log;
        EXPECT_NEAR(results.residuals[0], auto_diff_residuals[0], kResEps);
        EXPECT_NEAR(results.residuals[1], auto_diff_residuals[1], kResEps);
      }
    }
  }
}

TEST(ReprojErrorCostFunctor, AnalyticalVersusAutoDiff) {
  TestAnalyticalReprojError<SimplePinholeCameraModel>({200, 100, 120});
  TestAnalyticalReprojError<PinholeCameraModel>({200, 210, 100, 120});
  TestAnalyticalReprojError<SimpleRadialCameraModel>({200, 100, 120, 0.1});
  TestAnalyticalReprojError<RadialCameraModel>({200, 100, 120, 0.1, 0.05});
  TestAnalyticalReprojError<OpenCVCameraModel>(
      {200, 210, 100, 120, -0.1, 0.05, -0.001, 0.002});
  TestAnalyticalReprojError<FullOpenCVCameraModel>(
      {200, 210, 100, 120, -0.1, 0.05, -0.001, 0.002, 0.01, 0.02, -0.02, 0.01});
  TestAnalyticalReprojError<FOVCameraModel>({200, 210, 100, 120, 0.9});
  TestAnalyticalReprojError<SimpleRadialFisheyeCameraModel>(
      {200, 100, 120, 0.1});
  TestAnalyticalReprojError<RadialFisheyeCameraModel>(
      {200, 100, 120, 0.1, 0.02});
  TestAnalyticalReprojError<OpenCVFisheyeCameraModel>(
      {200, 210, 100, 120, -0.05, 0.02, -0.001, 0.001});
  TestAnalyticalReprojError<ThinPrismFisheyeCameraModel>({200,
                                                          210,
                                                          100,
                                                          120,
                                                          -0.05,
                                                          0.02,
                                                          -0.001,
                                                          0.001,
                                                          0.001,
                                                          0.002,
                                                          0.001,
                                                          -0.001});
  TestAnalyticalReprojError<RadTanThinPrismFisheyeModel>({200,
                                                          210,
                                                          100,
                                                          120,
                                                          -0.05,
                                                          0.02,
                                                          -0.005,
                                                          0.001,
                                                          0.0005,
                                                          0.0002,
                                                          -0.001,
                                                          0.001,
                                                          0.001,
                                                          -0.001,
                                                          0.0005,
                                                          -0.0005});
  TestAnalyticalReprojError<SimpleFisheyeCameraModel>({200, 100, 120});
  TestAnalyticalReprojError<FisheyeCameraModel>({200, 210, 100, 120});
  TestAnalyticalReprojError<SimpleDivisionCameraModel>({200, 100, 120, 0.1});
  TestAnalyticalReprojError<DivisionCameraModel>({200, 210, 100, 120, 0.1});
  TestAnalyticalReprojError<EUCMCameraModel>({200, 210, 100, 120, 0.6, 1.2});
  TestAnalyticalReprojError<EquirectangularCameraModel>({1000, 500});
}

TEST(ReprojErrorConstantPoseCostFunctor, AnalyticalVersusAutoDiff) {
  TestAnalyticalReprojErrorConstantPose<SimplePinholeCameraModel>(
      {200, 100, 120});
  TestAnalyticalReprojErrorConstantPose<PinholeCameraModel>(
      {200, 210, 100, 120});
  TestAnalyticalReprojErrorConstantPose<SimpleRadialCameraModel>(
      {200, 100, 120, 0.1});
  TestAnalyticalReprojErrorConstantPose<RadialCameraModel>(
      {200, 100, 120, 0.1, 0.05});
  TestAnalyticalReprojErrorConstantPose<OpenCVCameraModel>(
      {200, 210, 100, 120, -0.1, 0.05, -0.001, 0.002});
  TestAnalyticalReprojErrorConstantPose<FullOpenCVCameraModel>(
      {200, 210, 100, 120, -0.1, 0.05, -0.001, 0.002, 0.01, 0.02, -0.02, 0.01});
  TestAnalyticalReprojErrorConstantPose<FOVCameraModel>(
      {200, 210, 100, 120, 0.9});
  TestAnalyticalReprojErrorConstantPose<SimpleRadialFisheyeCameraModel>(
      {200, 100, 120, 0.1});
  TestAnalyticalReprojErrorConstantPose<RadialFisheyeCameraModel>(
      {200, 100, 120, 0.1, 0.02});
  TestAnalyticalReprojErrorConstantPose<OpenCVFisheyeCameraModel>(
      {200, 210, 100, 120, -0.05, 0.02, -0.001, 0.001});
  TestAnalyticalReprojErrorConstantPose<ThinPrismFisheyeCameraModel>({200,
                                                                      210,
                                                                      100,
                                                                      120,
                                                                      -0.05,
                                                                      0.02,
                                                                      -0.001,
                                                                      0.001,
                                                                      0.001,
                                                                      0.002,
                                                                      0.001,
                                                                      -0.001});
  TestAnalyticalReprojErrorConstantPose<RadTanThinPrismFisheyeModel>({200,
                                                                      210,
                                                                      100,
                                                                      120,
                                                                      -0.05,
                                                                      0.02,
                                                                      -0.005,
                                                                      0.001,
                                                                      0.0005,
                                                                      0.0002,
                                                                      -0.001,
                                                                      0.001,
                                                                      0.001,
                                                                      -0.001,
                                                                      0.0005,
                                                                      -0.0005});
  TestAnalyticalReprojErrorConstantPose<SimpleFisheyeCameraModel>(
      {200, 100, 120});
  TestAnalyticalReprojErrorConstantPose<FisheyeCameraModel>(
      {200, 210, 100, 120});
  TestAnalyticalReprojErrorConstantPose<SimpleDivisionCameraModel>(
      {200, 100, 120, 0.1});
  TestAnalyticalReprojErrorConstantPose<DivisionCameraModel>(
      {200, 210, 100, 120, 0.1});
  TestAnalyticalReprojErrorConstantPose<EUCMCameraModel>(
      {200, 210, 100, 120, 0.6, 1.2});
  TestAnalyticalReprojErrorConstantPose<EquirectangularCameraModel>(
      {1000, 500});
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

constexpr double kEquirectangularCameraWidth = 1000;
constexpr double kEquirectangularCameraHeight = 500;

TEST(WrapEquirectangularHorizontalSeam, Nominal) {
  const double camera_params[2] = {kEquirectangularCameraWidth,
                                   kEquirectangularCameraHeight};

  // No-op for non-periodic models, regardless of residual magnitude.
  {
    double residuals[2] = {993.0, -7.0};
    WrapEquirectangularHorizontalSeam<SimplePinholeCameraModel>(camera_params,
                                                                residuals);
    EXPECT_EQ(residuals[0], 993.0);
    EXPECT_EQ(residuals[1], -7.0);
  }

  // Equirectangular folds the x-residual into [-w/2, w/2); y is untouched.
  {
    double residuals[2] = {1000.0, 5.0};  // Exactly one period -> 0.
    WrapEquirectangularHorizontalSeam<EquirectangularCameraModel>(camera_params,
                                                                  residuals);
    EXPECT_EQ(residuals[0], 0.0);
    EXPECT_EQ(residuals[1], 5.0);
  }
  {
    double residuals[2] = {998.0, 0.0};  // Just under a period -> -2.
    WrapEquirectangularHorizontalSeam<EquirectangularCameraModel>(camera_params,
                                                                  residuals);
    EXPECT_EQ(residuals[0], -2.0);
  }
  {
    double residuals[2] = {-998.0, 0.0};
    WrapEquirectangularHorizontalSeam<EquirectangularCameraModel>(camera_params,
                                                                  residuals);
    EXPECT_EQ(residuals[0], 2.0);
  }
  {
    double residuals[2] = {-3.0, 0.0};  // Already minimal -> unchanged.
    WrapEquirectangularHorizontalSeam<EquirectangularCameraModel>(camera_params,
                                                                  residuals);
    EXPECT_EQ(residuals[0], -3.0);
  }
  {
    double residuals[2] = {500.0, 0.0};  // +w/2 boundary folds to -w/2.
    WrapEquirectangularHorizontalSeam<EquirectangularCameraModel>(camera_params,
                                                                  residuals);
    EXPECT_EQ(residuals[0], -500.0);
  }
}

template <typename CreateCostFunction>
void ExpectEquirectangularSeamWrap(
    const CreateCostFunction& create,
    const std::vector<const double*>& parameters) {
  std::unique_ptr<ceres::CostFunction> cost_function(
      create(Eigen::Vector2d(2, kEquirectangularCameraHeight / 2)));
  double residuals[2];
  EXPECT_TRUE(cost_function->Evaluate(parameters.data(), residuals, nullptr));
  EXPECT_EQ(residuals[0], -2);  // Raw ~w folded into [-w/2, w/2).
  EXPECT_EQ(residuals[1], 0);
}

TEST(ReprojErrorCostFunctor, EquirectangularSeamWrap) {
  double cam_from_world[7] = {0, 0, 0, 1, 0, 0, 0};
  double point3D[3] = {0, 0, -1};  // Azimuth = ±π -> projects to x = w.
  double camera_params[2] = {kEquirectangularCameraWidth,
                             kEquirectangularCameraHeight};
  ExpectEquirectangularSeamWrap(
      [](const Eigen::Vector2d& point2D) {
        return ReprojErrorCostFunctor<EquirectangularCameraModel>::Create(
            point2D);
      },
      {point3D, cam_from_world, camera_params});
}

TEST(ReprojErrorConstantPoseCostFunctor, EquirectangularSeamWrap) {
  const Rigid3d cam_from_world;
  double point3D[3] = {0, 0, -1};
  double camera_params[2] = {kEquirectangularCameraWidth,
                             kEquirectangularCameraHeight};
  ExpectEquirectangularSeamWrap(
      [&cam_from_world](const Eigen::Vector2d& point2D) {
        return ReprojErrorConstantPoseCostFunctor<
            EquirectangularCameraModel>::Create(point2D, cam_from_world);
      },
      {point3D, camera_params});
}

TEST(ReprojErrorConstantPoint3DCostFunctor, EquirectangularSeamWrap) {
  const Eigen::Vector3d point3D(0, 0, -1);
  double cam_from_world[7] = {0, 0, 0, 1, 0, 0, 0};
  double camera_params[2] = {kEquirectangularCameraWidth,
                             kEquirectangularCameraHeight};
  ExpectEquirectangularSeamWrap(
      [&point3D](const Eigen::Vector2d& point2D) {
        return ReprojErrorConstantPoint3DCostFunctor<
            EquirectangularCameraModel>::Create(point2D, point3D);
      },
      {cam_from_world, camera_params});
}

TEST(RigReprojErrorCostFunctor, EquirectangularSeamWrap) {
  double cam_from_rig[7] = {0, 0, 0, 1, 0, 0, 0};
  double rig_from_world[7] = {0, 0, 0, 1, 0, 0, 0};
  double point3D[3] = {0, 0, -1};
  double camera_params[2] = {kEquirectangularCameraWidth,
                             kEquirectangularCameraHeight};
  ExpectEquirectangularSeamWrap(
      [](const Eigen::Vector2d& point2D) {
        return RigReprojErrorCostFunctor<EquirectangularCameraModel>::Create(
            point2D);
      },
      {point3D, cam_from_rig, rig_from_world, camera_params});
}

TEST(RigReprojErrorConstantRigCostFunctor, EquirectangularSeamWrap) {
  const Rigid3d cam_from_rig;
  double rig_from_world[7] = {0, 0, 0, 1, 0, 0, 0};
  double point3D[3] = {0, 0, -1};
  double camera_params[2] = {kEquirectangularCameraWidth,
                             kEquirectangularCameraHeight};
  ExpectEquirectangularSeamWrap(
      [&cam_from_rig](const Eigen::Vector2d& point2D) {
        return RigReprojErrorConstantRigCostFunctor<
            EquirectangularCameraModel>::Create(point2D, cam_from_rig);
      },
      {point3D, rig_from_world, camera_params});
}

}  // namespace
}  // namespace colmap

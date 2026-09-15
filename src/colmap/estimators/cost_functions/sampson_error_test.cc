// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/cost_functions/sampson_error.h"

#include "colmap/util/eigen_matchers.h"

#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(SampsonErrorCostFunctor, Nominal) {
  std::unique_ptr<ceres::CostFunction> cost_function(
      SampsonErrorCostFunctor::Create(Eigen::Vector2d(0, 0),
                                      Eigen::Vector2d(0, 0)));
  double cam_from_world[7] = {0, 0, 0, 1, 0, 1, 0};
  double residuals[1];
  const double* parameters[1] = {cam_from_world};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);

  cost_function.reset(SampsonErrorCostFunctor::Create(Eigen::Vector2d(0, 0),
                                                      Eigen::Vector2d(1, 0)));
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_NEAR(residuals[0] * residuals[0], 0.5, 1e-6);

  cost_function.reset(SampsonErrorCostFunctor::Create(Eigen::Vector2d(0, 0),
                                                      Eigen::Vector2d(1, 1)));
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_NEAR(residuals[0] * residuals[0], 0.5, 1e-6);
}

TEST(RelPoseParamsFromRigid3d, RoundTrip) {
  const Rigid3d pose(Eigen::Quaterniond(Eigen::AngleAxisd(
                         0.7, Eigen::Vector3d(0.2, -1, 0.5).normalized())),
                     Eigen::Vector3d(0.6, -0.3, 1.0));
  const RelPoseParams params = RelPoseParamsFromRigid3d(pose);
  const Eigen::Vector4d quat_coeffs = pose.rotation().normalized().coeffs();
  const Eigen::Vector3d unit_translation = pose.translation().normalized();
  EXPECT_THAT(Eigen::Vector4d(params.head<4>()),
              EigenMatrixNear(quat_coeffs, 1e-12));
  EXPECT_THAT(Eigen::Vector3d(params.tail<3>()),
              EigenMatrixNear(unit_translation, 1e-12));

  const Rigid3d round_trip = Rigid3dFromRelPoseParams(params.data());
  EXPECT_THAT(Eigen::Vector4d(round_trip.rotation().coeffs()),
              EigenMatrixNear(quat_coeffs, 1e-12));
  EXPECT_THAT(Eigen::Vector3d(round_trip.translation()),
              EigenMatrixNear(unit_translation, 1e-12));
}
}  // namespace
}  // namespace colmap

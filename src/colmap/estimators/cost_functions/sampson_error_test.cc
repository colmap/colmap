// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/cost_functions/sampson_error.h"

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

}  // namespace
}  // namespace colmap

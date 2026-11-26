// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/ceres_loss_function.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <ceres/ceres.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(CeresLossFunction, GemanMcClureValueAndDerivatives) {
  const double a = 0.5;
  const double c = a * a;
  auto loss = CreateCeresLossFunction(CeresLossFunctionType::GEMAN_MCCLURE, a);
  ASSERT_NE(loss, nullptr);

  double rho0[3] = {};
  loss->Evaluate(0.0, rho0);
  EXPECT_DOUBLE_EQ(rho0[0], 0.0);
  EXPECT_DOUBLE_EQ(rho0[1], 1.0);
  EXPECT_DOUBLE_EQ(rho0[2], -2.0 / c);

  for (const double s : {0.01, 0.1, 0.25, 1.0, 3.5, 100.0}) {
    double rho[3] = {};
    loss->Evaluate(s, rho);
    const double expected_val = (c * s) / (c + s);
    EXPECT_NEAR(rho[0], expected_val, 1e-12);
    EXPECT_LT(rho[0], c);
    EXPECT_GT(rho[1], 0.0);
    EXPECT_LT(rho[2], 0.0);

    const double eps = 1e-6 * std::max(1.0, s);
    double rp[3] = {};
    double rm[3] = {};
    loss->Evaluate(s + eps, rp);
    loss->Evaluate(s - eps, rm);
    EXPECT_NEAR(rho[1], (rp[0] - rm[0]) / (2.0 * eps), 1e-6 * std::abs(rho[1]));
    EXPECT_NEAR(rho[2], (rp[1] - rm[1]) / (2.0 * eps), 1e-6 * std::abs(rho[2]));
  }
}

TEST(CeresLossFunction, GemanMcClureWeightScaling) {
  const double a = 0.5;
  const double w = 2.5;
  auto base = CreateCeresLossFunction(CeresLossFunctionType::GEMAN_MCCLURE, a);
  auto scaled =
      CreateCeresLossFunction(CeresLossFunctionType::GEMAN_MCCLURE, a, w);
  for (const double s : {0.0, 0.2, 2.0}) {
    double r_base[3] = {};
    double r_scaled[3] = {};
    base->Evaluate(s, r_base);
    scaled->Evaluate(s, r_scaled);
    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR(r_scaled[i], w * r_base[i], 1e-12);
    }
  }
}

TEST(CeresLossFunction, GemanMcClureValidationAndEnumConversion) {
  EXPECT_EQ(CeresLossFunctionTypeFromString("GEMAN_MCCLURE"),
            CeresLossFunctionType::GEMAN_MCCLURE);
  EXPECT_EQ(CeresLossFunctionTypeToString(CeresLossFunctionType::GEMAN_MCCLURE),
            "GEMAN_MCCLURE");

  EXPECT_TRUE(
      IsValidCeresLossFunction(CeresLossFunctionType::GEMAN_MCCLURE, 0.5));
  for (const double bad_scale : {0.0,
                                 -0.5,
                                 1e-200,
                                 1e-154,
                                 1e200,
                                 std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_FALSE(IsValidCeresLossFunction(CeresLossFunctionType::GEMAN_MCCLURE,
                                          bad_scale));
    EXPECT_THROW(CreateCeresLossFunction(CeresLossFunctionType::GEMAN_MCCLURE,
                                         bad_scale),
                 std::invalid_argument);
  }
  for (const double bad_weight : {0.0,
                                  -1.0,
                                  std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_FALSE(IsValidCeresLossFunction(
        CeresLossFunctionType::GEMAN_MCCLURE, 0.5, bad_weight));
    EXPECT_THROW(CreateCeresLossFunction(
                     CeresLossFunctionType::GEMAN_MCCLURE, 0.5, bad_weight),
                 std::invalid_argument);
  }
}

struct ScalarResidual {
  explicit ScalarResidual(const double target) : target(target) {}

  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    residual[0] = x[0] - T(target);
    return true;
  }

  const double target;
};

TEST(CeresLossFunction, GemanMcClureSuppressesOutliersInCeresSolve) {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);
  double x = 0.0;
  auto loss =
      CreateCeresLossFunction(CeresLossFunctionType::GEMAN_MCCLURE, 1.0);
  for (const double target : {0.9, 1.0, 1.0, 1.1, 100.0}) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<ScalarResidual, 1, 1>(
            new ScalarResidual(target)),
        loss.get(),
        &x);
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  EXPECT_EQ(summary.termination_type, ceres::CONVERGENCE);
  EXPECT_NEAR(x, 1.0, 1e-2);
}

TEST(CeresLossFunction, GemanMcClureExtremeResidualsAndScales) {
  {
    const double a = 0.5;
    const double c = a * a;
    auto loss =
        CreateCeresLossFunction(CeresLossFunctionType::GEMAN_MCCLURE, a);
    double rho[3] = {};
    loss->Evaluate(std::numeric_limits<double>::infinity(), rho);
    EXPECT_DOUBLE_EQ(rho[0], c);
    EXPECT_DOUBLE_EQ(rho[1], 0.0);
    EXPECT_DOUBLE_EQ(rho[2], 0.0);
  }
  {
    const double a = 1e154;
    const double c = a * a;
    auto loss =
        CreateCeresLossFunction(CeresLossFunctionType::GEMAN_MCCLURE, a);
    double rho_eq[3] = {};
    loss->Evaluate(c, rho_eq);
    EXPECT_TRUE(std::isfinite(rho_eq[0]));
    EXPECT_NEAR(rho_eq[0] / c, 0.5, 1e-12);
    EXPECT_NEAR(rho_eq[1], 0.25, 1e-12);
    EXPECT_TRUE(std::isfinite(rho_eq[2]));

    double rho_gt[3] = {};
    loss->Evaluate(1.5 * c, rho_gt);
    EXPECT_TRUE(std::isfinite(rho_gt[0]));
    EXPECT_NEAR(rho_gt[0] / c, 0.6, 1e-12);
    EXPECT_NEAR(rho_gt[1], 0.16, 1e-12);
    EXPECT_TRUE(std::isfinite(rho_gt[2]));
  }
}

}  // namespace
}  // namespace colmap

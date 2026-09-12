// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/cost_functions/utils.h"

#include "colmap/util/eigen_matchers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(NormalPriorCostFunctor, Nominal) {
  const Eigen::Vector3d prior(1, 2, 3);

  std::unique_ptr<ceres::CostFunction> cost_function(
      NormalPriorCostFunctor<3>::Create(prior));
  ASSERT_NE(cost_function, nullptr);
  EXPECT_EQ(cost_function->num_residuals(), 3);

  Eigen::Vector3d residuals;
  const double* parameters_zero[1] = {prior.data()};
  EXPECT_TRUE(
      cost_function->Evaluate(parameters_zero, residuals.data(), nullptr));
  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-10));

  const Eigen::Vector3d param(4, 5, 6);
  const double* parameters[1] = {param.data()};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals,
              EigenMatrixNear(Eigen::Vector3d(param - prior), 1e-10));
}

TEST(NormalErrorCostFunctor, Nominal) {
  const Eigen::Vector3d param0(1, 2, 3);
  const Eigen::Vector3d param1(4, 5, 6);

  std::unique_ptr<ceres::CostFunction> cost_function(
      NormalErrorCostFunctor<3>::Create());
  ASSERT_NE(cost_function, nullptr);
  EXPECT_EQ(cost_function->num_residuals(), 3);

  Eigen::Vector3d residuals;
  const double* parameters_zero[2] = {param0.data(), param0.data()};
  EXPECT_TRUE(
      cost_function->Evaluate(parameters_zero, residuals.data(), nullptr));
  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-10));

  const double* parameters[2] = {param0.data(), param1.data()};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals,
              EigenMatrixNear(Eigen::Vector3d(param0 - param1), 1e-10));
}

TEST(CovarianceWeightedCostFunctor, NormalPriorCostFunctor) {
  const Eigen::Vector3d prior(1, 2, 3);
  const Eigen::Vector3d param(4, 5, 6);
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
  covariance(0, 0) = 4.0;

  std::unique_ptr<ceres::CostFunction> cost_function(
      CovarianceWeightedCostFunctor<NormalPriorCostFunctor<3>>::Create(
          covariance, prior));

  Eigen::Vector3d residuals;
  const double* parameters[1] = {param.data()};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals,
              EigenMatrixNear(Eigen::Vector3d(0.5 * (param[0] - prior[0]),
                                              1.0 * (param[1] - prior[1]),
                                              1.0 * (param[2] - prior[2])),
                              1e-10));
}

TEST(CovarianceWeightedCostFunctor, NormalErrorCostFunctor) {
  const Eigen::Vector3d param0(1, 2, 3);
  const Eigen::Vector3d param1(4, 5, 6);
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
  covariance(0, 0) = 4.0;

  std::unique_ptr<ceres::CostFunction> cost_function(
      CovarianceWeightedCostFunctor<NormalErrorCostFunctor<3>>::Create(
          covariance));

  Eigen::Vector3d residuals;
  const double* parameters[2] = {param0.data(), param1.data()};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals,
              EigenMatrixNear(Eigen::Vector3d(0.5 * (param0[0] - param1[0]),
                                              1.0 * (param0[1] - param1[1]),
                                              1.0 * (param0[2] - param1[2])),
                              1e-10));
}

TEST(ScaleWeightedCostFunctor, NormalPriorCostFunctorStddevVec) {
  const Eigen::Vector3d prior(1, 2, 3);
  const Eigen::Vector3d param(4, 5, 6);

  std::unique_ptr<ceres::CostFunction> cost_function(
      ScaleWeightedCostFunctor<NormalPriorCostFunctor<3>>::Create(
          Eigen::Vector3d(2.0, 1.0, 0.5), prior));

  Eigen::Vector3d residuals;
  const double* parameters[1] = {param.data()};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals,
              EigenMatrixNear(Eigen::Vector3d((param[0] - prior[0]) / 2.0,
                                              (param[1] - prior[1]) / 1.0,
                                              (param[2] - prior[2]) / 0.5),
                              1e-10));
}

TEST(ScaleWeightedCostFunctor, NormalErrorCostFunctorSingleStddev) {
  const Eigen::Vector3d param0(1, 2, 3);
  const Eigen::Vector3d param1(4, 5, 6);

  std::unique_ptr<ceres::CostFunction> cost_function(
      ScaleWeightedCostFunctor<NormalErrorCostFunctor<3>>::Create(2.0));

  Eigen::Vector3d residuals;
  const double* parameters[2] = {param0.data(), param1.data()};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals,
              EigenMatrixNear(Eigen::Vector3d((param0 - param1) / 2.0), 1e-10));
}

TEST(ScaleWeightedCostFunctor, MatchesCovarianceWeighted) {
  const Eigen::Vector3d prior(1, 2, 3);
  const Eigen::Vector3d param(4, 5, 6);
  const Eigen::Vector3d stddevs(2.0, 1.0, 0.5);

  std::unique_ptr<ceres::CostFunction> scale_cost_function(
      ScaleWeightedCostFunctor<NormalPriorCostFunctor<3>>::Create(stddevs,
                                                                  prior));
  std::unique_ptr<ceres::CostFunction> cov_cost_function(
      CovarianceWeightedCostFunctor<NormalPriorCostFunctor<3>>::Create(
          stddevs.cwiseAbs2().asDiagonal(), prior));

  const double* parameters[1] = {param.data()};
  Eigen::Vector3d scale_residuals;
  Eigen::Vector3d cov_residuals;
  Eigen::Matrix3d scale_jacobian;
  Eigen::Matrix3d cov_jacobian;
  double* scale_jacobians[1] = {scale_jacobian.data()};
  double* cov_jacobians[1] = {cov_jacobian.data()};
  EXPECT_TRUE(scale_cost_function->Evaluate(
      parameters, scale_residuals.data(), scale_jacobians));
  EXPECT_TRUE(cov_cost_function->Evaluate(
      parameters, cov_residuals.data(), cov_jacobians));

  EXPECT_THAT(scale_residuals, EigenMatrixNear(cov_residuals, 1e-10));
  EXPECT_THAT(scale_jacobian, EigenMatrixNear(cov_jacobian, 1e-10));
}

// Analytical counterpart of NormalPriorCostFunctor<3>, to check that the
// wrapper also accepts an inner cost function that computes its own jacobians.
class AnalyticalNormalPriorCostFunction
    : public ceres::SizedCostFunction<3, 3> {
 public:
  explicit AnalyticalNormalPriorCostFunction(const Eigen::Vector3d& prior)
      : prior_(prior) {}

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    Eigen::Map<Eigen::Vector3d> residuals_vec(residuals);
    residuals_vec = Eigen::Map<const Eigen::Vector3d>(parameters[0]) - prior_;
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(jacobians[0])
          .setIdentity();
    }
    return true;
  }

 private:
  const Eigen::Vector3d prior_;
};

TEST(CovarianceWeightedCostFunction, WrapsAnalyticalCostFunction) {
  const Eigen::Vector3d prior(1, 2, 3);
  const Eigen::Vector3d param(4, 5, 6);
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
  covariance(0, 0) = 4.0;

  std::unique_ptr<ceres::CostFunction> autodiff_cost_function(
      CovarianceWeightedCostFunctor<NormalPriorCostFunctor<3>>::Create(
          covariance, prior));
  std::unique_ptr<ceres::CostFunction> analytical_cost_function(
      new CovarianceWeightedCostFunction<NormalPriorCostFunctor<3>>(
          covariance, new AnalyticalNormalPriorCostFunction(prior)));

  const double* parameters[1] = {param.data()};
  Eigen::Vector3d autodiff_residuals;
  Eigen::Vector3d analytical_residuals;
  Eigen::Matrix3d autodiff_jacobian;
  Eigen::Matrix3d analytical_jacobian;
  double* autodiff_jacobians[1] = {autodiff_jacobian.data()};
  double* analytical_jacobians[1] = {analytical_jacobian.data()};
  EXPECT_TRUE(autodiff_cost_function->Evaluate(
      parameters, autodiff_residuals.data(), autodiff_jacobians));
  EXPECT_TRUE(analytical_cost_function->Evaluate(
      parameters, analytical_residuals.data(), analytical_jacobians));

  EXPECT_THAT(analytical_residuals, EigenMatrixNear(autodiff_residuals, 1e-10));
  EXPECT_THAT(analytical_jacobian, EigenMatrixNear(autodiff_jacobian, 1e-10));

  // Ceres also calls Evaluate without jacobians.
  Eigen::Vector3d residuals_only;
  EXPECT_TRUE(analytical_cost_function->Evaluate(
      parameters, residuals_only.data(), nullptr));
  EXPECT_THAT(residuals_only, EigenMatrixNear(autodiff_residuals, 1e-10));
}

}  // namespace
}  // namespace colmap

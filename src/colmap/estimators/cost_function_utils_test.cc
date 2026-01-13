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

#include "colmap/estimators/cost_function_utils.h"

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

}  // namespace
}  // namespace colmap

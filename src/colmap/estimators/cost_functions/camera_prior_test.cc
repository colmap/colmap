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

#include "colmap/estimators/cost_functions/camera_prior.h"

#include "colmap/sensor/models.h"
#include "colmap/util/eigen_matchers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

using CostFunctor = CameraParamsPriorCostFunctor<SimpleRadialCameraModel>;
using VectorN = CostFunctor::VectorN;

std::vector<double> ToVector(const VectorN& vec) {
  return std::vector<double>(vec.data(), vec.data() + vec.size());
}

TEST(CameraParamsPriorCostFunctor, ZeroAtPrior) {
  const VectorN priors(100, 50, 40, 0.01);
  std::unique_ptr<ceres::CostFunction> cost_function(
      CostFunctor::Create(ToVector(VectorN::Constant(2.0)), ToVector(priors)));

  VectorN params = priors;
  VectorN residuals = VectorN::Constant(std::numeric_limits<double>::max());
  const double* parameters[1] = {params.data()};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals, EigenMatrixNear(VectorN::Zero().eval(), 1e-12));
}

TEST(CameraParamsPriorCostFunctor, ScalesLinearlyWithWeight) {
  const VectorN priors(100, 50, 40, 0.01);
  VectorN params = priors;
  params += VectorN(10, -5, 2, 0.5);
  const double* parameters[1] = {params.data()};

  VectorN residuals1 = VectorN::Zero();
  std::unique_ptr<ceres::CostFunction> cost_function1(
      CostFunctor::Create(ToVector(VectorN::Constant(1.0)), ToVector(priors)));
  EXPECT_TRUE(cost_function1->Evaluate(parameters, residuals1.data(), nullptr));
  EXPECT_THAT(residuals1,
              EigenMatrixNear(VectorN(10, -5, 2, 0.5).eval(), 1e-12));

  VectorN residuals3 = VectorN::Zero();
  std::unique_ptr<ceres::CostFunction> cost_function3(
      CostFunctor::Create(ToVector(VectorN::Constant(3.0)), ToVector(priors)));
  EXPECT_TRUE(cost_function3->Evaluate(parameters, residuals3.data(), nullptr));
  EXPECT_THAT(residuals3, EigenMatrixNear((3 * residuals1).eval(), 1e-12));
}

TEST(CameraParamsPriorCostFunctor, ZeroWeightIsUnconstrained) {
  const VectorN weights(1.0, 0.0, 0.0, 2.0);
  const VectorN priors(100, 50, 40, 0.01);
  std::unique_ptr<ceres::CostFunction> cost_function(
      CostFunctor::Create(ToVector(weights), ToVector(priors)));

  VectorN params(100, 1000, -1000, 0.01);
  VectorN residuals = VectorN::Constant(std::numeric_limits<double>::max());
  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> jacobian =
      Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Constant(
          std::numeric_limits<double>::max());
  const double* parameters[1] = {params.data()};
  double* jacobians[1] = {jacobian.data()};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), jacobians));

  EXPECT_THAT(residuals, EigenMatrixNear(VectorN::Zero().eval(), 1e-12));
  EXPECT_THAT(jacobian,
              EigenMatrixNear(Eigen::Matrix4d(weights.asDiagonal()), 1e-12));
}

TEST(CameraParamsPriorCostFunctor, Jacobian) {
  const VectorN weights(1.5, 0.0, 2.5, 0.5);
  const VectorN priors(100, 50, 40, 0.01);
  std::unique_ptr<ceres::CostFunction> cost_function(
      CostFunctor::Create(ToVector(weights), ToVector(priors)));

  VectorN params(120, 60, 30, -0.02);
  std::vector<double*> parameter_blocks{params.data()};

  ceres::NumericDiffOptions numeric_diff_options;
  ceres::GradientChecker gradient_checker(
      cost_function.get(),
#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
      static_cast<const std::vector<const ceres::Manifold*>*>(nullptr),
#else
      static_cast<const std::vector<const ceres::LocalParameterization*>*>(
          nullptr),
#endif
      numeric_diff_options);
  ceres::GradientChecker::ProbeResults results;
  EXPECT_TRUE(gradient_checker.Probe(parameter_blocks.data(), 1e-6, &results))
      << results.error_log;
}

}  // namespace
}  // namespace colmap

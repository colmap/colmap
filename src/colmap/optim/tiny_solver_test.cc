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

#include "colmap/optim/tiny_solver.h"

#include "colmap/estimators/cost_functions/tiny_manifold.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/tiny_solver_autodiff_function.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Linear least squares: residual = A * p - b, with analytic Jacobian.
struct LinearResidual {
  using Scalar = double;
  enum { NUM_RESIDUALS = 3, NUM_PARAMETERS = 2 };

  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    Eigen::Matrix<double, 3, 2> A;
    A << 1, 0, 0, 1, 1, 1;
    const Eigen::Vector3d b(1, 2, 4);
    const Eigen::Map<const Eigen::Vector2d> p(parameters);
    Eigen::Map<Eigen::Vector3d> res(residuals);
    res = A * p - b;
    if (jacobian != nullptr) {
      // Column-major, NUM_RESIDUALS x NUM_PARAMETERS.
      Eigen::Map<Eigen::Matrix<double, 3, 2>> jac(jacobian);
      jac = A;
    }
    return true;
  }
};

TEST(TinySolver, EuclideanConvergesToNormalEquationsSolution) {
  Eigen::Matrix<double, 3, 2> A;
  A << 1, 0, 0, 1, 1, 1;
  const Eigen::Vector3d b(1, 2, 4);
  const Eigen::Vector2d expected =
      (A.transpose() * A).ldlt().solve(A.transpose() * b);

  TinySolver<LinearResidual> solver;
  TinySolver<LinearResidual>::Options options;
  options.gradient_tolerance = 0;
  options.parameter_tolerance = 0;
  options.function_tolerance = 0;
  options.max_num_iterations = 100;
  Eigen::Vector2d x(0, 0);
  const auto& summary = solver.Solve(LinearResidual(), &x, options);

  EXPECT_NE(summary.status, TinySolver<LinearResidual>::COST_FUNCTION_FAILED);
  EXPECT_LT((x - expected).norm(), 1e-9);
  // Started from x = 0, so initial_cost = 0.5 * ||b||^2 = 0.5 * 21.
  EXPECT_NEAR(summary.initial_cost, 10.5, 1e-9);
  // Cost must not increase, and at the least-squares optimum J'f(x) vanishes.
  EXPECT_GE(summary.final_cost, 0.0);
  EXPECT_LE(summary.final_cost, summary.initial_cost);
  EXPECT_NEAR(summary.gradient_max_norm, 0.0, 1e-6);
  EXPECT_GE(summary.iterations, 1);
  EXPECT_LE(summary.iterations, options.max_num_iterations);
}

// Autodiff functor minimizing ||x - target||^2; on the unit sphere the optimum
// is target.normalized().
struct SphereFitResidual {
  Eigen::Vector3d target;
  template <typename T>
  bool operator()(const T* const x, T* residuals) const {
    residuals[0] = x[0] - T(target(0));
    residuals[1] = x[1] - T(target(1));
    residuals[2] = x[2] - T(target(2));
    return true;
  }
};

TEST(TinySolver, ManifoldConvergesAndStaysOnManifold) {
  SphereFitResidual functor;
  functor.target = Eigen::Vector3d(0.3, -0.7, 0.5);

  using AutoDiff = ceres::TinySolverAutoDiffFunction<SphereFitResidual, 3, 3>;
  AutoDiff f(functor);

  using Solver = TinySolver<AutoDiff, SphereManifold<3>>;
  Solver solver;
  Solver::Options options;
  options.gradient_tolerance = 0;
  options.parameter_tolerance = 0;
  options.function_tolerance = 0;
  options.max_num_iterations = 100;
  Eigen::Vector3d x = Eigen::Vector3d(1, 0, 0);  // Unit seed.
  solver.Solve(f, &x, options);

  EXPECT_NEAR(x.norm(), 1.0, 1e-9);
  EXPECT_LT((x - functor.target.normalized()).norm(), 1e-6);
}

// A functor that always fails to evaluate.
struct FailingResidual {
  using Scalar = double;
  enum { NUM_RESIDUALS = 2, NUM_PARAMETERS = 2 };
  bool operator()(const double* /*parameters*/,
                  double* /*residuals*/,
                  double* /*jacobian*/) const {
    return false;
  }
};

TEST(TinySolver, ReportsCostFunctionFailure) {
  TinySolver<FailingResidual> solver;
  Eigen::Vector2d x(1, 2);
  const auto& summary = solver.Solve(FailingResidual(), &x);
  EXPECT_EQ(summary.status, TinySolver<FailingResidual>::COST_FUNCTION_FAILED);
}

}  // namespace
}  // namespace colmap

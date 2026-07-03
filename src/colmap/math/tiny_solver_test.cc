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

#include "colmap/math/tiny_solver.h"

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
  solver.options.gradient_tolerance = 0;
  solver.options.parameter_tolerance = 0;
  solver.options.function_tolerance = 0;
  solver.options.max_num_iterations = 100;
  Eigen::Vector2d x(0, 0);
  const auto& summary = solver.Solve(LinearResidual(), &x);

  EXPECT_NE(summary.status, TinySolver<LinearResidual>::COST_FUNCTION_FAILED);
  EXPECT_LT((x - expected).norm(), 1e-9);
}

// Minimal 3->2 unit-sphere manifold used to exercise the solver's manifold
// projection path independently of the cost_functions library.
struct TestSphereManifold {
  static constexpr int kAmbientSize = 3;
  static constexpr int kTangentSize = 2;
  [[maybe_unused]] static constexpr bool kIsEuclidean = false;

  void Plus(const double* x, const double* delta, double* x_plus) const {
    const Eigen::Map<const Eigen::Vector3d> xv(x);
    const Eigen::Vector3d x_hat = xv.normalized();
    const Eigen::Vector3d b1 = x_hat.unitOrthogonal();
    const Eigen::Vector3d b2 = x_hat.cross(b1);
    Eigen::Map<Eigen::Vector3d> out(x_plus);
    out = (xv + delta[0] * b1 + delta[1] * b2).normalized();
  }

  void PlusJacobian(const double* x, double* jacobian) const {
    const Eigen::Map<const Eigen::Vector3d> xv(x);
    const Eigen::Vector3d x_hat = xv.normalized();
    const Eigen::Vector3d b1 = x_hat.unitOrthogonal();
    const Eigen::Vector3d b2 = x_hat.cross(b1);
    for (int r = 0; r < 3; ++r) {
      jacobian[r * 2 + 0] = b1(r);
      jacobian[r * 2 + 1] = b2(r);
    }
  }
};

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

  TinySolver<AutoDiff, TestSphereManifold> solver;
  solver.options.gradient_tolerance = 0;
  solver.options.parameter_tolerance = 0;
  solver.options.function_tolerance = 0;
  solver.options.max_num_iterations = 100;
  Eigen::Vector3d x = Eigen::Vector3d(1, 0, 0);  // Unit seed.
  solver.Solve(f, &x);

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

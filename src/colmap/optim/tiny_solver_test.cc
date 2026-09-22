// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/tiny_solver.h"

#include "colmap/estimators/cost_functions/tiny_manifold.h"

#include <limits>
#include <vector>

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

// Dynamic-size least squares: residual_i = p - values[i], so the optimum is
// the mean of the values. Exercises the dynamically-sized residual path that
// all production cost functors use.
struct DynamicMeanResidual {
  using Scalar = double;
  enum { NUM_RESIDUALS = Eigen::Dynamic, NUM_PARAMETERS = 1 };

  std::vector<double> values;

  int NumResiduals() const { return static_cast<int>(values.size()); }

  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    for (size_t i = 0; i < values.size(); ++i) {
      residuals[i] = parameters[0] - values[i];
      if (jacobian != nullptr) {
        jacobian[i] = 1.0;
      }
    }
    return true;
  }
};

TEST(TinySolver, DynamicResidualsConvergeAndResizeOnReuse) {
  TinySolver<DynamicMeanResidual> solver;
  TinySolver<DynamicMeanResidual>::Options options;
  options.gradient_tolerance = 0;
  options.parameter_tolerance = 0;
  options.function_tolerance = 0;
  options.max_num_iterations = 100;
  Eigen::Matrix<double, 1, 1> x;
  DynamicMeanResidual mean_of_three;
  mean_of_three.values = {1.0, 2.0, 3.0};
  x << 0.0;
  const auto first_summary = solver.Solve(mean_of_three, &x, options);
  EXPECT_NE(first_summary.status,
            TinySolver<DynamicMeanResidual>::COST_FUNCTION_FAILED);
  EXPECT_NEAR(x(0), 2.0, 1e-6);

  // Reuse the same solver instance with a different residual count, exercising
  // the buffer resize in Initialize() on both growth and shrinkage.
  DynamicMeanResidual mean_of_five;
  mean_of_five.values = {1.0, 2.0, 3.0, 4.0, 15.0};
  x << 0.0;
  const auto second_summary = solver.Solve(mean_of_five, &x, options);
  EXPECT_NE(second_summary.status,
            TinySolver<DynamicMeanResidual>::COST_FUNCTION_FAILED);
  EXPECT_NEAR(x(0), 5.0, 1e-6);

  DynamicMeanResidual mean_of_two;
  mean_of_two.values = {1.0, 4.0};
  x << 0.0;
  const auto third_summary = solver.Solve(mean_of_two, &x, options);
  EXPECT_NE(third_summary.status,
            TinySolver<DynamicMeanResidual>::COST_FUNCTION_FAILED);
  EXPECT_NEAR(x(0), 2.5, 1e-6);
}

TEST(TinySolver, SolverInstanceIsReusable) {
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
  for (const Eigen::Vector2d& start :
       {Eigen::Vector2d(0, 0), Eigen::Vector2d(10, -5)}) {
    Eigen::Vector2d x = start;
    const auto summary = solver.Solve(LinearResidual(), &x, options);
    EXPECT_NE(summary.status, TinySolver<LinearResidual>::COST_FUNCTION_FAILED);
    EXPECT_LT((x - expected).norm(), 1e-9);
  }
}

TEST(TinySolver, ReportsGradientTooSmallAtOptimum) {
  Eigen::Matrix<double, 3, 2> A;
  A << 1, 0, 0, 1, 1, 1;
  const Eigen::Vector3d b(1, 2, 4);
  const Eigen::Vector2d expected =
      (A.transpose() * A).ldlt().solve(A.transpose() * b);

  TinySolver<LinearResidual> solver;
  Eigen::Vector2d x = expected;
  const auto summary = solver.Solve(LinearResidual(), &x);
  EXPECT_EQ(summary.status, TinySolver<LinearResidual>::GRADIENT_TOO_SMALL);
  EXPECT_EQ(summary.iterations, 0);
}

// Exactly satisfiable system for the COST_TOO_SMALL test: starting at the
// solution gives exactly zero cost.
struct IdentityResidual {
  using Scalar = double;
  enum { NUM_RESIDUALS = 2, NUM_PARAMETERS = 2 };

  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    residuals[0] = parameters[0] - 3.0;
    residuals[1] = parameters[1] + 1.0;
    if (jacobian != nullptr) {
      jacobian[0] = 1.0;
      jacobian[1] = 0.0;
      jacobian[2] = 0.0;
      jacobian[3] = 1.0;
    }
    return true;
  }
};

TEST(TinySolver, ReportsCostTooSmallAtExactSolution) {
  TinySolver<IdentityResidual> solver;
  TinySolver<IdentityResidual>::Options options;
  // Let the cost check (not the gradient check, which runs first) trigger.
  options.gradient_tolerance = 0;
  Eigen::Vector2d x(3.0, -1.0);
  const auto summary = solver.Solve(IdentityResidual(), &x, options);
  EXPECT_EQ(summary.status, TinySolver<IdentityResidual>::COST_TOO_SMALL);
  EXPECT_EQ(summary.iterations, 0);
}

TEST(TinySolver, ReportsRelativeStepSizeTooSmall) {
  TinySolver<LinearResidual> solver;
  TinySolver<LinearResidual>::Options options;
  options.parameter_tolerance = 1e10;  // Any first step is "too small".
  Eigen::Vector2d x(0, 0);
  const auto summary = solver.Solve(LinearResidual(), &x, options);
  EXPECT_EQ(summary.status,
            TinySolver<LinearResidual>::RELATIVE_STEP_SIZE_TOO_SMALL);
  EXPECT_EQ(summary.iterations, 1);
  EXPECT_EQ(x, Eigen::Vector2d(0, 0));  // No step was taken.
}

TEST(TinySolver, ReportsHitMaxIterations) {
  TinySolver<LinearResidual> solver;
  TinySolver<LinearResidual>::Options options;
  options.gradient_tolerance = 0;
  options.parameter_tolerance = 0;
  options.function_tolerance = 0;
  options.max_num_iterations = 2;  // A single damped step cannot converge.
  Eigen::Vector2d x(0, 0);
  const auto summary = solver.Solve(LinearResidual(), &x, options);
  EXPECT_EQ(summary.status, TinySolver<LinearResidual>::HIT_MAX_ITERATIONS);
  EXPECT_EQ(summary.iterations, 2);
  EXPECT_LT(summary.final_cost, summary.initial_cost);
}

TEST(TinySolver, ReportsCostChangeTooSmall) {
  TinySolver<LinearResidual> solver;
  TinySolver<LinearResidual>::Options options;
  options.function_tolerance = 1e10;  // Any first accepted step qualifies.
  Eigen::Vector2d x(0, 0);
  const auto summary = solver.Solve(LinearResidual(), &x, options);
  EXPECT_EQ(summary.status, TinySolver<LinearResidual>::COST_CHANGE_TOO_SMALL);
  EXPECT_EQ(summary.iterations, 1);
  EXPECT_LT(summary.final_cost, summary.initial_cost);
}

// Least squares with O(1e9) data and O(1e17) optimal cost, with analytic
// Jacobian. An absolute function tolerance could never terminate here within
// a few iterations, while a relative one does.
struct ScaledLinearResidual {
  using Scalar = double;
  enum { NUM_RESIDUALS = 3, NUM_PARAMETERS = 2 };

  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    Eigen::Matrix<double, 3, 2> A;
    A << 1, 0, 0, 1, 1, 1;
    const Eigen::Vector3d b(1e9, 2e9, 4e9);
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

TEST(TinySolver, CostChangeToleranceIsRelativeToCost) {
  Eigen::Matrix<double, 3, 2> A;
  A << 1, 0, 0, 1, 1, 1;
  const Eigen::Vector3d b(1e9, 2e9, 4e9);
  const Eigen::Vector2d expected =
      (A.transpose() * A).ldlt().solve(A.transpose() * b);

  TinySolver<ScaledLinearResidual> solver;
  TinySolver<ScaledLinearResidual>::Options options;
  // Isolate the function-tolerance check: with the other criteria disabled,
  // only a cost change relative to the O(1e17) cost can terminate before the
  // iteration limit.
  options.gradient_tolerance = 0;
  options.parameter_tolerance = 0;
  options.function_tolerance = 1e-6;
  options.cost_threshold = 0;
  options.max_num_iterations = 3;
  Eigen::Vector2d x(0, 0);
  const auto summary = solver.Solve(ScaledLinearResidual(), &x, options);
  EXPECT_EQ(summary.status,
            TinySolver<ScaledLinearResidual>::COST_CHANGE_TOO_SMALL);
  EXPECT_LT((x - expected).norm() / expected.norm(), 1e-6);
}

// Succeeds only at the start point: every trial step fails to evaluate, so all
// steps must be rejected and the input left unchanged.
struct FailExceptAtStartResidual {
  using Scalar = double;
  enum { NUM_RESIDUALS = 2, NUM_PARAMETERS = 2 };

  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    if (parameters[0] != 0.0 || parameters[1] != 0.0) {
      return false;
    }
    residuals[0] = 1.0;
    residuals[1] = 2.0;
    if (jacobian != nullptr) {
      jacobian[0] = 1.0;
      jacobian[1] = 0.0;
      jacobian[2] = 0.0;
      jacobian[3] = 1.0;
    }
    return true;
  }
};

TEST(TinySolver, TrialPointFailuresAreRejectedWithoutProgress) {
  TinySolver<FailExceptAtStartResidual> solver;
  TinySolver<FailExceptAtStartResidual>::Options options;
  Eigen::Vector2d x(0, 0);
  const auto summary = solver.Solve(FailExceptAtStartResidual(), &x, options);
  // The trust region shrinks until the step underflows the parameter tolerance.
  EXPECT_EQ(
      summary.status,
      TinySolver<FailExceptAtStartResidual>::RELATIVE_STEP_SIZE_TOO_SMALL);
  EXPECT_LT(summary.iterations, options.max_num_iterations);
  EXPECT_EQ(x, Eigen::Vector2d(0, 0));
}

// Fails the Jacobian evaluation after the first accepted step: the
// residuals-only trial evaluation succeeds, but re-evaluating with the
// Jacobian at the accepted point fails.
struct FailOnSecondJacobianEvalResidual {
  using Scalar = double;
  enum { NUM_RESIDUALS = 2, NUM_PARAMETERS = 2 };

  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    residuals[0] = parameters[0] - 1.0;
    residuals[1] = parameters[1] - 2.0;
    if (jacobian != nullptr) {
      if (num_jacobian_evals_++ > 0) {
        return false;
      }
      jacobian[0] = 1.0;
      jacobian[1] = 0.0;
      jacobian[2] = 0.0;
      jacobian[3] = 1.0;
    }
    return true;
  }

  mutable int num_jacobian_evals_ = 0;
};

TEST(TinySolver, PostAcceptUpdateFailureReportsCostFunctionFailed) {
  TinySolver<FailOnSecondJacobianEvalResidual> solver;
  FailOnSecondJacobianEvalResidual functor;
  Eigen::Vector2d x(0, 0);
  const auto summary = solver.Solve(functor, &x);
  EXPECT_EQ(summary.status,
            TinySolver<FailOnSecondJacobianEvalResidual>::COST_FUNCTION_FAILED);
  EXPECT_EQ(summary.iterations, 1);
  EXPECT_TRUE(x.allFinite());
  // The reported cost is the cost of the accepted point, not the stale cost
  // from before the step.
  Eigen::Vector2d residuals;
  ASSERT_TRUE(functor(x.data(), residuals.data(), nullptr));
  EXPECT_DOUBLE_EQ(summary.final_cost, 0.5 * residuals.squaredNorm());
}

// Linear solver stub that always reports factorization failure, exercising the
// solver's rejected-step path deterministically.
struct AlwaysFailingLinearSolver {
  void compute(const Eigen::Matrix2d&) { info_ = Eigen::NumericalIssue; }
  Eigen::Vector2d solve(const Eigen::Vector2d&) const {
    return Eigen::Vector2d::Zero();
  }
  Eigen::ComputationInfo info() const { return info_; }

  Eigen::ComputationInfo info_ = Eigen::Success;
};

TEST(TinySolver, LinearSolverFailuresAreRejectedWithoutProgress) {
  using Solver = TinySolver<LinearResidual,
                            EuclideanManifold<2>,
                            AlwaysFailingLinearSolver>;
  Solver solver;
  Solver::Options options;
  options.max_num_iterations = 10;
  Eigen::Vector2d x(0, 0);
  const auto summary = solver.Solve(LinearResidual(), &x, options);
  // No step is ever taken; the iterations are exhausted rejecting.
  EXPECT_EQ(summary.status, Solver::HIT_MAX_ITERATIONS);
  EXPECT_EQ(summary.iterations, options.max_num_iterations);
  EXPECT_EQ(x, Eigen::Vector2d(0, 0));
}

TEST(TinySolver, InitialCostFunctionFailureLeavesInputUnchanged) {
  TinySolver<FailingResidual> solver;
  Eigen::Vector2d x(1, 2);
  const auto summary = solver.Solve(FailingResidual(), &x);
  EXPECT_EQ(summary.status, TinySolver<FailingResidual>::COST_FUNCTION_FAILED);
  EXPECT_EQ(summary.iterations, 0);
  EXPECT_EQ(x, Eigen::Vector2d(1, 2));
}

// Returns success but writes NaN residuals, modeling a cost function that
// misreports success with garbage output.
struct NaNResidual {
  using Scalar = double;
  enum { NUM_RESIDUALS = 2, NUM_PARAMETERS = 2 };

  bool operator()(const double* /*parameters*/,
                  double* residuals,
                  double* jacobian) const {
    residuals[0] = std::numeric_limits<double>::quiet_NaN();
    residuals[1] = std::numeric_limits<double>::quiet_NaN();
    if (jacobian != nullptr) {
      jacobian[0] = 1.0;
      jacobian[1] = 0.0;
      jacobian[2] = 0.0;
      jacobian[3] = 1.0;
    }
    return true;
  }
};

TEST(TinySolver, ReportsNumericalFailureOnNonFiniteInitialResiduals) {
  TinySolver<NaNResidual> solver;
  Eigen::Vector2d x(0, 0);
  const auto summary = solver.Solve(NaNResidual(), &x);
  EXPECT_EQ(summary.status, TinySolver<NaNResidual>::NUMERICAL_FAILURE);
  EXPECT_EQ(summary.iterations, 0);
  EXPECT_EQ(x, Eigen::Vector2d(0, 0));
}

// Returns success with finite residuals but one NaN Jacobian entry. A partial
// NaN is important here because maxCoeff() does not guarantee NaN propagation.
struct NaNJacobianResidual {
  using Scalar = double;
  enum { NUM_RESIDUALS = 2, NUM_PARAMETERS = 2 };

  bool operator()(const double* /*parameters*/,
                  double* residuals,
                  double* jacobian) const {
    residuals[0] = 1.0;
    residuals[1] = 2.0;
    if (jacobian != nullptr) {
      jacobian[0] = 1.0;
      jacobian[1] = 0.0;
      jacobian[2] = std::numeric_limits<double>::quiet_NaN();
      jacobian[3] = 1.0;
    }
    return true;
  }
};

TEST(TinySolver, ReportsNumericalFailureOnNonFiniteInitialJacobian) {
  TinySolver<NaNJacobianResidual> solver;
  Eigen::Vector2d x(0, 0);
  const auto summary = solver.Solve(NaNJacobianResidual(), &x);
  EXPECT_EQ(summary.status, TinySolver<NaNJacobianResidual>::NUMERICAL_FAILURE);
  EXPECT_EQ(summary.iterations, 0);
  EXPECT_EQ(x, Eigen::Vector2d(0, 0));
}

TEST(TinySolver, ReportsNumericalFailureOnNonFiniteInitialParameters) {
  TinySolver<LinearResidual> solver;
  Eigen::Vector2d x(std::numeric_limits<double>::quiet_NaN(), 0);
  const auto summary = solver.Solve(LinearResidual(), &x);
  EXPECT_EQ(summary.status, TinySolver<LinearResidual>::NUMERICAL_FAILURE);
  EXPECT_EQ(summary.iterations, 0);
}

// Returns a NaN Jacobian after the first accepted step (residuals stay
// finite), modeling a derivative that blows up away from the start point.
struct NaNOnSecondJacobianEvalResidual {
  using Scalar = double;
  enum { NUM_RESIDUALS = 2, NUM_PARAMETERS = 2 };

  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    residuals[0] = parameters[0] - 1.0;
    residuals[1] = parameters[1] - 2.0;
    if (jacobian != nullptr) {
      if (num_jacobian_evals_++ > 0) {
        jacobian[0] = 1.0;
        jacobian[1] = 0.0;
        jacobian[2] = std::numeric_limits<double>::quiet_NaN();
        jacobian[3] = 1.0;
      } else {
        jacobian[0] = 1.0;
        jacobian[1] = 0.0;
        jacobian[2] = 0.0;
        jacobian[3] = 1.0;
      }
    }
    return true;
  }

  mutable int num_jacobian_evals_ = 0;
};

TEST(TinySolver, PostAcceptNonFiniteJacobianReportsNumericalFailure) {
  TinySolver<NaNOnSecondJacobianEvalResidual> solver;
  NaNOnSecondJacobianEvalResidual functor;
  Eigen::Vector2d x(0, 0);
  const auto summary = solver.Solve(functor, &x);
  EXPECT_EQ(summary.status,
            TinySolver<NaNOnSecondJacobianEvalResidual>::NUMERICAL_FAILURE);
  EXPECT_EQ(summary.iterations, 1);
  // The reported cost is the cost of the accepted point, not the stale cost
  // from before the step.
  Eigen::Vector2d residuals;
  ASSERT_TRUE(functor(x.data(), residuals.data(), nullptr));
  EXPECT_DOUBLE_EQ(summary.final_cost, 0.5 * residuals.squaredNorm());
}

// Linear solver stub that reports success but solves to NaN, modeling a silent
// factorization failure.
struct NaNSolvingLinearSolver {
  void compute(const Eigen::Matrix2d&) {}
  Eigen::Vector2d solve(const Eigen::Vector2d&) const {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return Eigen::Vector2d(nan, nan);
  }
  Eigen::ComputationInfo info() const { return Eigen::Success; }
};

TEST(TinySolver, NonFiniteStepReportsNumericalFailure) {
  using Solver =
      TinySolver<LinearResidual, EuclideanManifold<2>, NaNSolvingLinearSolver>;
  Solver solver;
  Eigen::Vector2d x(0, 0);
  const auto summary = solver.Solve(LinearResidual(), &x);
  EXPECT_EQ(summary.status, Solver::NUMERICAL_FAILURE);
  EXPECT_EQ(summary.iterations, 1);
  EXPECT_EQ(x, Eigen::Vector2d(0, 0));
}

// Manifold stub whose retraction always leaves the ambient space, modeling a
// retraction that breaks down (e.g. normalizing a zero vector).
struct NaNPlusManifold {
  static constexpr int kAmbientSize = 2;
  static constexpr int kTangentSize = 2;
  [[maybe_unused]] static constexpr bool kIsEuclidean = false;

  void Plus(const double* /*x*/,
            const double* /*delta*/,
            double* x_plus_delta) const {
    x_plus_delta[0] = x_plus_delta[1] =
        std::numeric_limits<double>::quiet_NaN();
  }

  void PlusJacobian(const double* /*x*/, double* jacobian) const {
    jacobian[0] = 1.0;
    jacobian[1] = 0.0;
    jacobian[2] = 0.0;
    jacobian[3] = 1.0;
  }
};

TEST(TinySolver, NonFiniteRetractionIsRejectedWithoutProgress) {
  using Solver = TinySolver<LinearResidual, NaNPlusManifold>;
  Solver solver;
  Solver::Options options;
  Eigen::Vector2d x(0, 0);
  const auto summary = solver.Solve(LinearResidual(), &x, options);
  // Every trial is rejected; the trust region shrinks until the step
  // underflows the parameter tolerance.
  EXPECT_EQ(summary.status, Solver::RELATIVE_STEP_SIZE_TOO_SMALL);
  EXPECT_LT(summary.iterations, options.max_num_iterations);
  EXPECT_EQ(x, Eigen::Vector2d(0, 0));
}

// Returns success everywhere but NaN residuals away from the start point: the
// trials carry no information and must all be rejected.
struct NaNExceptAtStartResidual {
  using Scalar = double;
  enum { NUM_RESIDUALS = 2, NUM_PARAMETERS = 2 };

  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    if (parameters[0] != 0.0 || parameters[1] != 0.0) {
      const double nan = std::numeric_limits<double>::quiet_NaN();
      residuals[0] = nan;
      residuals[1] = nan;
      if (jacobian != nullptr) {
        jacobian[0] = jacobian[1] = jacobian[2] = jacobian[3] = nan;
      }
      return true;
    }
    residuals[0] = 1.0;
    residuals[1] = 2.0;
    if (jacobian != nullptr) {
      jacobian[0] = 1.0;
      jacobian[1] = 0.0;
      jacobian[2] = 0.0;
      jacobian[3] = 1.0;
    }
    return true;
  }
};

TEST(TinySolver, TrialPointNonFiniteResidualsAreRejectedWithoutProgress) {
  TinySolver<NaNExceptAtStartResidual> solver;
  TinySolver<NaNExceptAtStartResidual>::Options options;
  Eigen::Vector2d x(0, 0);
  const auto summary = solver.Solve(NaNExceptAtStartResidual(), &x, options);
  EXPECT_EQ(summary.status,
            TinySolver<NaNExceptAtStartResidual>::RELATIVE_STEP_SIZE_TOO_SMALL);
  EXPECT_LT(summary.iterations, options.max_num_iterations);
  EXPECT_EQ(x, Eigen::Vector2d(0, 0));
}

TEST(EuclideanManifold, PlusAddsDeltaAndJacobianIsIdentity) {
  const EuclideanManifold<3> manifold;
  const double x[3] = {1.0, -2.0, 0.5};
  const double delta[3] = {0.1, 0.2, -0.3};
  double x_plus[3];
  manifold.Plus(x, delta, x_plus);
  EXPECT_DOUBLE_EQ(x_plus[0], 1.1);
  EXPECT_DOUBLE_EQ(x_plus[1], -1.8);
  EXPECT_DOUBLE_EQ(x_plus[2], 0.2);
  double jacobian[9];
  manifold.PlusJacobian(x, jacobian);
  const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(
      jacobian);
  EXPECT_EQ(J, Eigen::Matrix3d::Identity());
}

TEST(EuclideanManifold, SingleDimensionPlusAndJacobian) {
  const EuclideanManifold<1> manifold;
  const double x[1] = {2.0};
  const double delta[1] = {0.5};
  double x_plus[1];
  manifold.Plus(x, delta, x_plus);
  EXPECT_DOUBLE_EQ(x_plus[0], 2.5);
  double jacobian[1];
  manifold.PlusJacobian(x, jacobian);
  EXPECT_DOUBLE_EQ(jacobian[0], 1.0);
}

}  // namespace
}  // namespace colmap

// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: mierle@gmail.com (Keir Mierle)
//
// This is a customized copy of ceres::TinySolver (ceres/tiny_solver.h) adapted
// for COLMAP. It differs from upstream in three ways:
//
//   1. Manifold support. The solver takes an optional compile-time Manifold
//      policy that decouples the ambient parameter size from the tangent step
//      size. The parameter block carries the full ambient state; the solver
//      steps via `Manifold::Plus` and projects the (ambient) Jacobian into the
//      tangent space via `Manifold::PlusJacobian`, exactly as full Ceres does.
//      The default manifold is the identity (EuclideanManifold), which
//      reproduces the original plain Levenberg-Marquardt behavior.
//
//   2. A few of the upstream robustness TODOs are addressed: cost-function
//      evaluation failures and linear-solver failures are handled instead of
//      ignored, and a COST_FUNCTION_FAILED status is reported.
//
//   3. It only supports statically sized parameter/tangent dimensions (the
//      number of residuals may still be dynamic), which keeps it fixed-size and
//      allocation-free.
//
// Like upstream, this file has no dependencies beyond Eigen.
//
// A tiny least squares solver using Levenberg-Marquardt, intended for solving
// small dense problems with low latency and low overhead. The implementation
// takes care to do all allocation up front, so that no memory is allocated
// during solving. This is especially useful when solving many similar problems;
// for example, inverse pixel distortion for every pixel on a grid.
//
// Algorithm based off of:
//
// [1] K. Madsen, H. Nielsen, O. Tingleoff.
//     Methods for Non-linear Least Squares Problems.
//     http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3215/pdf/imm3215.pdf

#pragma once

#include "colmap/util/logging.h"

#include <cmath>

#include <Eigen/Dense>

namespace colmap {

// The default (identity) manifold: the tangent space equals the ambient space
// and `Plus` is plain vector addition. Selecting this manifold reproduces the
// original ceres::TinySolver behavior.
template <int N>
struct EuclideanManifold {
  static constexpr int kAmbientSize = N;
  static constexpr int kTangentSize = N;
  [[maybe_unused]] static constexpr bool kIsEuclidean = true;

  void Plus(const double* x, const double* delta, double* x_plus_delta) const {
    for (int i = 0; i < N; ++i) {
      x_plus_delta[i] = x[i] + delta[i];
    }
  }

  // Identity Jacobian (row-major, N x N). Unused by the solver on the Euclidean
  // fast path, but provided so the manifold satisfies the full concept.
  void PlusJacobian(const double* /*x*/, double* jacobian) const {
    Eigen::Map<Eigen::Matrix<double, N, N, Eigen::RowMajor>>(jacobian)
        .setIdentity();
  }
};

// To use the tiny solver, create a class or struct that allows computing the
// cost function (described below). This is similar to a ceres::CostFunction,
// but is different to enable statically allocating all memory for the solver
// (specifically, enum sizes). Key parts are the Scalar typedef, the enums to
// describe problem sizes (needed to remove all heap allocations), and the
// operator() overload to evaluate the cost and (optionally) jacobians.
//
//   struct TinySolverCostFunctionTraits {
//     typedef double Scalar;
//     enum {
//       NUM_RESIDUALS = <int> OR Eigen::Dynamic,
//       NUM_PARAMETERS = <int>,
//     };
//     bool operator()(const double* parameters,
//                     double* residuals,
//                     double* jacobian) const;
//
//     int NumResiduals() const;  -- Needed if NUM_RESIDUALS == Eigen::Dynamic.
//   };
//
// For operator(), the size of the objects is:
//
//   double* parameters -- NUM_PARAMETERS (the ambient parameterization)
//   double* residuals  -- NUM_RESIDUALS or NumResiduals()
//   double* jacobian   -- NUM_RESIDUALS * NUM_PARAMETERS in column-major format
//                         (Eigen's default); or nullptr if no jacobian
//                         requested. This is the Jacobian with respect to the
//                         ambient parameters; the solver projects it into the
//                         tangent space via the manifold.
//
// The solver supports either a statically or dynamically sized number of
// residuals. If the number of residuals is dynamic then the Function must
// define:
//
//   int NumResiduals() const;
//
// The number of parameters (ambient) and the manifold's tangent size must be
// statically sized.
template <typename Function,
          typename Manifold = EuclideanManifold<Function::NUM_PARAMETERS>,
          typename LinearSolver =
              Eigen::LDLT<Eigen::Matrix<typename Function::Scalar,  //
                                        Manifold::kTangentSize,     //
                                        Manifold::kTangentSize>>>
// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
class TinySolver {
 public:
  enum {
    NUM_RESIDUALS = Function::NUM_RESIDUALS,
    // The size of the ambient parameter block operated on by `Function`.
    NUM_PARAMETERS = Function::NUM_PARAMETERS,
    // The size of the tangent space stepped in by the solver.
    NUM_TANGENT = Manifold::kTangentSize,
  };
  using Scalar = typename Function::Scalar;
  // The ambient parameter block (e.g. a quaternion + translation).
  using Parameters = Eigen::Matrix<Scalar, NUM_PARAMETERS, 1>;
  // A tangent-space vector (e.g. an so(3) + sphere increment).
  using Tangent = Eigen::Matrix<Scalar, NUM_TANGENT, 1>;

  static_assert(NUM_PARAMETERS != Eigen::Dynamic,
                "TinySolver requires a statically sized parameter block.");
  static_assert(NUM_TANGENT != Eigen::Dynamic,
                "TinySolver requires a statically sized tangent space.");
  static_assert(static_cast<int>(NUM_PARAMETERS) == Manifold::kAmbientSize,
                "Function::NUM_PARAMETERS must match Manifold::kAmbientSize.");

  enum Status {
    // max_norm |J'(x) * f(x)| < gradient_tolerance
    GRADIENT_TOO_SMALL,
    //  ||dx|| <= parameter_tolerance * (||x|| + parameter_tolerance)
    RELATIVE_STEP_SIZE_TOO_SMALL,
    // cost_threshold > ||f(x)||^2 / 2
    COST_TOO_SMALL,
    // num_iterations >= max_num_iterations
    HIT_MAX_ITERATIONS,
    // (new_cost - old_cost) < function_tolerance * old_cost
    COST_CHANGE_TOO_SMALL,
    // The user cost function returned false (failed to evaluate) at the initial
    // point, so no meaningful step could be taken.
    COST_FUNCTION_FAILED,
  };

  struct Options {
    int max_num_iterations = 50;

    // max_norm |J'(x) * f(x)| < gradient_tolerance
    Scalar gradient_tolerance = 1e-10;

    //  ||dx|| <= parameter_tolerance * (||x|| + parameter_tolerance)
    Scalar parameter_tolerance = 1e-8;

    // (new_cost - old_cost) < function_tolerance * old_cost
    Scalar function_tolerance = 1e-6;

    // cost_threshold > ||f(x)||^2 / 2
    Scalar cost_threshold = std::numeric_limits<Scalar>::epsilon();

    Scalar initial_trust_region_radius = 1e4;

    void Check() const {
      THROW_CHECK_GT(max_num_iterations, 0);
      THROW_CHECK_GE(gradient_tolerance, 0);
      THROW_CHECK_GE(parameter_tolerance, 0);
      THROW_CHECK_GE(function_tolerance, 0);
      THROW_CHECK_GE(cost_threshold, 0);
      THROW_CHECK_GT(initial_trust_region_radius, 0);
    }
  };

  struct Summary {
    // 1/2 ||f(x_0)||^2
    Scalar initial_cost = -1;
    // 1/2 ||f(x)||^2
    Scalar final_cost = -1;
    // max_norm(J'f(x))
    Scalar gradient_max_norm = -1;
    int iterations = -1;
    Status status = HIT_MAX_ITERATIONS;
  };

  const Summary& Solve(const Function& function,
                       Parameters* x_and_min,
                       const Options& options = Options()) {
    THROW_CHECK_NOTNULL(x_and_min);
    options.Check();
    Initialize(function);
    Parameters& x = *x_and_min;
    summary_ = Summary();
    summary_.iterations = 0;

    // Bail out cleanly if the cost function cannot be evaluated at the initial
    // point; there is nothing meaningful the solver can do in that case.
    if (!Update(function, x)) {
      summary_.status = COST_FUNCTION_FAILED;
      return summary_;
    }
    summary_.initial_cost = cost_;
    summary_.final_cost = cost_;

    if (summary_.gradient_max_norm < options.gradient_tolerance) {
      summary_.status = GRADIENT_TOO_SMALL;
      return summary_;
    }

    if (cost_ < options.cost_threshold) {
      summary_.status = COST_TOO_SMALL;
      return summary_;
    }

    Scalar u = 1.0 / options.initial_trust_region_radius;
    Scalar v = 2;

    for (summary_.iterations = 1;
         summary_.iterations < options.max_num_iterations;
         summary_.iterations++) {
      jtj_regularized_ = jtj_;
      const Scalar min_diagonal = 1e-6;
      const Scalar max_diagonal = 1e32;
      for (int i = 0; i < dx_.rows(); ++i) {
        jtj_regularized_(i, i) +=
            u * (std::min)((std::max)(jtj_(i, i), min_diagonal), max_diagonal);
      }

      linear_solver_.compute(jtj_regularized_);
      // If the factorization failed, the regularized normal equations were not
      // solvable. Treat this like a rejected step: shrink the trust region
      // (which increases the regularization) and try again.
      if (linear_solver_.info() != Eigen::Success) {
        u *= v;
        v *= 2;
        continue;
      }
      lm_step_ = linear_solver_.solve(g_);
      dx_ = jacobi_scaling_.asDiagonal() * lm_step_;

      // Adding parameter_tolerance to x.norm() ensures that this
      // works if x is near zero.
      const Scalar parameter_tolerance =
          options.parameter_tolerance *
          (x.norm() + options.parameter_tolerance);
      if (dx_.norm() < parameter_tolerance) {
        summary_.status = RELATIVE_STEP_SIZE_TOO_SMALL;
        break;
      }
      manifold_.Plus(x.data(), dx_.data(), x_new_.data());

      // If the cost function fails to evaluate at the trial point, reject the
      // step and shrink the trust region rather than acting on garbage.
      if (!function(x_new_.data(), f_x_new_.data(), nullptr)) {
        u *= v;
        v *= 2;
        continue;
      }

      const Scalar cost_change = (2 * cost_ - f_x_new_.squaredNorm());
      // TODO: Better more numerically stable evaluation.
      const Scalar model_cost_change = lm_step_.dot(2 * g_ - jtj_ * lm_step_);

      // rho is the ratio of the actual reduction in error to the reduction
      // in error that would be obtained if the problem was linear. See [1]
      // for details.
      Scalar rho(cost_change / model_cost_change);
      if (rho > 0) {
        // Accept the Levenberg-Marquardt step because the linear
        // model fits well.
        x = x_new_;

        if (std::abs(cost_change) < options.function_tolerance) {
          cost_ = f_x_new_.squaredNorm() / 2;
          summary_.status = COST_CHANGE_TOO_SMALL;
          break;
        }

        // The cost function already evaluated successfully at x_new_ == x
        // above, so re-evaluating (now also for the Jacobian) is not expected
        // to fail; guard against it regardless.
        if (!Update(function, x)) {
          summary_.status = COST_FUNCTION_FAILED;
          break;
        }
        if (summary_.gradient_max_norm < options.gradient_tolerance) {
          summary_.status = GRADIENT_TOO_SMALL;
          break;
        }

        if (cost_ < options.cost_threshold) {
          summary_.status = COST_TOO_SMALL;
          break;
        }

        Scalar tmp = Scalar(2 * rho - 1);
        u = u * (std::max)(Scalar(1 / 3.), Scalar(1) - tmp * tmp * tmp);
        v = 2;

      } else {
        // Reject the update because either the normal equations failed to solve
        // or the local linear model was not good (rho < 0).

        // Additionally if the cost change is too small, then terminate.
        if (std::abs(cost_change) < options.function_tolerance) {
          // Terminate
          summary_.status = COST_CHANGE_TOO_SMALL;
          break;
        }

        // Reduce the size of the trust region.
        u *= v;
        v *= 2;
      }
    }

    summary_.final_cost = cost_;
    return summary_;
  }

 private:
  bool Update(const Function& function, const Parameters& x) {
    // Evaluate the residuals and the Jacobian with respect to the ambient
    // parameters, then project it into the tangent space of the manifold at x:
    // J_tangent = J_ambient * d(Plus(x, delta))/d(delta) |_{delta = 0}.
    if constexpr (Manifold::kIsEuclidean) {
      // The tangent space equals the ambient space, so the cost function writes
      // its Jacobian straight into jacobian_ (no projection, no copy).
      if (!function(x.data(), residuals_.data(), jacobian_.data())) {
        return false;
      }
      residuals_ = -residuals_;
    } else {
      if (!function(x.data(), residuals_.data(), jacobian_ambient_.data())) {
        return false;
      }
      residuals_ = -residuals_;
      // PlusJacobian writes a row-major (ambient x tangent) matrix.
      constexpr int kOrder =
          (NUM_TANGENT == 1) ? Eigen::ColMajor : Eigen::RowMajor;
      Eigen::Matrix<Scalar, NUM_PARAMETERS, NUM_TANGENT, kOrder> plus_jacobian;
      manifold_.PlusJacobian(x.data(), plus_jacobian.data());
      jacobian_ = jacobian_ambient_ * plus_jacobian;
    }

    // On the first iteration, compute a diagonal (Jacobi) scaling
    // matrix, which we store as a vector.
    if (summary_.iterations == 0) {
      // jacobi_scaling = 1 / (1 + diagonal(J'J))
      //
      // 1 is added to the denominator to regularize small diagonal
      // entries.
      jacobi_scaling_ = 1.0 / (1.0 + jacobian_.colwise().norm().array());
    }

    // This explicitly computes the normal equations, which is numerically
    // unstable. Nevertheless, it is often good enough and is fast.
    //
    // TODO: Refactor this to allow for DenseQR factorization.
    jacobian_ = jacobian_ * jacobi_scaling_.asDiagonal();
    jtj_ = jacobian_.transpose() * jacobian_;
    g_ = jacobian_.transpose() * residuals_;
    summary_.gradient_max_norm = g_.array().abs().maxCoeff();
    cost_ = residuals_.squaredNorm() / 2;
    return true;
  }

  // Preallocate everything, including temporary storage needed for solving the
  // linear system. This allows reusing the intermediate storage across solves.
  //
  // The (potentially over-aligned) fixed-size Eigen members are declared first
  // and the scalars last so the compiler does not insert padding between an
  // aligned matrix and a scalar (clang-analyzer-optin.performance.Padding).
  Parameters x_new_;
  Tangent dx_, g_, jacobi_scaling_, lm_step_;
  Eigen::Matrix<Scalar, NUM_RESIDUALS, 1> residuals_, f_x_new_;
  // jacobian_ is the tangent-space Jacobian used to form the normal equations.
  // For a non-Euclidean manifold the cost function writes the ambient Jacobian
  // into jacobian_ambient_, which is then projected into jacobian_. For the
  // Euclidean identity manifold the cost function writes jacobian_ directly, so
  // jacobian_ambient_ is unused and sized to zero columns.
  static constexpr int kAmbientJacobianCols =
      Manifold::kIsEuclidean ? 0 : static_cast<int>(NUM_PARAMETERS);
  Eigen::Matrix<Scalar, NUM_RESIDUALS, kAmbientJacobianCols> jacobian_ambient_;
  Eigen::Matrix<Scalar, NUM_RESIDUALS, NUM_TANGENT> jacobian_;
  Eigen::Matrix<Scalar, NUM_TANGENT, NUM_TANGENT> jtj_, jtj_regularized_;
  LinearSolver linear_solver_;
  Scalar cost_;
  Summary summary_;
  // The manifold is stateless for the parameterizations used here, so the
  // default-constructed instance is sufficient.
  Manifold manifold_;

  // Only the number of residuals may be dynamically sized; the parameter and
  // tangent dimensions are static, so nothing else needs allocation.
  void Initialize(const Function& function) {
    if constexpr (NUM_RESIDUALS == Eigen::Dynamic) {
      const int num_residuals = function.NumResiduals();
      residuals_.resize(num_residuals);
      f_x_new_.resize(num_residuals);
      jacobian_ambient_.resize(num_residuals, kAmbientJacobianCols);
      jacobian_.resize(num_residuals, NUM_TANGENT);
    }
  }
};

}  // namespace colmap

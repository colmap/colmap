// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/eigen_alignment.h"

#include <memory>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace colmap {

struct LeastAbsoluteDeviationLinearSolverImpl;

// Least absolute deviations (LAD) fitting via ADMM by solving the problem:
//
//        min || A x - b ||_1
//
// The solution is returned in the vector x and the iterative solver is
// initialized with the given value. This implementation is based on the paper
// "Distributed Optimization and Statistical Learning via the Alternating
// Direction Method of Multipliers" by Boyd et al. and the Matlab implementation
// at https://web.stanford.edu/~boyd/papers/admm/least_abs_deviations/lad.html
struct LeastAbsoluteDeviationSolver {
  struct Options {
    // Augmented Lagrangian parameter.
    double rho = 1.0;

    // Over-relaxation parameter, typical values are between 1.0 and 1.8.
    double alpha = 1.0;

    // Maximum solver iterations.
    int max_num_iterations = 1000;

    // Absolute and relative solution thresholds, as suggested by Boyd et al.
    double absolute_tolerance = 1e-4;
    double relative_tolerance = 1e-2;

    // Tikhonov ridge added to the diagonal of the normal equations A^T A
    // before factorization. Set to a small positive value (e.g., 1e-12) for
    // poorly conditioned but mathematically positive definite systems where
    // CHOLMOD's supernodal Cholesky may report "matrix not positive definite".
    // Default 0 disables regularization (no computational overhead).
    double ridge_regularization = 0;

    enum class SolverType {
      SimplicialLLT,
      SupernodalCholmodLLT,
    };
    SolverType solver_type = SolverType::SimplicialLLT;
  };

  LeastAbsoluteDeviationSolver(const Options& options,
                               const Eigen::SparseMatrix<double>& A);

  // Returns false if the factorization of A^T A failed during construction
  // (e.g., the system is rank deficient or numerically not positive definite),
  // in which case Solve always returns false without producing NaN output.
  bool Valid() const { return valid_; }

  bool Solve(const Eigen::VectorXd& b, Eigen::VectorXd* x) const;

 private:
  const Options& options_;
  const Eigen::SparseMatrix<double>& A_;
  const std::shared_ptr<LeastAbsoluteDeviationLinearSolverImpl> linear_solver_;
  bool valid_ = false;
};

}  // namespace colmap

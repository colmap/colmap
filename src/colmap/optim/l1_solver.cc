// This code is adapted from Theia library (http://theia-sfm.org/),
// with its original L1 solver adapted from
//  "https://web.stanford.edu/~boyd/papers/admm/least_abs_deviations/lad.html"

#pragma once

#include "colmap/optim/l1_solver.h"

#include <colmap/util/logging.h>

#include <Eigen/Cholesky>
#include <Eigen/CholmodSupport>
#include <Eigen/Core>

namespace colmap {
namespace {

  Eigen::VectorXd Shrinkage(const Eigen::VectorXd& vec,
                                   const double kappa) {
    Eigen::ArrayXd zero_vec(vec.size());
    zero_vec.setZero();
    return zero_vec.max(vec.array() - kappa) -
           zero_vec.max(-vec.array() - kappa);
  }

}

  L1Solver::L1Solver(const L1SolverOptions& options, const Eigen::SparseMatrix<double>& A)
      : options_(options), A_(A) {
    // Pre-compute the sparsity pattern.
    const Eigen::SparseMatrix<double> spd_mat = A_.transpose() * A_;
    linear_solver_.compute(spd_mat);
  }

  void L1Solver::Solve(const Eigen::VectorXd& rhs, Eigen::VectorXd* solution) {
    Eigen::VectorXd& x = *solution;
    Eigen::VectorXd z(A_.rows()), u(A_.rows());
    z.setZero();
    u.setZero();

    Eigen::VectorXd a_times_x(A_.rows()), z_old(z.size()), ax_hat(A_.rows());
    // Precompute some convergence terms.
    const double rhs_norm = rhs.norm();
    const double primal_abs_tolerance_eps =
        std::sqrt(A_.rows()) * options_.absolute_tolerance;
    const double dual_abs_tolerance_eps =
        std::sqrt(A_.cols()) * options_.absolute_tolerance;

    const std::string row_format =
        "  % 4d     % 4.4e     % 4.4e     % 4.4e     % 4.4e";
    for (int i = 0; i < options_.max_num_iterations; i++) {
      // Update x.
      x.noalias() = linear_solver_.solve(A_.transpose() * (rhs + z - u));
      if (linear_solver_.info() != Eigen::Success) {
        LOG(ERROR) << "L1 Minimization failed. Could not solve the sparse "
                      "linear system with Cholesky Decomposition";
        return;
      }

      a_times_x.noalias() = A_ * x;
      ax_hat.noalias() = options_.alpha * a_times_x;
      ax_hat.noalias() += (1.0 - options_.alpha) * (z + rhs);

      // Update z and set z_old.
      std::swap(z, z_old);
      z.noalias() = Shrinkage(ax_hat - rhs + u, 1.0 / options_.rho);

      // Update u.
      u.noalias() += ax_hat - z - rhs;

      // Compute the convergence terms.
      const double r_norm = (a_times_x - z - rhs).norm();
      const double s_norm =
          (-options_.rho * A_.transpose() * (z - z_old)).norm();
      const double max_norm = std::max({a_times_x.norm(), z.norm(), rhs_norm});
      const double primal_eps =
          primal_abs_tolerance_eps + options_.relative_tolerance * max_norm;
      const double dual_eps = dual_abs_tolerance_eps +
                              options_.relative_tolerance *
                                  (options_.rho * A_.transpose() * u).norm();

      // Determine if the minimizer has converged.
      if (r_norm < primal_eps && s_norm < dual_eps) {
        break;
      }
    }
  }

}  // namespace colmap

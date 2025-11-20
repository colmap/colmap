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

#include "colmap/optim/least_absolute_deviations.h"

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <memory>

#include <Eigen/CholmodSupport>
#include <Eigen/SparseCholesky>

namespace colmap {
namespace {

Eigen::VectorXd Shrinkage(const Eigen::VectorXd& a, const double kappa) {
  const Eigen::VectorXd a_plus_kappa = a.array() + kappa;
  const Eigen::VectorXd a_minus_kappa = a.array() - kappa;
  return a_plus_kappa.cwiseMin(0) + a_minus_kappa.cwiseMax(0);
}

struct LinearSolver {
  virtual ~LinearSolver() = default;
  virtual bool Solve(const Eigen::VectorXd& b, Eigen::VectorXd* x) = 0;
};

struct SimplicialLLTLinearSolver : public LinearSolver {
  explicit SimplicialLLTLinearSolver(const Eigen::SparseMatrix<double>& A) {
    linear_solver_.compute(A.transpose() * A);
  }

  bool Solve(const Eigen::VectorXd& b, Eigen::VectorXd* x) override {
    x->noalias() = linear_solver_.solve(b);
    return linear_solver_.info() == Eigen::Success;
  }

 private:
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> linear_solver_;
};

struct SupernodalCholmodLLTLinearSolver : public LinearSolver {
  explicit SupernodalCholmodLLTLinearSolver(
      const Eigen::SparseMatrix<double>& A) {
    linear_solver_.compute(A.transpose() * A);
  }

  bool Solve(const Eigen::VectorXd& b, Eigen::VectorXd* x) override {
    x->noalias() = linear_solver_.solve(b);
    return linear_solver_.info() == Eigen::Success;
  }

 private:
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> linear_solver_;
};

}  // namespace

bool SolveLeastAbsoluteDeviations(const LeastAbsoluteDeviationsOptions& options,
                                  const Eigen::SparseMatrix<double>& A,
                                  const Eigen::VectorXd& b,
                                  Eigen::VectorXd* x) {
  THROW_CHECK_NOTNULL(x);
  THROW_CHECK_GT(options.rho, 0);
  THROW_CHECK_GT(options.alpha, 0);
  THROW_CHECK_GT(options.max_num_iterations, 0);
  THROW_CHECK_GE(options.absolute_tolerance, 0);
  THROW_CHECK_GE(options.relative_tolerance, 0);

  if (A.rows() < A.cols()) {
    LOG(INFO) << "Undertermined systems not supported.";
    return false;
  }

  std::unique_ptr<LinearSolver> linear_solver;
  switch (options.solver_type) {
    case LeastAbsoluteDeviationsOptions::SolverType::SimplicialLLT:
      linear_solver = std::make_unique<SimplicialLLTLinearSolver>(A);
      break;
    case LeastAbsoluteDeviationsOptions::SolverType::SupernodalCholmodLLT:
      linear_solver = std::make_unique<SupernodalCholmodLLTLinearSolver>(A);
      break;
    default:
      LOG(FATAL) << "Unknown linear solver type";
  }

  Eigen::VectorXd z = Eigen::VectorXd::Zero(A.rows());
  Eigen::VectorXd z_old(A.rows());
  Eigen::VectorXd u = Eigen::VectorXd::Zero(A.rows());

  Eigen::VectorXd Ax(A.rows());
  Eigen::VectorXd Ax_hat(A.rows());

  const double b_norm = b.norm();
  const double eps_pri_threshold =
      std::sqrt(A.rows()) * options.absolute_tolerance;
  const double eps_dual_threshold =
      std::sqrt(A.cols()) * options.absolute_tolerance;

  for (int i = 0; i < options.max_num_iterations; ++i) {
    if (!linear_solver->Solve(A.transpose() * (b + z - u), x)) {
      return false;
    }

    Ax.noalias() = A * *x;
    Ax_hat.noalias() = options.alpha * Ax + (1 - options.alpha) * (z + b);

    std::swap(z, z_old);
    z.noalias() = Shrinkage(Ax_hat - b + u, 1 / options.rho);

    u.noalias() += Ax_hat - z - b;

    const double r_norm = (Ax - z - b).norm();
    const double s_norm = (-options.rho * A.transpose() * (z - z_old)).norm();
    const double eps_pri =
        eps_pri_threshold + options.relative_tolerance *
                                std::max(b_norm, std::max(Ax.norm(), z.norm()));
    const double eps_dual =
        eps_dual_threshold +
        options.relative_tolerance * (options.rho * A.transpose() * u).norm();

    if (r_norm < eps_pri && s_norm < eps_dual) {
      break;
    }
  }

  return true;
}

}  // namespace colmap

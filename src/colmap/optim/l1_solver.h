// This code is adapted from Theia library (http://theia-sfm.org/),
// with its original L1 solver adapted from
//  "https://web.stanford.edu/~boyd/papers/admm/least_abs_deviations/lad.html"

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/CholmodSupport>
#include <Eigen/Sparse>

namespace colmap {

struct L1SolverOptions {
  int max_num_iterations = 1000;
  // Rho is the augmented Lagrangian parameter.
  double rho = 1.0;
  // Alpha is the over-relaxation parameter (typically between 1.0 and 1.8).
  double alpha = 1.0;

  double absolute_tolerance = 1e-4;
  double relative_tolerance = 1e-2;
};

// An L1 norm (|| A * x - b ||_1) approximation solver based on ADMM
// (alternating direction method of multipliers,
// https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf).
// TODO: L1 solver for dense matrix.
class L1Solver {
 public:
  L1Solver(const L1SolverOptions& options, const Eigen::SparseMatrix<double>& A);

  void Solve(const Eigen::VectorXd& rhs, Eigen::VectorXd* solution);

 private:
  const L1SolverOptions& options_;

  // Matrix A in || Ax - b ||_1.
  const Eigen::SparseMatrix<double> A_;

  // Cholesky linear solver. Since our linear system is an SPD matrix, we can
  // utilize the Cholesky factorization.
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> linear_solver_;
};

}  // namespace colmap

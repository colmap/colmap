// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <Eigen/CholmodSupport>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

namespace colmap {

// Sparse Cholesky solver that tries CHOLMOD supernodal LLT first (fastest for
// large systems) and falls back to Eigen's simplicial LDLT (more numerically
// tolerant) once supernodal reports the matrix is not positive definite.
// Supernodal CHOLMOD aborts at the first non-positive pivot in a dense block,
// which can trigger on mathematically PD but ill-conditioned systems.
class SparseCholeskyWithFallbackSolver {
 public:
  // One-shot factorization (analyze + factorize).
  bool Compute(const Eigen::SparseMatrix<double>& A);

  // For iterative reuse with matrices of identical sparsity but changing
  // values, call AnalyzePattern once then Factorize per iteration.
  void AnalyzePattern(const Eigen::SparseMatrix<double>& A);
  bool Factorize(const Eigen::SparseMatrix<double>& A);

  bool Solve(const Eigen::VectorXd& b, Eigen::VectorXd* x) const;

 private:
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> supernodal_;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_;
  bool use_ldlt_ = false;
};

}  // namespace colmap

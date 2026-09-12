// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/sparse_cholesky.h"

#include "colmap/util/logging.h"

namespace colmap {

void SparseCholeskyWithFallbackSolver::AnalyzePattern(
    const Eigen::SparseMatrix<double>& A) {
  supernodal_.analyzePattern(A);
  // LDLT pattern is analyzed lazily on first fallback, since the common case
  // never needs it.
}

bool SparseCholeskyWithFallbackSolver::Factorize(
    const Eigen::SparseMatrix<double>& A) {
  if (!use_ldlt_) {
    supernodal_.factorize(A);
    if (supernodal_.info() == Eigen::Success) {
      return true;
    }
    LOG(WARNING) << "Supernodal Cholesky factorization failed; falling back "
                    "to simplicial LDLT for ill-conditioned system.";
    ldlt_.analyzePattern(A);
    use_ldlt_ = true;
  }
  ldlt_.factorize(A);
  return ldlt_.info() == Eigen::Success;
}

bool SparseCholeskyWithFallbackSolver::Compute(
    const Eigen::SparseMatrix<double>& A) {
  AnalyzePattern(A);
  return Factorize(A);
}

bool SparseCholeskyWithFallbackSolver::Solve(const Eigen::VectorXd& b,
                                             Eigen::VectorXd* x) const {
  THROW_CHECK_NOTNULL(x);
  if (use_ldlt_) {
    x->noalias() = ldlt_.solve(b);
    return ldlt_.info() == Eigen::Success;
  }
  x->noalias() = supernodal_.solve(b);
  return supernodal_.info() == Eigen::Success;
}

}  // namespace colmap

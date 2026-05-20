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

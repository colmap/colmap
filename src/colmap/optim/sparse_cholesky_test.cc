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

#include "colmap/optim/sparse_cholesky.h"

#include "colmap/util/eigen_matchers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

// 1D Laplacian of a chain of n nodes with the first diagonal entry increased
// by 1 to fix the gauge (otherwise the matrix is rank-deficient). This is the
// pose-graph structure that drives the rotation-averaging fix; it is PD but
// becomes ill-conditioned as n grows.
Eigen::SparseMatrix<double> ChainLaplacianGaugeFixed(int n) {
  Eigen::SparseMatrix<double> A(n, n);
  std::vector<Eigen::Triplet<double>> triplets;
  for (int i = 0; i < n; ++i) {
    double diag = 0;
    if (i > 0) {
      triplets.emplace_back(i, i - 1, -1);
      diag += 1;
    }
    if (i < n - 1) {
      triplets.emplace_back(i, i + 1, -1);
      diag += 1;
    }
    triplets.emplace_back(i, i, diag);
  }
  // Gauge fix: pin node 0.
  triplets.emplace_back(0, 0, 1);
  A.setFromTriplets(triplets.begin(), triplets.end());
  return A;
}

TEST(SparseCholeskyWithFallbackSolver, ComputeAndSolveDiagonal) {
  Eigen::SparseMatrix<double> A(3, 3);
  A.insert(0, 0) = 2;
  A.insert(1, 1) = 3;
  A.insert(2, 2) = 4;

  SparseCholeskyWithFallbackSolver solver;
  ASSERT_TRUE(solver.Compute(A));

  Eigen::VectorXd b(3);
  b << 2, 6, 12;
  Eigen::VectorXd x;
  ASSERT_TRUE(solver.Solve(b, &x));
  EXPECT_THAT(x, EigenMatrixNear(Eigen::Vector3d(1, 2, 3)));
}

TEST(SparseCholeskyWithFallbackSolver, ComputeAndSolveChain) {
  const Eigen::SparseMatrix<double> A = ChainLaplacianGaugeFixed(10);
  const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(A.rows(), 1, 10);

  SparseCholeskyWithFallbackSolver solver;
  ASSERT_TRUE(solver.Compute(A));

  Eigen::VectorXd x;
  ASSERT_TRUE(solver.Solve(b, &x));
  EXPECT_THAT(A * x, EigenMatrixNear(b, 1e-9));
}

TEST(SparseCholeskyWithFallbackSolver,
     AnalyzeAndFactorizeReusedAcrossMatrices) {
  // Same sparsity, different numeric values — mirrors IRLS reuse pattern.
  const Eigen::SparseMatrix<double> A1 = ChainLaplacianGaugeFixed(8);
  Eigen::SparseMatrix<double> A2 = A1;
  for (int i = 0; i < A2.cols(); ++i) {
    A2.coeffRef(i, i) += 0.5;
  }

  SparseCholeskyWithFallbackSolver solver;
  solver.AnalyzePattern(A1);

  const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(A1.rows(), 1, 8);
  Eigen::VectorXd x;

  ASSERT_TRUE(solver.Factorize(A1));
  ASSERT_TRUE(solver.Solve(b, &x));
  EXPECT_THAT(A1 * x, EigenMatrixNear(b, 1e-9));

  ASSERT_TRUE(solver.Factorize(A2));
  ASSERT_TRUE(solver.Solve(b, &x));
  EXPECT_THAT(A2 * x, EigenMatrixNear(b, 1e-9));
}

TEST(SparseCholeskyWithFallbackSolver, ComputeReturnsFalseOnSingularMatrix) {
  // 2x2 rank-1 matrix: [[1,1],[1,1]]. Singular, so both supernodal and LDLT
  // must detect the failure and Compute must return false rather than NaN.
  Eigen::SparseMatrix<double> A(2, 2);
  A.insert(0, 0) = 1;
  A.insert(0, 1) = 1;
  A.insert(1, 0) = 1;
  A.insert(1, 1) = 1;

  SparseCholeskyWithFallbackSolver solver;
  EXPECT_FALSE(solver.Compute(A));
}

TEST(SparseCholeskyWithFallbackSolver, RidgeMakesSingularMatrixSolvable) {
  // Same singular matrix as above with a small ridge added to the diagonal
  // is PD and must factorize successfully.
  Eigen::SparseMatrix<double> A(2, 2);
  A.insert(0, 0) = 1 + 1e-6;
  A.insert(0, 1) = 1;
  A.insert(1, 0) = 1;
  A.insert(1, 1) = 1 + 1e-6;

  SparseCholeskyWithFallbackSolver solver;
  ASSERT_TRUE(solver.Compute(A));

  Eigen::VectorXd b(2);
  b << 2, 2;
  Eigen::VectorXd x;
  ASSERT_TRUE(solver.Solve(b, &x));
  EXPECT_FALSE(x.array().isNaN().any());
}

TEST(SparseCholeskyWithFallbackSolver, FallsBackToLdltOnIndefiniteMatrix) {
  // diag(1, 1, -1e-20) is mathematically indefinite. Supernodal LLT
  // fundamentally requires strictly positive pivots (it takes square roots),
  // so CHOLMOD reliably rejects this across versions. SimplicialLDLT accepts
  // indefinite matrices by allowing negative entries in D. The wrapper must
  // fall back transparently and Solve must return the correct (exact for a
  // diagonal system) solution rather than NaN.
  Eigen::SparseMatrix<double> A(3, 3);
  A.insert(0, 0) = 1;
  A.insert(1, 1) = 1;
  A.insert(2, 2) = -1e-20;

  SparseCholeskyWithFallbackSolver solver;
  ASSERT_TRUE(solver.Compute(A));

  Eigen::VectorXd b(3);
  b << 2, 3, -5e-20;
  Eigen::VectorXd x;
  ASSERT_TRUE(solver.Solve(b, &x));
  EXPECT_THAT(x, EigenMatrixNear(Eigen::Vector3d(2, 3, 5), 1e-10));
}

TEST(SparseCholeskyWithFallbackSolver, IllConditionedChain) {
  // Long chain Laplacian + gauge fix. Mathematically PD but condition number
  // grows as O(n^2). Exercises the regime that motivated the fallback.
  const Eigen::SparseMatrix<double> A = ChainLaplacianGaugeFixed(500);
  const Eigen::VectorXd b = Eigen::VectorXd::Random(A.rows());

  SparseCholeskyWithFallbackSolver solver;
  ASSERT_TRUE(solver.Compute(A));

  Eigen::VectorXd x;
  ASSERT_TRUE(solver.Solve(b, &x));
  EXPECT_FALSE(x.array().isNaN().any());
  EXPECT_THAT(A * x, EigenMatrixNear(b, 1e-6));
}

}  // namespace
}  // namespace colmap

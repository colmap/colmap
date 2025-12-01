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
#include "colmap/util/eigen_matchers.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace colmap {
namespace {

class ParameterizedLeastAbsoluteDeviationsTests
    : public ::testing::TestWithParam<
          LeastAbsoluteDeviationSolver::Options::SolverType> {
 protected:
  static LeastAbsoluteDeviationSolver::Options GetOptions() {
    LeastAbsoluteDeviationSolver::Options options;
    options.solver_type = GetParam();
    return options;
  }
};

TEST_P(ParameterizedLeastAbsoluteDeviationsTests, OverDetermined) {
  Eigen::SparseMatrix<double> A(4, 3);
  for (int i = 0; i < A.rows(); ++i) {
    for (int j = 0; j < A.cols(); ++j) {
      A.insert(i, j) = i * A.cols() + j + 1;
    }
  }
  A.coeffRef(0, 0) = 10;

  Eigen::VectorXd b(A.rows());
  for (int i = 0; i < b.size(); ++i) {
    b(i) = i + 1;
  }

  Eigen::VectorXd x = Eigen::VectorXd::Zero(A.cols());

  LeastAbsoluteDeviationSolver::Options options = GetOptions();
  LeastAbsoluteDeviationSolver solver(options, A);
  EXPECT_TRUE(solver.Solve(b, &x));

  // Reference solution obtained with Boyd's Matlab implementation.
  EXPECT_THAT(x, EigenMatrixNear(Eigen::Vector3d(0, 0, 1 / 3.0)));

  const Eigen::VectorXd residual = A * x - b;
  EXPECT_LE(residual.norm(), 1e-6);
}

TEST_P(ParameterizedLeastAbsoluteDeviationsTests, WellDetermined) {
  Eigen::SparseMatrix<double> A(3, 3);
  for (int i = 0; i < A.rows(); ++i) {
    for (int j = 0; j < A.cols(); ++j) {
      A.insert(i, j) = i * A.cols() + j + 1;
    }
  }
  A.coeffRef(0, 0) = 10;

  Eigen::VectorXd b(A.rows());
  for (int i = 0; i < b.size(); ++i) {
    b(i) = i + 1;
  }

  Eigen::VectorXd x = Eigen::VectorXd::Zero(A.cols());

  LeastAbsoluteDeviationSolver::Options options = GetOptions();
  LeastAbsoluteDeviationSolver solver(options, A);
  EXPECT_TRUE(solver.Solve(b, &x));

  // Reference solution obtained with Boyd's Matlab implementation.
  EXPECT_THAT(x, EigenMatrixNear(Eigen::Vector3d(0, 0, 1 / 3.0)));

  const Eigen::VectorXd residual = A * x - b;
  EXPECT_LE(residual.norm(), 1e-6);
}

TEST_P(ParameterizedLeastAbsoluteDeviationsTests, UnderDetermined) {
  // In this case, the system is rank-deficient and not positive semi-definite.
  Eigen::SparseMatrix<double> A(2, 3);
  Eigen::VectorXd b(A.rows());
  Eigen::VectorXd x = Eigen::VectorXd::Zero(A.cols());
  LeastAbsoluteDeviationSolver::Options options = GetOptions();
  EXPECT_THROW(LeastAbsoluteDeviationSolver(options, A), std::runtime_error);
}

TEST_P(ParameterizedLeastAbsoluteDeviationsTests, SimpleOverdeterminedSystem) {
  Eigen::SparseMatrix<double> A(4, 3);
  A.insert(0, 0) = 1.0;
  A.insert(0, 1) = 0.0;
  A.insert(0, 2) = 0.0;
  A.insert(1, 0) = 0.0;
  A.insert(1, 1) = 1.0;
  A.insert(1, 2) = 0.0;
  A.insert(2, 0) = 0.0;
  A.insert(2, 1) = 0.0;
  A.insert(2, 2) = 1.0;
  A.insert(3, 0) = 1.0;
  A.insert(3, 1) = 1.0;
  A.insert(3, 2) = 1.0;

  Eigen::VectorXd b(4);
  b << 1.0, 2.0, 3.0, 6.0;

  LeastAbsoluteDeviationSolver::Options options = GetOptions();

  Eigen::VectorXd x = Eigen::VectorXd::Zero(3);
  EXPECT_GT((A * x - b).lpNorm<1>(), 1e-1);

  LeastAbsoluteDeviationSolver solver(options, A);
  solver.Solve(b, &x);
  EXPECT_LE((A * x - b).lpNorm<1>(), 1e-6);
}

TEST_P(ParameterizedLeastAbsoluteDeviationsTests, DiagonalSystem) {
  const int n = 5;
  Eigen::SparseMatrix<double> A(n, n);
  for (int i = 0; i < n; ++i) {
    A.insert(i, i) = i + 1.0;
  }

  Eigen::VectorXd b(n);
  for (int i = 0; i < n; ++i) {
    b(i) = (i + 1.0) * (i + 2.0);
  }

  LeastAbsoluteDeviationSolver::Options options = GetOptions();
  options.max_num_iterations = 100;

  Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
  LeastAbsoluteDeviationSolver solver(options, A);
  solver.Solve(b, &x);

  // For diagonal systems, the L1 solution should be close to x_i = b_i / A_ii
  Eigen::VectorXd expected(n);
  for (int i = 0; i < n; ++i) {
    expected(i) = b(i) / A.coeff(i, i);
  }

  EXPECT_LE((x - expected).lpNorm<1>(), 1e-6);
}

TEST_P(ParameterizedLeastAbsoluteDeviationsTests, OverdeterminedWithOutliers) {
  Eigen::SparseMatrix<double> A(6, 2);
  for (int i = 0; i < 6; ++i) {
    A.insert(i, 0) = 1.0;
    A.insert(i, 1) = static_cast<double>(i);
  }

  // Linear relationship b = 2 + 3*x, but with outliers
  Eigen::VectorXd b(6);
  b << 2.0, 5.0, 8.0, 11.0, 1000.0, 17.0;  // b[4] is an outlier

  LeastAbsoluteDeviationSolver::Options options = GetOptions();
  options.max_num_iterations = 1000;

  Eigen::VectorXd x = Eigen::VectorXd::Zero(2);
  LeastAbsoluteDeviationSolver solver(options, A);
  solver.Solve(b, &x);

  // The L1 solution should be more robust to the outlier than L2
  // Expected solution is approximately [2, 3]
  EXPECT_NEAR(x(0), 2.0, 1e-3);
  EXPECT_NEAR(x(1), 3.0, 1e-3);
}

TEST_P(ParameterizedLeastAbsoluteDeviationsTests, IdentityMatrix) {
  // Test with identity matrix - trivial case
  const int n = 4;
  Eigen::SparseMatrix<double> A(n, n);
  for (int i = 0; i < n; ++i) {
    A.insert(i, i) = 1.0;
  }

  Eigen::VectorXd b(n);
  b << 1.0, 2.0, 3.0, 4.0;

  LeastAbsoluteDeviationSolver::Options options = GetOptions();

  Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
  LeastAbsoluteDeviationSolver solver(options, A);
  solver.Solve(b, &x);

  // Solution should be exactly b for identity matrix
  EXPECT_THAT(x, EigenMatrixNear(b, 1e-3));
}

TEST_P(ParameterizedLeastAbsoluteDeviationsTests, ScaledIdentityMatrix) {
  // Test with scaled identity matrix
  const int n = 3;
  const double scale = 5.0;
  Eigen::SparseMatrix<double> A(n, n);
  for (int i = 0; i < n; ++i) {
    A.insert(i, i) = scale;
  }

  Eigen::VectorXd b(n);
  b << 5.0, 10.0, 15.0;

  LeastAbsoluteDeviationSolver::Options options = GetOptions();

  Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
  LeastAbsoluteDeviationSolver solver(options, A);
  solver.Solve(b, &x);

  // Solution should be b / scale
  Eigen::VectorXd expected = b / scale;
  EXPECT_THAT(x, EigenMatrixNear(expected, 1e-3));
}

TEST_P(ParameterizedLeastAbsoluteDeviationsTests, ToleranceSettings) {
  // Test that tighter tolerances produce more accurate results
  Eigen::SparseMatrix<double> A(5, 3);
  // Create a well-conditioned matrix with full column rank
  A.insert(0, 0) = 3.0;
  A.insert(0, 1) = 0.5;
  A.insert(0, 2) = 0.2;
  A.insert(1, 0) = 0.5;
  A.insert(1, 1) = 2.5;
  A.insert(1, 2) = 0.3;
  A.insert(2, 0) = 0.2;
  A.insert(2, 1) = 0.3;
  A.insert(2, 2) = 2.0;
  A.insert(3, 0) = 1.0;
  A.insert(3, 1) = 1.5;
  A.insert(3, 2) = 0.5;
  A.insert(4, 0) = 0.7;
  A.insert(4, 1) = 0.6;
  A.insert(4, 2) = 1.8;

  Eigen::VectorXd b(5);
  b << 1.0, 2.0, 3.0, 4.0, 5.0;

  // Loose tolerance
  Eigen::VectorXd x1 = Eigen::VectorXd::Zero(3);
  {
    LeastAbsoluteDeviationSolver::Options options = GetOptions();
    options.absolute_tolerance = 1e-1;
    options.relative_tolerance = 1e-1;

    LeastAbsoluteDeviationSolver solver(options, A);
    solver.Solve(b, &x1);
  }

  // Tight tolerance
  Eigen::VectorXd x2 = Eigen::VectorXd::Zero(3);
  {
    LeastAbsoluteDeviationSolver::Options options = GetOptions();
    options.absolute_tolerance = 1e-6;
    options.relative_tolerance = 1e-4;
    options.max_num_iterations = 2000;

    LeastAbsoluteDeviationSolver solver(options, A);
    solver.Solve(b, &x2);
  }

  // The tighter tolerance solution should have lower or equal residual
  const double residual1 = (A * x1 - b).lpNorm<1>();
  const double residual2 = (A * x2 - b).lpNorm<1>();
  EXPECT_LT(residual2, 0.99 * residual1);
}

INSTANTIATE_TEST_SUITE_P(
    LeastAbsoluteDeviationsTests,
    ParameterizedLeastAbsoluteDeviationsTests,
    ::testing::Values(
        LeastAbsoluteDeviationSolver::Options::SolverType::SimplicialLLT,
        LeastAbsoluteDeviationSolver::Options::SolverType::
            SupernodalCholmodLLT));

}  // namespace
}  // namespace colmap

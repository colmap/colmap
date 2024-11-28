// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/math/random.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/eigen_matchers.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(SolveLeastAbsoluteDeviations, OverDetermined) {
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

  LeastAbsoluteDeviationsOptions options;
  EXPECT_TRUE(SolveLeastAbsoluteDeviations(options, A, b, &x));

  // Reference solution obtained with Boyd's Matlab implementation.
  EXPECT_THAT(x, EigenMatrixNear(Eigen::Vector3d(0, 0, 1 / 3.0)));

  const Eigen::VectorXd residual = A * x - b;
  EXPECT_LE(residual.norm(), 1e-6);
}

TEST(SolveLeastAbsoluteDeviations, WellDetermined) {
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

  LeastAbsoluteDeviationsOptions options;
  EXPECT_TRUE(SolveLeastAbsoluteDeviations(options, A, b, &x));

  // Reference solution obtained with Boyd's Matlab implementation.
  EXPECT_THAT(x, EigenMatrixNear(Eigen::Vector3d(0, 0, 1 / 3.0)));

  const Eigen::VectorXd residual = A * x - b;
  EXPECT_LE(residual.norm(), 1e-6);
}

TEST(SolveLeastAbsoluteDeviations, UnderDetermined) {
  // In this case, the system is rank-deficient and not positive semi-definite.
  Eigen::SparseMatrix<double> A(2, 3);
  Eigen::VectorXd b(A.rows());
  Eigen::VectorXd x = Eigen::VectorXd::Zero(A.cols());
  LeastAbsoluteDeviationsOptions options;
  EXPECT_FALSE(SolveLeastAbsoluteDeviations(options, A, b, &x));
}

}  // namespace
}  // namespace colmap

// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#define TEST_NAME "optim/least_absolute_deviations"
#include "util/testing.h"

#include <Eigen/Dense>

#include "optim/least_absolute_deviations.h"
#include "util/random.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestOverDetermined) {
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
  BOOST_CHECK(SolveLeastAbsoluteDeviations(options, A, b, &x));

  // Reference solution obtained with Boyd's Matlab implementation.
  const Eigen::Vector3d x_ref(0, 0, 1 / 3.0);
  BOOST_CHECK(x.isApprox(x_ref));

  const Eigen::VectorXd residual = A * x - b;
  BOOST_CHECK_LE(residual.norm(), 1e-6);
}

BOOST_AUTO_TEST_CASE(TestWellDetermined) {
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
  BOOST_CHECK(SolveLeastAbsoluteDeviations(options, A, b, &x));

  // Reference solution obtained with Boyd's Matlab implementation.
  const Eigen::Vector3d x_ref(0, 0, 1 / 3.0);
  BOOST_CHECK(x.isApprox(x_ref));

  const Eigen::VectorXd residual = A * x - b;
  BOOST_CHECK_LE(residual.norm(), 1e-6);
}

BOOST_AUTO_TEST_CASE(TestUnderDetermined) {
  // In this case, the system is rank-deficient and not positive semi-definite.
  Eigen::SparseMatrix<double> A(2, 3);
  Eigen::VectorXd b(A.rows());
  Eigen::VectorXd x = Eigen::VectorXd::Zero(A.cols());
  LeastAbsoluteDeviationsOptions options;
  BOOST_CHECK(!SolveLeastAbsoluteDeviations(options, A, b, &x));
}

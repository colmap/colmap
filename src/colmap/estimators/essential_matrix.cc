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

#include "colmap/estimators/essential_matrix.h"

#include "colmap/estimators/utils.h"
#include "colmap/math/math.h"
#include "colmap/math/polynomial.h"
#include "colmap/util/logging.h"

#include <complex>

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

namespace colmap {

void EssentialMatrixFivePointEstimator::Estimate(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    std::vector<M_t>* models) {
  CHECK_EQ(points1.size(), points2.size());
  CHECK(models != nullptr);

  models->clear();

  // Step 1: Extraction of the nullspace x, y, z, w.

  Eigen::Matrix<double, Eigen::Dynamic, 9> Q(points1.size(), 9);
  for (size_t i = 0; i < points1.size(); ++i) {
    const double x1_0 = points1[i](0);
    const double x1_1 = points1[i](1);
    const double x2_0 = points2[i](0);
    const double x2_1 = points2[i](1);
    Q(i, 0) = x1_0 * x2_0;
    Q(i, 1) = x1_1 * x2_0;
    Q(i, 2) = x2_0;
    Q(i, 3) = x1_0 * x2_1;
    Q(i, 4) = x1_1 * x2_1;
    Q(i, 5) = x2_1;
    Q(i, 6) = x1_0;
    Q(i, 7) = x1_1;
    Q(i, 8) = 1;
  }

  // Extract the 4 Eigen vectors corresponding to the smallest singular values.
  const Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
      Q, Eigen::ComputeFullV);
  const Eigen::Matrix<double, 9, 4> E = svd.matrixV().block<9, 4>(0, 5);

  // Step 3: Gauss-Jordan elimination with partial pivoting on A.

  Eigen::Matrix<double, 10, 20> A;
#include "colmap/estimators/essential_matrix_poly.h"
  Eigen::Matrix<double, 10, 10> AA =
      A.block<10, 10>(0, 0).partialPivLu().solve(A.block<10, 10>(0, 10));

  // Step 4: Expansion of the determinant polynomial of the 3x3 polynomial
  //         matrix B to obtain the tenth degree polynomial.

  Eigen::Matrix<double, 13, 3> B;
  for (size_t i = 0; i < 3; ++i) {
    B(0, i) = 0;
    B(4, i) = 0;
    B(8, i) = 0;
    B.block<3, 1>(1, i) = AA.block<1, 3>(i * 2 + 4, 0);
    B.block<3, 1>(5, i) = AA.block<1, 3>(i * 2 + 4, 3);
    B.block<4, 1>(9, i) = AA.block<1, 4>(i * 2 + 4, 6);
    B.block<3, 1>(0, i) -= AA.block<1, 3>(i * 2 + 5, 0);
    B.block<3, 1>(4, i) -= AA.block<1, 3>(i * 2 + 5, 3);
    B.block<4, 1>(8, i) -= AA.block<1, 4>(i * 2 + 5, 6);
  }

  // Step 5: Extraction of roots from the degree 10 polynomial.
  Eigen::Matrix<double, 11, 1> coeffs;
#include "colmap/estimators/essential_matrix_coeffs.h"

  Eigen::VectorXd roots_real;
  Eigen::VectorXd roots_imag;
  if (!FindPolynomialRootsCompanionMatrix(coeffs, &roots_real, &roots_imag)) {
    return;
  }

  models->reserve(roots_real.size());

  for (Eigen::VectorXd::Index i = 0; i < roots_imag.size(); ++i) {
    const double kMaxRootImag = 1e-10;
    if (std::abs(roots_imag(i)) > kMaxRootImag) {
      continue;
    }

    const double z1 = roots_real(i);
    const double z2 = z1 * z1;
    const double z3 = z2 * z1;
    const double z4 = z3 * z1;

    Eigen::Matrix3d Bz;
    for (size_t j = 0; j < 3; ++j) {
      Bz(j, 0) = B(0, j) * z3 + B(1, j) * z2 + B(2, j) * z1 + B(3, j);
      Bz(j, 1) = B(4, j) * z3 + B(5, j) * z2 + B(6, j) * z1 + B(7, j);
      Bz(j, 2) = B(8, j) * z4 + B(9, j) * z3 + B(10, j) * z2 + B(11, j) * z1 +
                 B(12, j);
    }

    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(Bz, Eigen::ComputeFullV);
    const Eigen::Vector3d X = svd.matrixV().block<3, 1>(0, 2);

    const double kMaxX3 = 1e-10;
    if (std::abs(X(2)) < kMaxX3) {
      continue;
    }

    Eigen::MatrixXd essential_vec = E.col(0) * (X(0) / X(2)) +
                                    E.col(1) * (X(1) / X(2)) + E.col(2) * z1 +
                                    E.col(3);
    essential_vec /= essential_vec.norm();

    const Eigen::Matrix3d essential_matrix =
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
            essential_vec.data());
    models->push_back(essential_matrix);
  }
}

void EssentialMatrixFivePointEstimator::Residuals(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    const M_t& E,
    std::vector<double>* residuals) {
  ComputeSquaredSampsonError(points1, points2, E, residuals);
}

void EssentialMatrixEightPointEstimator::Estimate(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    std::vector<M_t>* models) {
  CHECK_EQ(points1.size(), points2.size());
  CHECK(models != nullptr);

  models->clear();

  // Center and normalize image points for better numerical stability.
  std::vector<X_t> normed_points1;
  std::vector<Y_t> normed_points2;
  Eigen::Matrix3d normed_from_orig1;
  Eigen::Matrix3d normed_from_orig2;
  CenterAndNormalizeImagePoints(points1, &normed_points1, &normed_from_orig1);
  CenterAndNormalizeImagePoints(points2, &normed_points2, &normed_from_orig2);

  // Setup homogeneous linear equation as x2' * F * x1 = 0.
  Eigen::Matrix<double, Eigen::Dynamic, 9> cmatrix(points1.size(), 9);
  for (size_t i = 0; i < points1.size(); ++i) {
    cmatrix.block<1, 3>(i, 0) = normed_points1[i].homogeneous();
    cmatrix.block<1, 3>(i, 0) *= normed_points2[i].x();
    cmatrix.block<1, 3>(i, 3) = normed_points1[i].homogeneous();
    cmatrix.block<1, 3>(i, 3) *= normed_points2[i].y();
    cmatrix.block<1, 3>(i, 6) = normed_points1[i].homogeneous();
  }

  // Solve for the nullspace of the constraint matrix.
  Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> cmatrix_svd(
      cmatrix, Eigen::ComputeFullV);
  const Eigen::VectorXd ematrix_nullspace = cmatrix_svd.matrixV().col(8);
  const Eigen::Map<const Eigen::Matrix3d> ematrix_t(ematrix_nullspace.data());

  // De-normalize to image points.
  const Eigen::Matrix3d E_raw =
      normed_from_orig2.transpose() * ematrix_t.transpose() * normed_from_orig1;

  // Enforcing the internal constraint that two singular values must be equal
  // and one must be zero.
  Eigen::JacobiSVD<Eigen::Matrix3d> E_raw_svd(
      E_raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d singular_values = E_raw_svd.singularValues();
  singular_values(0) = (singular_values(0) + singular_values(1)) / 2.0;
  singular_values(1) = singular_values(0);
  singular_values(2) = 0.0;
  const Eigen::Matrix3d E = E_raw_svd.matrixU() * singular_values.asDiagonal() *
                            E_raw_svd.matrixV().transpose();

  models->resize(1);
  (*models)[0] = E;
}

void EssentialMatrixEightPointEstimator::Residuals(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    const M_t& E,
    std::vector<double>* residuals) {
  ComputeSquaredSampsonError(points1, points2, E, residuals);
}

}  // namespace colmap

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

#include "colmap/estimators/fundamental_matrix.h"

#include "colmap/estimators/utils.h"
#include "colmap/math/polynomial.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <cfloat>
#include <complex>
#include <vector>

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

namespace colmap {

void FundamentalMatrixSevenPointEstimator::Estimate(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    std::vector<M_t>* models) {
  THROW_CHECK_EQ(points1.size(), 7);
  THROW_CHECK_EQ(points2.size(), 7);
  THROW_CHECK(models != nullptr);

  models->clear();

  // Note that no normalization of the points is necessary here.

  // Setup system of equations: [points2(i,:), 1]' * F * [points1(i,:), 1]'.
  Eigen::Matrix<double, 7, 9> A;
  for (size_t i = 0; i < 7; ++i) {
    const double x0 = points1[i](0);
    const double y0 = points1[i](1);
    const double x1 = points2[i](0);
    const double y1 = points2[i](1);
    A(i, 0) = x1 * x0;
    A(i, 1) = x1 * y0;
    A(i, 2) = x1;
    A(i, 3) = y1 * x0;
    A(i, 4) = y1 * y0;
    A(i, 5) = y1;
    A(i, 6) = x0;
    A(i, 7) = y0;
    A(i, 8) = 1;
  }

  // 9 unknowns with 7 equations, so we have 2D null space.
  Eigen::JacobiSVD<Eigen::Matrix<double, 7, 9>> svd(A, Eigen::ComputeFullV);
  const Eigen::Matrix<double, 9, 9>& f = svd.matrixV();
  Eigen::Matrix<double, 1, 9> f1 = f.col(7);
  Eigen::Matrix<double, 1, 9> f2 = f.col(8);

  f1 -= f2;

  // Normalize, such that lambda + mu = 1
  // and add constraint det(F) = det(lambda * f1 + (1 - lambda) * f2).

  const double t0 = f1(4) * f1(8) - f1(5) * f1(7);
  const double t1 = f1(3) * f1(8) - f1(5) * f1(6);
  const double t2 = f1(3) * f1(7) - f1(4) * f1(6);
  const double t3 = f2(4) * f2(8) - f2(5) * f2(7);
  const double t4 = f2(3) * f2(8) - f2(5) * f2(6);
  const double t5 = f2(3) * f2(7) - f2(4) * f2(6);

  Eigen::Vector4d coeffs;
  coeffs(0) = f1(0) * t0 - f1(1) * t1 + f1(2) * t2;
  coeffs(1) = f2(0) * t0 - f2(1) * t1 + f2(2) * t2 -
              f2(3) * (f1(1) * f1(8) - f1(2) * f1(7)) +
              f2(4) * (f1(0) * f1(8) - f1(2) * f1(6)) -
              f2(5) * (f1(0) * f1(7) - f1(1) * f1(6)) +
              f2(6) * (f1(1) * f1(5) - f1(2) * f1(4)) -
              f2(7) * (f1(0) * f1(5) - f1(2) * f1(3)) +
              f2(8) * (f1(0) * f1(4) - f1(1) * f1(3));
  coeffs(2) = f1(0) * t3 - f1(1) * t4 + f1(2) * t5 -
              f1(3) * (f2(1) * f2(8) - f2(2) * f2(7)) +
              f1(4) * (f2(0) * f2(8) - f2(2) * f2(6)) -
              f1(5) * (f2(0) * f2(7) - f2(1) * f2(6)) +
              f1(6) * (f2(1) * f2(5) - f2(2) * f2(4)) -
              f1(7) * (f2(0) * f2(5) - f2(2) * f2(3)) +
              f1(8) * (f2(0) * f2(4) - f2(1) * f2(3));
  coeffs(3) = f2(0) * t3 - f2(1) * t4 + f2(2) * t5;

  Eigen::VectorXd roots_real;
  Eigen::VectorXd roots_imag;
  if (!FindPolynomialRootsCompanionMatrix(coeffs, &roots_real, &roots_imag)) {
    return;
  }

  models->reserve(roots_real.size());

  for (Eigen::VectorXd::Index i = 0; i < roots_real.size(); ++i) {
    const double kMaxRootImag = 1e-10;
    if (std::abs(roots_imag(i)) > kMaxRootImag) {
      continue;
    }

    const double lambda = roots_real(i);
    const double mu = 1;

    Eigen::MatrixXd F = lambda * f1 + mu * f2;

    F.resize(3, 3);

    const double kEps = 1e-10;
    if (std::abs(F(2, 2)) < kEps) {
      continue;
    }

    F /= F(2, 2);

    models->push_back(F.transpose());
  }
}

void FundamentalMatrixSevenPointEstimator::Residuals(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    const M_t& F,
    std::vector<double>* residuals) {
  ComputeSquaredSampsonError(points1, points2, F, residuals);
}

void FundamentalMatrixEightPointEstimator::Estimate(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    std::vector<M_t>* models) {
  THROW_CHECK_EQ(points1.size(), points2.size());
  THROW_CHECK(models != nullptr);

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
  const Eigen::VectorXd cmatrix_nullspace = cmatrix_svd.matrixV().col(8);
  const Eigen::Map<const Eigen::Matrix3d> ematrix_t(cmatrix_nullspace.data());

  // Enforcing the internal constraint that two singular values must non-zero
  // and one must be zero.
  Eigen::JacobiSVD<Eigen::Matrix3d> fmatrix_svd(
      ematrix_t.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d singular_values = fmatrix_svd.singularValues();
  singular_values(2) = 0.0;
  const Eigen::Matrix3d F = fmatrix_svd.matrixU() *
                            singular_values.asDiagonal() *
                            fmatrix_svd.matrixV().transpose();

  models->resize(1);
  (*models)[0] = normed_from_orig2.transpose() * F * normed_from_orig1;
}

void FundamentalMatrixEightPointEstimator::Residuals(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    const M_t& E,
    std::vector<double>* residuals) {
  ComputeSquaredSampsonError(points1, points2, E, residuals);
}

}  // namespace colmap

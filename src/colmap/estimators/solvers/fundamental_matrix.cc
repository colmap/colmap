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

#include "colmap/estimators/solvers/fundamental_matrix.h"

#include "colmap/estimators/cost_functions/tiny_manifold.h"
#include "colmap/estimators/cost_functions/tiny_sampson_error.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/normalization.h"
#include "colmap/math/polynomial.h"
#include "colmap/optim/tiny_solver.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <cfloat>
#include <cmath>
#include <vector>

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

namespace colmap {
namespace {

// The 7-DoF manifold of a factorized fundamental matrix: the two rotations of
// its SVD on SO(3), plus the singular value ratio as a 1-D Euclidean
// parameter. The ambient layout matches
// TinyFundamentalSampsonErrorCostFunctor: [qU (xyzw), qV (xyzw), sigma].
using FactorizedFundamentalMatrixManifold =
    ProductManifold<EigenQuaternionManifold,
                    EigenQuaternionManifold,
                    EuclideanManifold<1>>;

// Factorize F = U * diag(1, sigma, 0) * V^T into [qU, qV, sigma], dropping any
// rank-3 component as the eight-point estimator does. U and V are sign
// corrected to proper rotations so they can be represented by quaternions,
// which at most negates F and leaves the Sampson error unchanged.
bool FactorizeFundamentalMatrix(const Eigen::Matrix3d& F,
                                Eigen::Matrix<double, 9, 1>* params) {
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      F, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d& singular_values = svd.singularValues();
  // A zero or numerically rank-1 matrix has no meaningful factorization, as
  // U's second column is arbitrary once the second singular value vanishes.
  constexpr double kMinSingularValueRatio = 1e-12;
  if (!(singular_values(0) > 0) ||
      singular_values(1) <= kMinSingularValueRatio * singular_values(0)) {
    return false;
  }

  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();
  if (U.determinant() < 0) {
    U = -U;
  }
  if (V.determinant() < 0) {
    V = -V;
  }

  params->segment<4>(0) = Eigen::Quaterniond(U).normalized().coeffs();
  params->segment<4>(4) = Eigen::Quaterniond(V).normalized().coeffs();
  (*params)(8) = singular_values(1) / singular_values(0);
  return params->allFinite();
}

// Rescale a normalizing transform and its points, keeping the centroid.
void RescaleNormalizedImagePoints(double factor,
                                  std::vector<Eigen::Vector2d>* normed_points,
                                  Eigen::Matrix3d* normed_from_orig) {
  for (Eigen::Vector2d& point : *normed_points) {
    point *= factor;
  }
  normed_from_orig->topRows<2>() *= factor;
}

}  // namespace

void FundamentalMatrixSevenPointEstimator::Estimate(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    std::vector<M_t>* models) {
  THROW_CHECK_EQ(points1.size(), 7);
  THROW_CHECK_EQ(points2.size(), 7);
  THROW_CHECK(models != nullptr);

  models->clear();

  // Setup system of equations: [points2(i,:), 1]' * F * [points1(i,:), 1]'.
  Eigen::Matrix<double, 9, 7> A;
  for (size_t i = 0; i < 7; ++i) {
    A.col(i) << points1[i].x() * points2[i].homogeneous(),
        points1[i].y() * points2[i].homogeneous(), points2[i].homogeneous();
  }

  // 9 unknowns with 7 equations, so we have 2D null space.
  Eigen::Matrix<double, 9, 9> Q = A.fullPivHouseholderQr().matrixQ();

  // Normalize, such that lambda + mu = 1
  // and add constraint det(F) = det(lambda * f1 + (1 - lambda) * f2).

  auto f1 = Q.col(7);
  auto f2 = Q.col(8);
  f1 -= f2;

  const double t0 = f1(4) * f1(8) - f1(5) * f1(7);
  const double t1 = f1(3) * f1(8) - f1(5) * f1(6);
  const double t2 = f1(3) * f1(7) - f1(4) * f1(6);
  const double t3 = f2(4) * f2(8) - f2(5) * f2(7);
  const double t4 = f2(3) * f2(8) - f2(5) * f2(6);
  const double t5 = f2(3) * f2(7) - f2(4) * f2(6);

  Eigen::Vector4d coeffs;
  coeffs(0) = f1(0) * t0 - f1(1) * t1 + f1(2) * t2;
  if (std::abs(coeffs(0)) < 1e-16) {
    return;
  }

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

  coeffs.tail<3>() /= coeffs(0);

  Eigen::Vector3d roots;
  const int num_roots =
      FindCubicPolynomialRoots(coeffs(1), coeffs(2), coeffs(3), &roots);

  models->reserve(num_roots);
  for (int i = 0; i < num_roots; ++i) {
    const Eigen::Matrix<double, 9, 1> F = (f1 * roots[i] + f2).normalized();
    models->push_back(Eigen::Map<const Eigen::Matrix3d>(F.data()));
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
  THROW_CHECK_GE(points1.size(), 8);
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
  Eigen::Matrix<double, Eigen::Dynamic, 9> A(points1.size(), 9);
  for (size_t i = 0; i < points1.size(); ++i) {
    A.row(i) << normed_points2[i].x() *
                    normed_points1[i].transpose().homogeneous(),
        normed_points2[i].y() * normed_points1[i].transpose().homogeneous(),
        normed_points1[i].transpose().homogeneous();
  }

  // Solve for the nullspace of the constraint matrix.
  Eigen::Matrix3d Q;
  if (points1.size() == 8) {
    Eigen::Matrix<double, 9, 9> QQ =
        A.transpose().householderQr().householderQ();
    Q = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
        QQ.col(8).data());
  } else {
    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
        A, Eigen::ComputeFullV);
    Q = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
        svd.matrixV().col(8).data());
  }

  // Enforcing the internal constraint that two singular values must non-zero
  // and one must be zero.
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      Q, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d singular_values = svd.singularValues();
  singular_values(2) = 0.0;
  const Eigen::Matrix3d F =
      svd.matrixU() * singular_values.asDiagonal() * svd.matrixV().transpose();

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

bool RefineFundamentalMatrixSampson(const std::vector<Eigen::Vector2d>& points1,
                                    const std::vector<Eigen::Vector2d>& points2,
                                    Eigen::Matrix3d* F) {
  THROW_CHECK_EQ(points1.size(), points2.size());
  THROW_CHECK_GE(points1.size(),
                 FundamentalMatrixSampsonEstimator::kMinNumSamples);
  THROW_CHECK_NOTNULL(F);

  // Center and normalize image points for better numerical stability, as in the
  // eight-point estimator. On raw pixel coordinates the 7x7 normal equations
  // of the refinement are severely ill-conditioned.
  std::vector<Eigen::Vector2d> normed_points1;
  std::vector<Eigen::Vector2d> normed_points2;
  Eigen::Matrix3d normed_from_orig1;
  Eigen::Matrix3d normed_from_orig2;
  CenterAndNormalizeImagePoints(points1, &normed_points1, &normed_from_orig1);
  CenterAndNormalizeImagePoints(points2, &normed_points2, &normed_from_orig2);

  // Rescale both views by a common factor. The Sampson error is invariant to a
  // common scale, but not to two different ones, which would weight the two
  // terms of its denominator differently.
  const double scale1 = normed_from_orig1(0, 0);
  const double scale2 = normed_from_orig2(0, 0);
  const double scale = std::sqrt(scale1 * scale2);
  RescaleNormalizedImagePoints(
      scale / scale1, &normed_points1, &normed_from_orig1);
  RescaleNormalizedImagePoints(
      scale / scale2, &normed_points2, &normed_from_orig2);

  // Inverse of the map that the eight-point estimator applies to its solution.
  Eigen::Matrix<double, 9, 1> params;
  if (!FactorizeFundamentalMatrix(normed_from_orig2.transpose().inverse() *
                                      (*F) * normed_from_orig1.inverse(),
                                  &params)) {
    return false;
  }

  // Plain least squares: the points are assumed to be the inlier set, so
  // robustness comes from the RANSAC inlier selection.
  const TinyFundamentalSampsonErrorCostFunctor functor(normed_points1,
                                                       normed_points2);
  using Solver =
      TinySolver<decltype(functor), FactorizedFundamentalMatrixManifold>;
  Solver solver;
  Solver::Options options;
  options.max_num_iterations = 25;
  solver.Solve(functor, &params, options);

  if (!params.allFinite()) {
    return false;
  }

  *F = normed_from_orig2.transpose() *
       TinyFundamentalSampsonErrorCostFunctor::FundamentalFromParams(
           params.data()) *
       normed_from_orig1;
  return true;
}

bool FundamentalMatrixSampsonEstimator::Refine(const std::vector<X_t>& points1,
                                               const std::vector<Y_t>& points2,
                                               M_t* F) {
  return RefineFundamentalMatrixSampson(points1, points2, F);
}

void FundamentalMatrixSampsonEstimator::Residuals(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    const M_t& F,
    std::vector<double>* residuals) {
  ComputeSquaredSampsonError(points1, points2, F, residuals);
}

}  // namespace colmap

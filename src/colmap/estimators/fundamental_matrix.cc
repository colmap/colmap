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
#include <Eigen/SVD>
#include <PoseLib/solvers/relpose_7pt.h>

namespace colmap {

void FundamentalMatrixSevenPointEstimator::Estimate(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    std::vector<M_t>* models) {
  CHECK_EQ(points1.size(), 7);
  CHECK_EQ(points2.size(), 7);
  CHECK(models != nullptr);

  models->clear();

  thread_local static std::vector<Eigen::Vector3d> points1_h(7);
  thread_local static std::vector<Eigen::Vector3d> points2_h(7);
  for (int i = 0; i < 7; ++i) {
    points1_h[i] = points1[i].homogeneous();
    points2_h[i] = points2[i].homogeneous();
  }

  poselib::relpose_7pt(points1_h, points2_h, models);
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

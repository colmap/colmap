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

#include "colmap/estimators/solvers/essential_matrix.h"

#include "colmap/estimators/cost_functions/tiny_manifold.h"
#include "colmap/estimators/cost_functions/tiny_sampson_error.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/polynomial.h"
#include "colmap/optim/tiny_solver.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <PoseLib/solvers/relpose_5pt.h>

namespace colmap {
namespace {

// The 5-DoF manifold of a relative pose (rotation on SO(3), translation on the
// unit sphere), matching the block layout of Rigid3d::params
// ([qx, qy, qz, qw, tx, ty, tz]).
using RelativePoseManifold =
    ProductManifold<EigenQuaternionManifold, SphereManifold<3>>;

}  // namespace

void EssentialMatrixFivePointEstimator::Estimate(
    const std::vector<X_t>& cam_rays1,
    const std::vector<Y_t>& cam_rays2,
    std::vector<M_t>* models) {
  THROW_CHECK_EQ(cam_rays1.size(), cam_rays2.size());
  THROW_CHECK_GE(cam_rays1.size(), kMinNumSamples);
  THROW_CHECK_NOTNULL(models)->clear();

  // PoseLib's 5-point solver only supports the minimal case; the non-minimal
  // case falls through to the SVD-based solver below.
  if (cam_rays1.size() == kMinNumSamples) {
    std::vector<M_t> candidate_models;
    poselib::relpose_5pt(cam_rays1, cam_rays2, &candidate_models);
    // Keep only hypotheses whose minimal sample is in front of both cameras,
    // pruning geometrically invalid essential matrices before they are scored.
    Rigid3d cam2_from_cam1;
    std::vector<int> valid_indices;
    for (const M_t& candidate_model : candidate_models) {
      PoseFromEssentialMatrix(candidate_model,
                              cam_rays1,
                              cam_rays2,
                              &cam2_from_cam1,
                              &valid_indices);
      if (valid_indices.size() == kMinNumSamples) {
        models->push_back(candidate_model);
      }
    }
    return;
  }

  // Setup system of equations: cam_rays2(i)' * E * cam_rays1(i) = 0.

  Eigen::Matrix<double, Eigen::Dynamic, 9> Q(cam_rays1.size(), 9);
  for (size_t i = 0; i < cam_rays1.size(); ++i) {
    Q.row(i) << cam_rays2[i].x() * cam_rays1[i].transpose(),
        cam_rays2[i].y() * cam_rays1[i].transpose(),
        cam_rays2[i].z() * cam_rays1[i].transpose();
  }

  // Step 1: Extraction of the nullspace. The minimal case is handled by
  // PoseLib above, so we always reach this with an over-determined system.

  const Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
      Q, Eigen::ComputeFullV);
  const Eigen::Matrix<double, 9, 4> E = svd.matrixV().rightCols<4>();

  // Step 2: Gauss-Jordan elimination with partial pivoting on A.

  Eigen::Matrix<double, 10, 20> A;
#include "colmap/estimators/solvers/essential_matrix_poly.h"
  const Eigen::Matrix<double, 10, 10> AA =
      A.block<10, 10>(0, 0).partialPivLu().solve(A.block<10, 10>(0, 10));

  // Step 3: Expansion of the determinant polynomial of the 3x3 polynomial
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

  // Step 4: Extraction of roots from the degree 10 polynomial.
  Eigen::Matrix<double, 11, 1> coeffs;
#include "colmap/estimators/solvers/essential_matrix_coeffs.h"

  Eigen::VectorXd roots_real;
  Eigen::VectorXd roots_imag;
  if (!FindPolynomialRootsCompanionMatrix(coeffs, &roots_real, &roots_imag)) {
    return;
  }

  const int num_roots = roots_real.size();
  models->reserve(num_roots);

  for (int i = 0; i < num_roots; ++i) {
    const double kMaxRootImag = 1e-10;
    if (std::abs(roots_imag(i)) > kMaxRootImag) {
      continue;
    }

    const double z1 = roots_real(i);
    const double z2 = z1 * z1;
    const double z3 = z2 * z1;
    const double z4 = z3 * z1;

    Eigen::Matrix3d Bz;
    for (int j = 0; j < 3; ++j) {
      Bz(j, 0) = B(0, j) * z3 + B(1, j) * z2 + B(2, j) * z1 + B(3, j);
      Bz(j, 1) = B(4, j) * z3 + B(5, j) * z2 + B(6, j) * z1 + B(7, j);
      Bz(j, 2) = B(8, j) * z4 + B(9, j) * z3 + B(10, j) * z2 + B(11, j) * z1 +
                 B(12, j);
    }

    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(Bz, Eigen::ComputeFullV);
    const Eigen::Vector3d X = svd.matrixV().rightCols<1>();

    const double kMaxX3 = 1e-10;
    if (std::abs(X(2)) < kMaxX3) {
      continue;
    }

    const Eigen::Matrix<double, 9, 1> e =
        (E.col(0) * (X(0) / X(2)) + E.col(1) * (X(1) / X(2)) + E.col(2) * z1 +
         E.col(3))
            .normalized();

    models->push_back(
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
            e.data()));
  }
}

void EssentialMatrixFivePointEstimator::Residuals(
    const std::vector<X_t>& cam_rays1,
    const std::vector<Y_t>& cam_rays2,
    const M_t& E,
    std::vector<double>* residuals) {
  ComputeSquaredSampsonErrorWithCheirality(cam_rays1, cam_rays2, E, residuals);
}

void EssentialMatrixEightPointEstimator::Estimate(
    const std::vector<X_t>& cam_rays1,
    const std::vector<Y_t>& cam_rays2,
    std::vector<M_t>* models) {
  THROW_CHECK_EQ(cam_rays1.size(), cam_rays2.size());
  THROW_CHECK_GE(cam_rays1.size(), 8);
  THROW_CHECK_NOTNULL(models)->clear();

  // Setup homogeneous linear equation as x2' * E * x1 = 0.
  Eigen::Matrix<double, Eigen::Dynamic, 9> A(cam_rays1.size(), 9);
  for (size_t i = 0; i < cam_rays1.size(); ++i) {
    A.row(i) << cam_rays2[i].x() * cam_rays1[i].transpose(),
        cam_rays2[i].y() * cam_rays1[i].transpose(),
        cam_rays2[i].z() * cam_rays1[i].transpose();
  }

  // Solve for the nullspace of the constraint matrix.
  Eigen::Matrix3d Q;
  if (cam_rays1.size() == 8) {
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

  // Enforcing the internal constraint that two singular values must be non-zero
  // and one must be zero.
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      Q, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d singular_values = svd.singularValues();
  singular_values(2) = 0.0;
  const Eigen::Matrix3d E =
      svd.matrixU() * singular_values.asDiagonal() * svd.matrixV().transpose();

  models->resize(1);
  (*models)[0] = E;
}

void EssentialMatrixEightPointEstimator::Residuals(
    const std::vector<X_t>& cam_rays1,
    const std::vector<Y_t>& cam_rays2,
    const M_t& E,
    std::vector<double>* residuals) {
  ComputeSquaredSampsonErrorWithCheirality(cam_rays1, cam_rays2, E, residuals);
}

void EssentialMatrixLMEstimator::Estimate(const std::vector<X_t>& cam_rays1,
                                          const std::vector<Y_t>& cam_rays2,
                                          std::vector<M_t>* models) {
  THROW_CHECK_EQ(cam_rays1.size(), cam_rays2.size());
  THROW_CHECK_GE(cam_rays1.size(),
                 EssentialMatrixEightPointEstimator::kMinNumSamples);
  THROW_CHECK_NOTNULL(models)->clear();

  // Self-seed with the eight-point solver.
  std::vector<M_t> init_models;
  EssentialMatrixEightPointEstimator::Estimate(
      cam_rays1, cam_rays2, &init_models);
  if (init_models.empty()) {
    return;
  }

  // Refine the seed in place. On a degenerate decomposition Refine leaves the
  // model unchanged, so the eight-point seed is returned either way.
  M_t E = init_models[0];
  Refine(cam_rays1, cam_rays2, &E);
  models->push_back(E);
}

bool EssentialMatrixLMEstimator::Refine(const std::vector<X_t>& cam_rays1,
                                        const std::vector<Y_t>& cam_rays2,
                                        M_t* E) {
  THROW_CHECK_EQ(cam_rays1.size(), cam_rays2.size());
  THROW_CHECK_GE(cam_rays1.size(), kMinNumSamples);
  THROW_CHECK_NOTNULL(E);

  // Decompose the initial essential matrix into a relative pose (resolving the
  // four-fold ambiguity via cheirality over the given rays).
  Rigid3d cam2_from_cam1;
  std::vector<int> valid_indices;
  PoseFromEssentialMatrix(
      *E, cam_rays1, cam_rays2, &cam2_from_cam1, &valid_indices);
  if (valid_indices.empty()) {
    // Degenerate configuration: leave the initial model unchanged.
    return false;
  }

  // Nonlinear Sampson refinement of the full 7-parameter pose via
  // ceres::TinySolver (fixed-size, allocation-free, autodiff), applying the
  // relative pose manifold (rotation on SO(3), translation on the unit sphere).
  // Plain least squares: the rays are assumed to be the inlier set, so
  // robustness comes from the RANSAC inlier selection.
  TinySampsonErrorCostFunctor functor(cam_rays1, cam_rays2);
  TinySampsonErrorCostFunctor::AutoDiffFunction f(functor);
  using Solver = TinySolver<decltype(f), RelativePoseManifold>;
  Solver solver;
  Solver::Options options;
  options.max_num_iterations = 25;

  Eigen::Matrix<double, 7, 1> x;
  x.head<4>() = cam2_from_cam1.rotation().normalized().coeffs();
  x.tail<3>() = cam2_from_cam1.translation().normalized();
  solver.Solve(f, &x, options);

  // Keep the refined pose only if the solve stayed finite; otherwise fall back
  // to the decomposed pose.
  if (x.allFinite()) {
    cam2_from_cam1 =
        Rigid3d(Eigen::Quaterniond(x.data()).normalized(), x.tail<3>());
  }
  *E = EssentialMatrixFromPose(cam2_from_cam1);
  return true;
}

void EssentialMatrixLMEstimator::Residuals(const std::vector<X_t>& cam_rays1,
                                           const std::vector<Y_t>& cam_rays2,
                                           const M_t& E,
                                           std::vector<double>* residuals) {
  ComputeSquaredSampsonErrorWithCheirality(cam_rays1, cam_rays2, E, residuals);
}

}  // namespace colmap

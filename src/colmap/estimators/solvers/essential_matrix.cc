// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/solvers/essential_matrix.h"

#include "colmap/estimators/cost_functions/tiny_manifold.h"
#include "colmap/estimators/cost_functions/tiny_sampson_error.h"
#include "colmap/estimators/solvers/utils.h"
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

  // PoseLib's 5-point solver only supports the minimal case. The non-minimal
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

  models->resize(1);
  (*models)[0] = SolveEpipolarConstraintMatrix(A);
}

void EssentialMatrixTangentSampsonEstimator::Estimate(
    const std::vector<X_t>& cam_rays1_with_jac,
    const std::vector<Y_t>& cam_rays2_with_jac,
    std::vector<M_t>* models) {
  EssentialMatrixFivePointEstimator::Estimate(
      RaysFromCamRaysWithJac(cam_rays1_with_jac),
      RaysFromCamRaysWithJac(cam_rays2_with_jac),
      models);
}

bool EssentialMatrixTangentSampsonEstimator::Refine(
    const std::vector<X_t>& cam_rays1_with_jac,
    const std::vector<Y_t>& cam_rays2_with_jac,
    M_t* E) {
  THROW_CHECK_EQ(cam_rays1_with_jac.size(), cam_rays2_with_jac.size());
  THROW_CHECK_GE(cam_rays1_with_jac.size(), kMinNumSamples);
  THROW_CHECK_NOTNULL(E);

  // Decompose the initial E into a relative pose (resolving the four-fold
  // ambiguity via cheirality over the bearings).
  Rigid3d cam2_from_cam1;
  std::vector<int> valid_indices;
  PoseFromEssentialMatrix(*E,
                          RaysFromCamRaysWithJac(cam_rays1_with_jac),
                          RaysFromCamRaysWithJac(cam_rays2_with_jac),
                          &cam2_from_cam1,
                          &valid_indices);
  if (valid_indices.empty()) {
    return false;
  }

  // Nonlinear pixel-space tangent Sampson refinement of the full 7-parameter
  // pose via colmap::TinySolver, applying the relative pose manifold. Plain
  // least squares: robustness comes from the RANSAC inlier selection.
  TinyTangentSampsonErrorCostFunctor f(cam_rays1_with_jac, cam_rays2_with_jac);
  using Solver = TinySolver<decltype(f), RelativePoseManifold>;
  Solver solver;
  Solver::Options options;
  options.max_num_iterations = 25;

  RelPoseParams x = RelPoseParamsFromRigid3d(cam2_from_cam1);
  if (solver.Solve(f, &x, options).status == Solver::NUMERICAL_FAILURE) {
    return false;
  }

  cam2_from_cam1 = Rigid3dFromRelPoseParams(x.data());
  *E = EssentialMatrixFromPose(cam2_from_cam1);
  return true;
}

void EssentialMatrixTangentSampsonEstimator::Residuals(
    const std::vector<X_t>& cam_rays1_with_jac,
    const std::vector<Y_t>& cam_rays2_with_jac,
    const M_t& E,
    std::vector<double>* residuals) {
  ComputeSquaredTangentSampsonErrorWithCheirality(
      cam_rays1_with_jac, cam_rays2_with_jac, E, residuals);
}

}  // namespace colmap

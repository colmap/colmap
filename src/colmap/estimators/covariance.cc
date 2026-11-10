// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/covariance.h"

#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/util/hash_containers.h"

#include <random>

#include <ceres/crs_matrix.h>

namespace colmap {
namespace {

// Points whose Hessian has a smaller ratio of extreme eigenvalues are
// eliminated with damping.
constexpr double kMinPointEigenvalueRatio = 1e-12;

bool ComputeSchurComplement(bool estimate_point_covs,
                            bool estimate_pose_covs,
                            bool estimate_other_covs,
                            double damping,
                            int point_num_params,
                            const std::vector<internal::PointParam>& points,
                            const std::vector<internal::PoseParam>& poses,
                            const std::vector<const double*>& others,
                            ceres::Problem& problem,
                            FlatHashMap<point3D_t, Eigen::MatrixXd>& point_covs,
                            Eigen::SparseMatrix<double>& S) {
  VLOG(2) << "Evaluating the Jacobian for Schur elimination";

  ceres::Problem::EvaluateOptions eval_options;
  eval_options.parameter_blocks.reserve(2 * poses.size() + points.size() +
                                        others.size());
  if (estimate_pose_covs || estimate_other_covs) {
    for (const auto& pose : poses) {
      eval_options.parameter_blocks.push_back(
          const_cast<double*>(pose.cam_from_world));
    }
    for (const double* other : others) {
      eval_options.parameter_blocks.push_back(const_cast<double*>(other));
    }
  }
  for (const auto& point : points) {
    eval_options.parameter_blocks.push_back(const_cast<double*>(point.xyz));
  }

  ceres::CRSMatrix J_full_crs;
  if (!problem.Evaluate(eval_options, nullptr, nullptr, nullptr, &J_full_crs)) {
    LOG(WARNING) << "Failed to evaluate Jacobian";
    return false;
  }

  const Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor>> J_full(
      J_full_crs.num_rows,
      J_full_crs.num_cols,
      J_full_crs.values.size(),
      J_full_crs.rows.data(),
      J_full_crs.cols.data(),
      J_full_crs.values.data());

  if (points.empty()) {
    S = J_full.transpose() * J_full;
    return true;
  }

  if (estimate_point_covs) {
    point_covs.reserve(points.size());
  }

  VLOG(2) << "Schur elimination of point parameters (n = " << point_num_params
          << ")";

  // Notice that here "a" refers to pose/other and "p" to point parameters.
  const Eigen::SparseMatrix<double> J_a =
      J_full.block(0, 0, J_full.rows(), J_full.cols() - point_num_params);
  const Eigen::SparseMatrix<double> J_p = J_full.block(
      0, J_full.cols() - point_num_params, J_full.rows(), point_num_params);
  const Eigen::SparseMatrix<double> H_aa = J_a.transpose() * J_a;
  const Eigen::SparseMatrix<double> H_ap = J_a.transpose() * J_p;
  const Eigen::SparseMatrix<double> H_pa = H_ap.transpose();
  Eigen::SparseMatrix<double> H_pp = J_p.transpose() * J_p;
  // In-place computation of H_pp_inv.
  Eigen::SparseMatrix<double>& H_pp_inv = H_pp;
  int point_param_idx = 0;
  for (const internal::PointParam& point : points) {
    const int tangent_size = ParameterBlockTangentSize(problem, point.xyz);
    const Eigen::MatrixXd H_pp_idx = H_pp.block(
        point_param_idx, point_param_idx, tangent_size, tangent_size);
    const Eigen::MatrixXd H_pp_idx_damped_inv =
        (H_pp_idx +
         damping * Eigen::MatrixXd::Identity(tangent_size, tangent_size))
            .inverse();
    if (estimate_pose_covs || estimate_other_covs) {
      // Eliminate well-conditioned points without damping, which would break
      // the gauge invariance of the Schur complement.
      const Eigen::VectorXd H_pp_idx_eigenvalues =
          Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(H_pp_idx,
                                                         Eigen::EigenvaluesOnly)
              .eigenvalues();
      const Eigen::MatrixXd H_pp_idx_inv =
          H_pp_idx_eigenvalues(0) > kMinPointEigenvalueRatio *
                                        H_pp_idx_eigenvalues(tangent_size - 1)
              ? H_pp_idx.inverse()
              : H_pp_idx_damped_inv;
      for (int i = 0; i < tangent_size; ++i) {
        for (int j = 0; j < tangent_size; ++j) {
          H_pp_inv.coeffRef(point_param_idx + i, point_param_idx + j) =
              H_pp_idx_inv(i, j);
        }
      }
    }
    if (estimate_point_covs) {
      // Point covariance conditioned on fixed pose/other parameters.
      point_covs.emplace(point.point3D_id, H_pp_idx_damped_inv);
    }
    point_param_idx += 3;
  }

  if (!estimate_pose_covs && !estimate_other_covs) {
    return true;
  }

  H_pp_inv.makeCompressed();
  S = H_aa - H_ap * H_pp_inv * H_pa;

  return true;
}

bool SchurEliminateOtherParams(double damping,
                               int pose_num_params,
                               int other_num_params,
                               Eigen::SparseMatrix<double>& S) {
  VLOG(2) << "Schur elimination of other parameters (n = " << other_num_params
          << ")";

  // Notice that here "c" refers to pose and "o" to other parameters.
  const Eigen::SparseMatrix<double> S_cc =
      S.block(0, 0, pose_num_params, pose_num_params);
  const Eigen::SparseMatrix<double> S_co =
      S.block(0, pose_num_params, pose_num_params, other_num_params);
  const Eigen::SparseMatrix<double> S_oc = S_co.transpose();
  Eigen::SparseMatrix<double> S_oo = S.block(
      pose_num_params, pose_num_params, other_num_params, other_num_params);
  for (int i = 0; i < other_num_params; ++i) {
    if (S_oo.coeff(i, i) == 0.0) {
      S_oo.coeffRef(i, i) = damping;
    } else {
      S_oo.coeffRef(i, i) += damping;
    }
  }

  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt_S_oo(S_oo);
  if (llt_S_oo.info() != Eigen::Success) {
    LOG(WARNING)
        << "Simplicial LLT for Schur elimination of other parameters failed";
    return false;
  }

  S = S_cc - S_co * llt_S_oo.solve(S_oc);

  return true;
}

// Eigenvalues of the Jacobi-scaled Schur complement (unit diagonal) below this
// threshold are considered zero. Gauge eigenvalues are at round-off level,
// since the points are eliminated without damping.
constexpr double kMaxNullEigenvalue = 1e-12;
// Minimum spectral gap above kMaxNullEigenvalue required for the smallest
// non-null eigenvalue when a gauge null space is present.
constexpr double kMinNullSpectralGap = 1e2;
// Regularization of the Jacobi-scaled Schur complement for the inverse
// iteration and the pseudo-inverse, above the round-off of its null space and
// well below the null space threshold. It biases the covariance by a relative
// O(kScaledRegularization / eigenvalue).
constexpr double kScaledRegularization = 1e-13;
// A similarity transform does not change the other parameters (intrinsics), so
// gauge null vectors of the Jacobi-scaled Schur complement (unit norm) must
// have components below this threshold on them.
constexpr double kMaxGaugeOtherParamsNorm = 1e-6;
// A bundle adjustment problem without any constraints on the gauge has a
// 7-DoF gauge freedom (Sim3).
constexpr int kMaxGaugeDim = 7;

// Returns L_inv such that L_inv^T * L_inv = S^-1, given S = P^T L D L^T P.
Eigen::MatrixXd LInverseFromLDLT(
    const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& ldlt_S) {
  const Eigen::VectorXd D_dense = ldlt_S.vectorD();
  const Eigen::SparseMatrix<double> L_sparse = ldlt_S.matrixL();
  Eigen::MatrixXd L_inv =
      Eigen::MatrixXd::Identity(L_sparse.rows(), L_sparse.cols());
  L_sparse.triangularView<Eigen::Lower>().solveInPlace(L_inv);
  for (int i = 0; i < L_inv.rows(); ++i) {
    L_inv.row(i) *= 1.0 / std::max(std::sqrt(std::max(D_dense(i), 0.)),
                                   std::numeric_limits<double>::min());
  }
  L_inv *= ldlt_S.permutationP();
  return L_inv;
}

// Computes L_inv such that L_inv^T * L_inv = S^-1 for a positive definite S.
bool ComputeLInverseFullRank(const Eigen::SparseMatrix<double>& S,
                             Eigen::MatrixXd& L_inv) {
  VLOG(2) << "Start sparse Cholesky decomposition (n = " << S.rows() << ")";
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_S(S);
  if (ldlt_S.info() != Eigen::Success ||
      (ldlt_S.vectorD().array() <= 0).any()) {
    LOG(WARNING) << "Simplicial LDLT for computing L_inv failed";
    return false;
  }
  VLOG(2) << "Finish sparse Cholesky decomposition.";
  L_inv = LInverseFromLDLT(ldlt_S);
  VLOG(2) << "Finish factorization.";
  return true;
}

// Computes L_inv such that L_inv^T * L_inv is the inverse of S or, if the
// problem leaves the gauge free, its pseudo-inverse, i.e., the covariance in
// the minimum-trace gauge (Förstner & Wrobel 2016, Eq. 4.224 with W = I).
// The null space is detected on the Jacobi-scaled S_s = D^-1/2 S D^-1/2,
// which is independent of the parameter units and the scene scale.
bool ComputeLInverse(const Eigen::SparseMatrix<double>& S,
                     int pose_num_params,
                     Eigen::MatrixXd& L_inv) {
  const int n = S.rows();
  const Eigen::VectorXd S_diag = S.diagonal();
  if ((S_diag.array() <= 0).any()) {
    LOG(WARNING) << "Unable to compute covariance. Some pose/other parameters "
                    "are not constrained by any residual.";
    return false;
  }
  const Eigen::VectorXd D_inv_sqrt = S_diag.cwiseSqrt().cwiseInverse();
  const Eigen::SparseMatrix<double> S_s =
      D_inv_sqrt.asDiagonal() * S * D_inv_sqrt.asDiagonal();
  Eigen::SparseMatrix<double> S_s_damped = S_s;
  for (int i = 0; i < n; ++i) {
    S_s_damped.coeffRef(i, i) += kScaledRegularization;
  }
  const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_S_s_damped(
      S_s_damped);
  if (ldlt_S_s_damped.info() != Eigen::Success ||
      (ldlt_S_s_damped.vectorD().array() <= 0).any()) {
    LOG(WARNING) << "Simplicial LDLT of the damped Schur complement failed";
    return false;
  }

  // Block inverse iteration for the smallest eigenvalues of S_s, with one more
  // vector than the gauge freedom to detect larger null spaces, followed by
  // Rayleigh-Ritz.
  const int block_size = std::min(kMaxGaugeDim + 1, n);
  std::mt19937 prng(0);
  std::normal_distribution<double> normal;
  Eigen::MatrixXd X = Eigen::MatrixXd::NullaryExpr(
      n, block_size, [&]() { return normal(prng); });
  for (int i = 0; i < 3; ++i) {
    X = Eigen::HouseholderQR<Eigen::MatrixXd>(ldlt_S_s_damped.solve(X))
            .householderQ() *
        Eigen::MatrixXd::Identity(n, block_size);
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ritz(X.transpose() *
                                                            (S_s * X));
  const int num_null =
      (ritz.eigenvalues().array() <= kMaxNullEigenvalue).count();
  VLOG(2) << "Smallest eigenvalues of the scaled Schur complement: "
          << ritz.eigenvalues().transpose();
  if (num_null == 0) {
    return ComputeLInverseFullRank(S, L_inv);
  }
  if (num_null > kMaxGaugeDim || num_null == n) {
    LOG(WARNING) << StringPrintf(
        "Unable to compute covariance. The Schur complement on pose/other "
        "parameters has a null space of dimension %d or more, which exceeds "
        "the gauge freedom. This is likely due to the pose/other parameters "
        "being underconstrained or other degeneracies.",
        num_null);
    return false;
  }
  if (ritz.eigenvalues()(num_null) < kMinNullSpectralGap * kMaxNullEigenvalue) {
    LOG(WARNING) << StringPrintf(
        "Unable to compute covariance. The Schur complement on pose/other "
        "parameters has an ambiguous near-zero eigenvalue (%e) that is not "
        "clearly separated from the gauge null space.",
        ritz.eigenvalues()(num_null));
    return false;
  }

  // Eigenvalues are sorted in increasing order.
  Eigen::MatrixXd N_s = X * ritz.eigenvectors().leftCols(num_null);
  if (N_s.bottomRows(n - pose_num_params).norm() > kMaxGaugeOtherParamsNorm) {
    LOG(WARNING) << StringPrintf(
        "Unable to compute covariance. The Schur complement on pose/other "
        "parameters has a null space of dimension %d that involves the other "
        "parameters and is thus not a gauge freedom.",
        num_null);
    return false;
  }
  VLOG(2) << "Gauge freedom of dimension " << num_null;

  // Null space of S, orthonormal in the unscaled tangent coordinates.
  N_s.bottomRows(n - pose_num_params).setZero();
  const Eigen::MatrixXd N =
      Eigen::HouseholderQR<Eigen::MatrixXd>(D_inv_sqrt.asDiagonal() * N_s)
          .householderQ() *
      Eigen::MatrixXd::Identity(n, num_null);

  // P * D^-1/2 * (S_s + eps * I)^-1 * D^-1/2 * P with P = I - N * N^T is the
  // pseudo-inverse of S up to O(eps): the regularized inverse is a
  // generalized inverse of S apart from its component along the null space.
  L_inv = LInverseFromLDLT(ldlt_S_s_damped) * D_inv_sqrt.asDiagonal();
  L_inv -= (L_inv * N) * N.transpose();
  return true;
}

Eigen::MatrixXd ExtractCovFromLInverse(const Eigen::MatrixXd& L_inv,
                                       int row_start,
                                       int col_start,
                                       int row_block_size,
                                       int col_block_size) {
  return L_inv.block(0, row_start, L_inv.rows(), row_block_size).transpose() *
         L_inv.block(0, col_start, L_inv.cols(), col_block_size);
}

}  // namespace

BACovariance::BACovariance(
    FlatHashMap<point3D_t, Eigen::MatrixXd> point_covs,
    NodeHashMap<image_t, std::pair<int, int>> pose_L_start_size,
    NodeHashMap<const double*, std::pair<int, int>> other_L_start_size,
    Eigen::MatrixXd L_inv)
    : point_covs_(std::move(point_covs)),
      pose_L_start_size_(std::move(pose_L_start_size)),
      other_L_start_size_(std::move(other_L_start_size)),
      L_inv_(std::move(L_inv)) {}

std::optional<Eigen::MatrixXd> BACovariance::GetPointCov(
    point3D_t point3D_id) const {
  const auto it = point_covs_.find(point3D_id);
  if (it == point_covs_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<Eigen::MatrixXd> BACovariance::GetCamCovFromWorld(
    image_t image_id) const {
  const auto it = pose_L_start_size_.find(image_id);
  if (it == pose_L_start_size_.end()) {
    return std::nullopt;
  }
  const auto [start, size] = it->second;
  return ExtractCovFromLInverse(L_inv_, start, start, size, size);
}

std::optional<Eigen::MatrixXd> BACovariance::GetCamCrossCovFromWorld(
    image_t image_id1, image_t image_id2) const {
  const auto it1 = pose_L_start_size_.find(image_id1);
  const auto it2 = pose_L_start_size_.find(image_id2);
  if (it1 == pose_L_start_size_.end() || it2 == pose_L_start_size_.end()) {
    return std::nullopt;
  }
  const auto [start1, size1] = it1->second;
  const auto [start2, size2] = it2->second;
  return ExtractCovFromLInverse(L_inv_, start1, start2, size1, size2);
}

std::optional<Eigen::MatrixXd> BACovariance::GetCam2CovFromCam1(
    image_t image_id1,
    const Rigid3d& cam1_from_world,
    image_t image_id2,
    const Rigid3d& cam2_from_world) const {
  auto cov_11 = GetCamCovFromWorld(image_id1);
  if (!cov_11.has_value()) return std::nullopt;
  if (cov_11->rows() != 6) {
    LOG(WARNING) << "cam1_from_world is not fully in the problem. This is "
                    "likely due to the pose being set (partially) constant. ";
    return std::nullopt;
  }
  auto cov_22 = GetCamCovFromWorld(image_id2);
  if (!cov_22.has_value()) return std::nullopt;
  if (cov_22->rows() != 6) {
    LOG(WARNING) << "cam2_from_world is not fully in the problem. This is "
                    "likely due to the pose being set (partially) constant. ";
    return std::nullopt;
  }
  auto cov_12 = GetCamCrossCovFromWorld(image_id1, image_id2);
  THROW_CHECK(cov_12.has_value());
  Eigen::Matrix<double, 12, 12> cov;
  cov.block<6, 6>(0, 0) = *cov_11;
  cov.block<6, 6>(6, 6) = *cov_22;
  cov.block<6, 6>(0, 6) = *cov_12;
  cov.block<6, 6>(6, 0) = cov_12->transpose();
  return GetCovarianceForRelativeRigid3d(cam1_from_world, cam2_from_world, cov);
}

std::optional<Eigen::MatrixXd> BACovariance::GetOtherParamsCov(
    const double* params) const {
  const auto it = other_L_start_size_.find(params);
  if (it == other_L_start_size_.end()) {
    return std::nullopt;
  }
  const auto [start, size] = it->second;
  return ExtractCovFromLInverse(L_inv_, start, start, size, size);
}

std::optional<BACovariance> EstimateBACovariance(
    const BACovarianceOptions& options,
    const Reconstruction& reconstruction,
    CeresBundleAdjuster& bundle_adjuster) {
  ceres::Problem& problem = *THROW_CHECK_NOTNULL(bundle_adjuster.Problem());
  return EstimateBACovarianceFromProblem(options, reconstruction, problem);
}

std::optional<BACovariance> EstimateBACovarianceFromProblem(
    const BACovarianceOptions& options,
    const Reconstruction& reconstruction,
    ceres::Problem& problem) {
  const bool estimate_point_covs =
      options.params == BACovarianceOptions::Params::POINTS ||
      options.params == BACovarianceOptions::Params::POSES_AND_POINTS ||
      options.params == BACovarianceOptions::Params::ALL;
  const bool estimate_pose_covs =
      options.params == BACovarianceOptions::Params::POSES ||
      options.params == BACovarianceOptions::Params::POSES_AND_POINTS ||
      options.params == BACovarianceOptions::Params::ALL;
  const bool estimate_other_covs =
      options.params == BACovarianceOptions::Params::ALL;

  const std::vector<internal::PointParam> points =
      internal::GetPointParams(reconstruction, problem);
  const std::vector<internal::PoseParam>& poses =
      options.experimental_custom_poses.empty()
          ? internal::GetPoseParams(reconstruction, problem)
          : options.experimental_custom_poses;
  const std::vector<const double*> others =
      GetOtherParams(problem, poses, points);

  int point_num_params = 0;
  int pose_num_params = 0;
  int other_num_params = 0;
  NodeHashMap<image_t, std::pair<int, int>> pose_L_start_size;
  NodeHashMap<const double*, std::pair<int, int>> other_L_start_size;
  for (const auto& point : points) {
    point_num_params += ParameterBlockTangentSize(problem, point.xyz);
  }
  if (estimate_pose_covs || estimate_other_covs) {
    pose_L_start_size.reserve(poses.size());
    for (const auto& pose : poses) {
      const int num_params =
          ParameterBlockTangentSize(problem, pose.cam_from_world);
      pose_L_start_size.emplace(pose.image_id,
                                std::make_pair(pose_num_params, num_params));
      pose_num_params += num_params;
    }

    other_L_start_size.reserve(poses.size());
    for (const double* other : others) {
      const int num_params = ParameterBlockTangentSize(problem, other);
      other_L_start_size.emplace(
          other,
          std::make_pair(pose_num_params + other_num_params, num_params));
      other_num_params += num_params;
    }
  }

  FlatHashMap<point3D_t, Eigen::MatrixXd> point_covs;
  Eigen::SparseMatrix<double> S;
  if (!ComputeSchurComplement(estimate_point_covs,
                              estimate_pose_covs,
                              estimate_other_covs,
                              options.damping,
                              point_num_params,
                              points,
                              poses,
                              others,
                              problem,
                              point_covs,
                              S)) {
    return std::nullopt;
  }

  if (!estimate_pose_covs && !estimate_other_covs) {
    return BACovariance(std::move(point_covs),
                        /*pose_L_start_size=*/{},
                        /*other_L_start_size=*/{},
                        /*L_inv=*/Eigen::MatrixXd());
  }

  if (!estimate_other_covs) {
    if (!SchurEliminateOtherParams(
            options.damping, pose_num_params, other_num_params, S)) {
      return std::nullopt;
    }
  }

  VLOG(2) << "Computing L inverse";

  Eigen::MatrixXd L_inv;
  if (!ComputeLInverse(S, pose_num_params, L_inv)) {
    return std::nullopt;
  }

  return BACovariance(std::move(point_covs),
                      std::move(pose_L_start_size),
                      std::move(other_L_start_size),
                      std::move(L_inv));
}

namespace internal {

std::vector<PoseParam> GetPoseParams(const Reconstruction& reconstruction,
                                     const ceres::Problem& problem) {
  std::vector<PoseParam> params;
  params.reserve(reconstruction.NumImages());
  for (const auto& [image_id, image] : reconstruction.Images()) {
    // TODO(jsch): Add support for non-trivial frames.
    THROW_CHECK(image.IsRefInFrame());
    const Rigid3d& cam_from_world = image.FramePtr()->RigFromWorld();
    if (problem.HasParameterBlock(cam_from_world.params.data()) &&
        !problem.IsParameterBlockConstant(
            const_cast<double*>(cam_from_world.params.data()))) {
      params.push_back({image_id, cam_from_world.params.data()});
    }
  }
  return params;
}

std::vector<PointParam> GetPointParams(const Reconstruction& reconstruction,
                                       const ceres::Problem& problem) {
  std::vector<PointParam> params;
  params.reserve(reconstruction.NumPoints3D());
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    const double* xyz = point3D.xyz.data();
    if (problem.HasParameterBlock(xyz) &&
        !problem.IsParameterBlockConstant(const_cast<double*>(xyz))) {
      params.push_back({point3D_id, xyz});
    }
  }
  return params;
}

std::vector<const double*> GetOtherParams(
    const ceres::Problem& problem,
    const std::vector<PoseParam>& poses,
    const std::vector<PointParam>& points) {
  FlatHashSet<const double*> pose_and_point_params;
  pose_and_point_params.reserve(poses.size() + points.size());
  for (const auto& pose : poses) {
    pose_and_point_params.insert(pose.cam_from_world);
  }
  for (const auto& point : points) {
    pose_and_point_params.insert(point.xyz);
  }

  std::vector<const double*> params;
  std::vector<double*> all_params;
  problem.GetParameterBlocks(&all_params);
  for (const double* param : all_params) {
    if (!problem.IsParameterBlockConstant(const_cast<double*>(param)) &&
        pose_and_point_params.count(param) == 0) {
      params.push_back(param);
    }
  }
  return params;
}

}  // namespace internal
}  // namespace colmap

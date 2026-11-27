// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/covariance.h"

#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/util/hash_containers.h"

#include <algorithm>

#include <ceres/crs_matrix.h>

namespace colmap {
namespace {

// Point Jacobian singular values below this fraction of the largest singular
// value are truncated during Schur elimination so ill-conditioned or
// rank-deficient points preserve the gauge invariance of the Schur complement.
constexpr double kMinPointSingularValueRatio = 1e-7;

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
  const int a_num_params = J_full.cols() - point_num_params;
  const bool compute_schur = a_num_params > 0;
  const Eigen::SparseMatrix<double> J_p =
      J_full.block(0, a_num_params, J_full.rows(), point_num_params);
  // Orthogonal left singular vectors U_p of each point's Jacobian block,
  // preserving the sparsity pattern of J_p. Forming S = J_a^T J_a - W^T W with
  // W = U_p^T J_a avoids squaring cond(J_p) via normal equations.
  std::vector<Eigen::Triplet<double>> U_p_triplets;
  if (compute_schur) {
    U_p_triplets.reserve(J_p.nonZeros());
  }

  std::vector<int> rows;
  int point_param_idx = 0;
  for (const internal::PointParam& point : points) {
    const int tangent_size = ParameterBlockTangentSize(problem, point.xyz);
    rows.clear();
    for (int c = 0; c < tangent_size; ++c) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(J_p,
                                                         point_param_idx + c);
           it;
           ++it) {
        rows.push_back(it.row());
      }
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());

    if (rows.empty()) {
      if (estimate_point_covs) {
        point_covs.emplace(point.point3D_id,
                           (1.0 / damping) * Eigen::MatrixXd::Identity(
                                                 tangent_size, tangent_size));
      }
      point_param_idx += tangent_size;
      continue;
    }

    Eigen::MatrixXd J_pi = Eigen::MatrixXd::Zero(rows.size(), tangent_size);
    for (int c = 0; c < tangent_size; ++c) {
      size_t row_pos = 0;
      for (Eigen::SparseMatrix<double>::InnerIterator it(J_p,
                                                         point_param_idx + c);
           it;
           ++it) {
        while (rows[row_pos] < it.row()) {
          ++row_pos;
        }
        J_pi(row_pos, c) = it.value();
      }
    }

    const unsigned int svd_options =
        (compute_schur ? Eigen::ComputeThinU : 0) |
        (estimate_point_covs ? Eigen::ComputeFullV : 0);
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(J_pi, svd_options);
    const Eigen::VectorXd& singular_values = svd.singularValues();

    if (compute_schur && singular_values(0) > 0) {
      const Eigen::MatrixXd& U = svd.matrixU();
      const double min_singular_value =
          kMinPointSingularValueRatio * singular_values(0);
      for (int r = 0; r < singular_values.size(); ++r) {
        if (singular_values(r) <= min_singular_value) {
          break;
        }
        for (size_t j = 0; j < rows.size(); ++j) {
          U_p_triplets.emplace_back(rows[j], point_param_idx + r, U(j, r));
        }
      }
    }

    if (estimate_point_covs) {
      // Point covariance conditioned on fixed pose/other parameters.
      const Eigen::MatrixXd& V = svd.matrixV();
      Eigen::VectorXd inv_eigs(tangent_size);
      for (int r = 0; r < tangent_size; ++r) {
        const double s = r < singular_values.size() ? singular_values(r) : 0.0;
        inv_eigs(r) = 1.0 / (s * s + damping);
      }
      point_covs.emplace(point.point3D_id,
                         V * inv_eigs.asDiagonal() * V.transpose());
    }
    point_param_idx += tangent_size;
  }

  if (!compute_schur) {
    return true;
  }

  Eigen::SparseMatrix<double> U_p(J_full.rows(), point_num_params);
  U_p.setFromTriplets(U_p_triplets.begin(), U_p_triplets.end());
  const Eigen::SparseMatrix<double> J_a =
      J_full.block(0, 0, J_full.rows(), a_num_params);
  const Eigen::SparseMatrix<double> W = U_p.transpose() * J_a;
  S = J_a.transpose() * J_a - W.transpose() * W;

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

bool ComputeLInverse(Eigen::SparseMatrix<double>& S, Eigen::MatrixXd& L_inv) {
  VLOG(2) << "Start sparse Cholesky decomposition (n = " << S.rows() << ")";
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_S(S);
  if (ldlt_S.info() != Eigen::Success) {
    LOG(WARNING) << "Simplicial LDLT for computing L_inv failed";
    return false;
  }

  const Eigen::VectorXd D_dense = ldlt_S.vectorD();

  const int rank = (D_dense.array().abs() > 1e-6).count();
  if (rank < S.rows()) {
    LOG(WARNING) << StringPrintf(
        "Unable to compute covariance. The Schur complement on pose/other "
        "parameters is rank deficient. Number of columns: %d, rank: %d. This "
        "is likely due to the pose/other parameters being underconstrained "
        "with Gauge ambiguity or other degeneracies.",
        S.rows(),
        rank);
    return false;
  }
  VLOG(2) << "Finish sparse Cholesky decomposition.";

  const Eigen::SparseMatrix<double> L_sparse = ldlt_S.matrixL();
  L_inv = Eigen::MatrixXd::Identity(L_sparse.rows(), L_sparse.cols());
  L_sparse.triangularView<Eigen::Lower>().solveInPlace(L_inv);
  for (int i = 0; i < S.rows(); ++i) {
    L_inv.row(i) *= 1.0 / std::max(std::sqrt(std::max(D_dense(i), 0.)),
                                   std::numeric_limits<double>::min());
  }
  L_inv *= ldlt_S.permutationP();
  VLOG(2) << "Finish factorization.";
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

  const bool compute_pose_or_other_covs =
      (estimate_pose_covs && pose_num_params > 0) ||
      (estimate_other_covs && other_num_params > 0);
  if (!compute_pose_or_other_covs) {
    return BACovariance(std::move(point_covs),
                        /*pose_L_start_size=*/{},
                        /*other_L_start_size=*/{},
                        /*L_inv=*/Eigen::MatrixXd());
  }

  if (!estimate_other_covs && other_num_params > 0) {
    if (!SchurEliminateOtherParams(
            options.damping, pose_num_params, other_num_params, S)) {
      return std::nullopt;
    }
  }

  VLOG(2) << "Computing L inverse";

  Eigen::MatrixXd L_inv;
  if (!ComputeLInverse(S, L_inv)) {
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

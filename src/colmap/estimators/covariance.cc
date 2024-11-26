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

#include "colmap/estimators/covariance.h"

#include "colmap/estimators/manifold.h"

#include <unordered_set>

#include <ceres/crs_matrix.h>

namespace colmap {
namespace {

bool ComputeSchurComplement(
    bool estimate_point_covs,
    bool estimate_pose_covs,
    bool estimate_other_covs,
    double damping,
    int point_num_params,
    const std::vector<internal::PointParam>& points,
    const std::vector<internal::PoseParam>& poses,
    const std::vector<const double*>& others,
    ceres::Problem& problem,
    std::unordered_map<point3D_t, Eigen::MatrixXd>& point_covs,
    Eigen::SparseMatrix<double>& S) {
  VLOG(2) << "Evaluating the Jacobian for Schur elimination";

  ceres::Problem::EvaluateOptions eval_options;
  eval_options.parameter_blocks.reserve(2 * poses.size() + points.size() +
                                        others.size());
  if (estimate_pose_covs || estimate_other_covs) {
    for (const auto& pose : poses) {
      if (pose.qvec != nullptr) {
        eval_options.parameter_blocks.push_back(const_cast<double*>(pose.qvec));
      }
      if (pose.tvec != nullptr) {
        eval_options.parameter_blocks.push_back(const_cast<double*>(pose.tvec));
      }
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

  VLOG(2) << "Schur elimination of point parameters";

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
    const Eigen::MatrixXd H_pp_idx =
        H_pp.block(
            point_param_idx, point_param_idx, tangent_size, tangent_size) +
        damping * Eigen::MatrixXd::Identity(tangent_size, tangent_size);
    const Eigen::MatrixXd H_pp_idx_inv = H_pp_idx.inverse();
    for (int i = 0; i < tangent_size; ++i) {
      for (int j = 0; j < tangent_size; ++j) {
        H_pp_inv.coeffRef(point_param_idx + i, point_param_idx + j) =
            H_pp_idx_inv(i, j);
      }
    }
    if (estimate_point_covs) {
      // Point covariance conditioned on fixed pose/other parameters.
      point_covs.emplace(point.point3D_id, H_pp_idx_inv);
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
  VLOG(2) << "Schur elimination of other parameters";

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
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_S(S);
  if (ldlt_S.info() != Eigen::Success) {
    LOG(WARNING) << "Simplicial LDLT for computing L_inv failed";
    return false;
  }

  const Eigen::VectorXd D_dense = ldlt_S.vectorD();

  const int rank = D_dense.nonZeros();
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

  const Eigen::SparseMatrix<double> L_sparse = ldlt_S.matrixL();
  L_inv = Eigen::MatrixXd::Identity(L_sparse.rows(), L_sparse.cols());
  L_sparse.triangularView<Eigen::Lower>().solveInPlace(L_inv);
  for (int i = 0; i < S.rows(); ++i) {
    L_inv.row(i) *= 1.0 / std::max(std::sqrt(std::max(D_dense(i), 0.)),
                                   std::numeric_limits<double>::min());
  }
  L_inv *= ldlt_S.permutationP();
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
    std::unordered_map<point3D_t, Eigen::MatrixXd> point_covs,
    std::unordered_map<image_t, std::pair<int, int>> pose_L_start_size,
    std::unordered_map<const double*, std::pair<int, int>> other_L_start_size,
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

std::optional<Eigen::MatrixXd> BACovariance::GetCamFromWorldCov(
    image_t image_id) const {
  const auto it = pose_L_start_size_.find(image_id);
  if (it == pose_L_start_size_.end()) {
    return std::nullopt;
  }
  const auto [start, size] = it->second;
  return ExtractCovFromLInverse(L_inv_, start, start, size, size);
}

std::optional<Eigen::MatrixXd> BACovariance::GetCam1FromCam2Cov(
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
    BundleAdjuster& bundle_adjuster) {
  ceres::Problem& problem = *THROW_CHECK_NOTNULL(bundle_adjuster.Problem());
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
  std::unordered_map<image_t, std::pair<int, int>> pose_L_start_size;
  std::unordered_map<const double*, std::pair<int, int>> other_L_start_size;
  for (const auto& point : points) {
    point_num_params += ParameterBlockTangentSize(problem, point.xyz);
  }
  if (estimate_pose_covs || estimate_other_covs) {
    pose_L_start_size.reserve(poses.size());
    for (const auto& pose : poses) {
      int num_params = 0;
      if (pose.qvec != nullptr) {
        num_params += ParameterBlockTangentSize(problem, pose.qvec);
      }
      if (pose.tvec != nullptr) {
        num_params += ParameterBlockTangentSize(problem, pose.tvec);
      }
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

  std::unordered_map<point3D_t, Eigen::MatrixXd> point_covs;
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
    const double* qvec = image.CamFromWorld().rotation.coeffs().data();
    if (!problem.HasParameterBlock(qvec) ||
        problem.IsParameterBlockConstant(const_cast<double*>(qvec))) {
      qvec = nullptr;
    }

    const double* tvec = image.CamFromWorld().translation.data();
    if (!problem.HasParameterBlock(tvec) ||
        problem.IsParameterBlockConstant(const_cast<double*>(tvec))) {
      tvec = nullptr;
    }

    if (qvec != nullptr || tvec != nullptr) {
      params.push_back({image_id, qvec, tvec});
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
  std::unordered_set<const double*> image_and_point_params;
  for (const auto& pose : poses) {
    image_and_point_params.insert(pose.qvec);
    image_and_point_params.insert(pose.tvec);
  }
  for (const auto& point : points) {
    image_and_point_params.insert(point.xyz);
  }

  std::vector<const double*> params;
  std::vector<double*> all_params;
  problem.GetParameterBlocks(&all_params);
  for (const double* param : all_params) {
    if (!problem.IsParameterBlockConstant(const_cast<double*>(param)) &&
        image_and_point_params.count(param) == 0) {
      params.push_back(param);
    }
  }
  return params;
}

}  // namespace internal
}  // namespace colmap

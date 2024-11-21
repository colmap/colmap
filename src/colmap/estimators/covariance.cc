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

BACovariance EstimateBACovariance(const BACovarianceOptions& options,
                                  const Reconstruction& reconstruction,
                                  BundleAdjuster& bundle_adjuster) {
  ceres::Problem& problem = *THROW_CHECK_NOTNULL(bundle_adjuster.Problem());
  const bool estimate_pose_covs =
      options.params == BACovarianceOptions::Params::kOnlyPoses ||
      options.params == BACovarianceOptions::Params::kPosesAndPoints ||
      options.params == BACovarianceOptions::Params::kAll;
  const bool estimate_point_covs =
      options.params == BACovarianceOptions::Params::kOnlyPoints ||
      options.params == BACovarianceOptions::Params::kPosesAndPoints ||
      options.params == BACovarianceOptions::Params::kAll;

  BACovariance ba_cov;

  const std::vector<detail::PoseParam> poses =
      detail::GetPoseParams(reconstruction, problem);
  const std::vector<detail::PointParam> points =
      detail::GetPointParams(reconstruction, problem);
  const std::vector<const double*> others =
      GetOtherParams(problem, poses, points);

  VLOG(2) << "Evaluating the Jacobian for Schur elimination";

  ceres::Problem::EvaluateOptions eval_options;
  eval_options.parameter_blocks.reserve(2 * poses.size() + points.size() +
                                        others.size());
  int pose_num_params = 0;
  int point_num_params = 0;
  int other_num_params = 0;
  for (const auto& pose : poses) {
    if (pose.qvec != nullptr) {
      eval_options.parameter_blocks.push_back(const_cast<double*>(pose.qvec));
      pose_num_params += ParameterBlockTangentSize(problem, pose.qvec);
    }
    if (pose.tvec != nullptr) {
      eval_options.parameter_blocks.push_back(const_cast<double*>(pose.tvec));
      pose_num_params += ParameterBlockTangentSize(problem, pose.tvec);
    }
  }
  for (const double* other : others) {
    eval_options.parameter_blocks.push_back(const_cast<double*>(other));
    other_num_params += ParameterBlockTangentSize(problem, other);
  }
  for (const auto& point : points) {
    eval_options.parameter_blocks.push_back(const_cast<double*>(point.xyz));
    point_num_params += ParameterBlockTangentSize(problem, point.xyz);
  }

  ceres::CRSMatrix J_full_crs;
  if (!problem.Evaluate(eval_options, nullptr, nullptr, nullptr, &J_full_crs)) {
    LOG(WARNING) << "Failed to evaluate Jacobian";
    ba_cov.success = false;
    return ba_cov;
  }

  const int num_residuals = J_full_crs.num_rows;
  const int num_params = J_full_crs.num_cols;
  const Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor>> J_full(
      J_full_crs.num_rows,
      J_full_crs.num_cols,
      J_full_crs.values.size(),
      J_full_crs.rows.data(),
      J_full_crs.cols.data(),
      J_full_crs.values.data());

  VLOG(2) << "Schur elimination on point parameters";

  // Notice that here "a" refers to the block of pose + other parameters.
  const Eigen::SparseMatrix<double> J_a =
      J_full.block(0, 0, num_residuals, num_params - point_num_params);
  const Eigen::SparseMatrix<double> J_p = J_full.block(
      0, num_params - point_num_params, num_residuals, point_num_params);
  const Eigen::SparseMatrix<double> H_aa = J_a.transpose() * J_a;
  const Eigen::SparseMatrix<double> H_ap = J_a.transpose() * J_p;
  const Eigen::SparseMatrix<double> H_pa = H_ap.transpose();
  Eigen::SparseMatrix<double> H_pp = J_p.transpose() * J_p;
  // In-place computation of H_pp_inv.
  Eigen::SparseMatrix<double>& H_pp_inv = H_pp;
  int point_param_idx = 0;
  for (const auto& point : points) {
    const Eigen::Matrix3d H_pp_idx =
        H_pp.block(point_param_idx, point_param_idx, 3, 3) +
        options.damping * Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d H_pp_idx_inv = H_pp_idx.inverse();
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        H_pp_inv.coeffRef(point_param_idx + i, point_param_idx + j) =
            H_pp_idx_inv(i, j);
      }
    }
    if (estimate_point_covs) {
      // Point covariance conditioned on fixed other variables.
      ba_cov.point_covs[point.point3D_id] = H_pp_idx_inv;
    }
    point_param_idx += 3;
  }

  if (!estimate_pose_covs) {
    ba_cov.success = true;
    return ba_cov;
  }

  H_pp_inv.makeCompressed();
  const Eigen::SparseMatrix<double> S_a = H_aa - H_ap * H_pp_inv * H_pa;

  VLOG(2) << "Schur elimination on other parameters";

  const Eigen::SparseMatrix<double> S_cc =
      S_a.block(0, 0, pose_num_params, pose_num_params);
  const Eigen::SparseMatrix<double> S_co =
      S_a.block(0, pose_num_params, pose_num_params, other_num_params);
  const Eigen::SparseMatrix<double> S_oc = S_co.transpose();
  Eigen::SparseMatrix<double> S_oo = S_a.block(
      pose_num_params, pose_num_params, other_num_params, other_num_params);
  for (int i = 0; i < other_num_params; ++i) {
    if (S_oo.coeff(i, i) == 0.0) {
      S_oo.coeffRef(i, i) = options.damping;
    } else {
      S_oo.coeffRef(i, i) += options.damping;
    }
  }
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt_S_oo(S_oo);
  const Eigen::SparseMatrix<double> S_c = S_cc - S_co * llt_S_oo.solve(S_oc);

  LOG(INFO) << "Schur elimination on pose parameters";

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_S_c(S_c);
  int rank = 0;
  for (int i = 0; i < S_c.rows(); ++i) {
    if (ldlt_S_c.vectorD().coeff(i) != 0.0) {
      rank++;
    }
  }
  if (rank < S_c.rows()) {
    LOG(WARNING) << StringPrintf(
        "Unable to compute bundle adjustment covariance. The Schur complement "
        "on pose parameters is rank deficient. Number of columns: %d, rank: "
        "%d. This is likely due to the poses being underconstrained with Gauge "
        "ambiguity.",
        S_c.rows(),
        rank);
    ba_cov.success = false;
    return ba_cov;
  }

  Eigen::SparseMatrix<double> I(S_c.rows(), S_c.cols());
  I.setIdentity();
  const Eigen::SparseMatrix<double> S_c_inv = ldlt_S_c.solve(I);

  int pose_param_idx = 0;
  for (const auto& pose : poses) {
    const int pose_tangent_size =
        (pose.qvec == nullptr ? 0
                              : ParameterBlockTangentSize(problem, pose.qvec)) +
        (pose.tvec == nullptr ? 0
                              : ParameterBlockTangentSize(problem, pose.tvec));
    ba_cov.pose_covs[pose.image_id] = S_c_inv.block(
        pose_param_idx, pose_param_idx, pose_tangent_size, pose_tangent_size);
    pose_param_idx += pose_tangent_size;
  }

  ba_cov.success = true;
  return ba_cov;
}

namespace detail {

std::vector<PoseParam> GetPoseParams(const Reconstruction& reconstruction,
                                     const ceres::Problem& problem) {
  std::vector<PoseParam> params;
  params.reserve(reconstruction.NumImages());
  for (const auto& image : reconstruction.Images()) {
    const double* qvec = image.second.CamFromWorld().rotation.coeffs().data();
    if (!problem.HasParameterBlock(qvec) ||
        problem.IsParameterBlockConstant(qvec)) {
      qvec = nullptr;
    }

    const double* tvec = image.second.CamFromWorld().translation.data();
    if (!problem.HasParameterBlock(tvec) ||
        problem.IsParameterBlockConstant(tvec)) {
      tvec = nullptr;
    }

    if (qvec != nullptr || tvec != nullptr) {
      params.push_back({image.first, qvec, tvec});
    }
  }
  return params;
}

std::vector<PointParam> GetPointParams(const Reconstruction& reconstruction,
                                       const ceres::Problem& problem) {
  std::vector<PointParam> params;
  params.reserve(reconstruction.NumPoints3D());
  for (const auto& point3D : reconstruction.Points3D()) {
    const double* xyz = point3D.second.xyz.data();
    if (problem.HasParameterBlock(xyz) &&
        !problem.IsParameterBlockConstant(xyz)) {
      params.push_back({point3D.first, xyz});
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
    if (!problem.IsParameterBlockConstant(param) &&
        image_and_point_params.count(param) == 0) {
      params.push_back(param);
    }
  }
  return params;
}

}  // namespace detail
}  // namespace colmap

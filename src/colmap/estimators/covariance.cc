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

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ceres/crs_matrix.h>

namespace colmap {

std::unordered_map<image_t, Eigen::MatrixXd> EstimatePoseCovariances(
    const Reconstruction& reconstruction,
    ceres::Problem& problem,
    double damping) {
  const std::vector<detail::PoseParam> poses =
      detail::GetPoseParams(reconstruction, problem);

  VLOG(2) << "Evaluating the Jacobian";

  ceres::Problem::EvaluateOptions eval_options;
  eval_options.parameter_blocks.reserve(2 * poses.size());
  for (const auto& pose : poses) {
    if (pose.qvec != nullptr) {
      eval_options.parameter_blocks.push_back(const_cast<double*>(pose.qvec));
    }
    if (pose.tvec != nullptr) {
      eval_options.parameter_blocks.push_back(const_cast<double*>(pose.tvec));
    }
  }

  ceres::CRSMatrix J_crs;
  if (!problem.Evaluate(eval_options, nullptr, nullptr, nullptr, &J_crs)) {
    LOG(WARNING) << "Failed to evaluate Jacobian";
    return {};
  }

  const Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor>> J(
      J_crs.num_rows,
      J_crs.num_cols,
      J_crs.values.size(),
      J_crs.rows.data(),
      J_crs.cols.data(),
      J_crs.values.data());

  VLOG(2) << "Computing pose covariances";

  std::unordered_map<image_t, Eigen::MatrixXd> covs;
  covs.reserve(poses.size());

  const Eigen::SparseMatrix<double> H = J.transpose() * J;
  int pose_param_idx = 0;
  for (const auto& pose : poses) {
    const int pose_tangent_size =
        (pose.qvec == nullptr ? 0
                              : ParameterBlockTangentSize(problem, pose.qvec)) +
        (pose.tvec == nullptr ? 0
                              : ParameterBlockTangentSize(problem, pose.tvec));
    const Eigen::MatrixXd H_identity =
        Eigen::MatrixXd::Identity(pose_tangent_size, pose_tangent_size);
    const Eigen::MatrixXd H_idx = H.block(pose_param_idx,
                                          pose_param_idx,
                                          pose_tangent_size,
                                          pose_tangent_size) +
                                  damping * H_identity;
    Eigen::MatrixXd H_idx_inverse = H_idx.inverse();
    const bool invertible = (H_idx_inverse * H_idx).isApprox(H_identity);
    if (invertible) {
      covs[pose.image_id] = std::move(H_idx_inverse);
    } else {
      VLOG(2) << "Failed to compute covariance for degenerate image: "
              << pose.image_id;
    }
    pose_param_idx += pose_tangent_size;
  }

  return covs;
}

std::unordered_map<point3D_t, Eigen::Matrix3d> EstimatePointCovariances(
    const Reconstruction& reconstruction,
    ceres::Problem& problem,
    double damping) {
  const std::vector<detail::PointParam> points =
      detail::GetPointParams(reconstruction, problem);

  VLOG(2) << "Evaluating the Jacobian";

  ceres::Problem::EvaluateOptions eval_options;
  eval_options.parameter_blocks.reserve(points.size());
  for (const auto& point : points) {
    eval_options.parameter_blocks.push_back(const_cast<double*>(point.xyz));
  }

  ceres::CRSMatrix J_crs;
  if (!problem.Evaluate(eval_options, nullptr, nullptr, nullptr, &J_crs)) {
    LOG(WARNING) << "Failed to evaluate Jacobian";
    return {};
  }

  const Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor>> J(
      J_crs.num_rows,
      J_crs.num_cols,
      J_crs.values.size(),
      J_crs.rows.data(),
      J_crs.cols.data(),
      J_crs.values.data());

  VLOG(2) << "Computing point covariances";

  std::unordered_map<point3D_t, Eigen::Matrix3d> covs;
  covs.reserve(points.size());

  const Eigen::SparseMatrix<double> H = J.transpose() * J;
  int point_param_idx = 0;
  for (const auto& point : points) {
    const Eigen::Matrix3d H_idx =
        H.block(point_param_idx, point_param_idx, 3, 3) +
        damping * Eigen::Matrix3d::Identity();
    Eigen::Matrix3d H_idx_inverse;
    bool invertible = false;
    H_idx.computeInverseWithCheck(H_idx_inverse, invertible);
    if (invertible) {
      covs[point.point3D_id] = H_idx_inverse;
    } else {
      VLOG(2) << "Failed to compute covariance for degenerate point: "
              << point.point3D_id;
    }
    point_param_idx += 3;
  }

  return covs;
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

}  // namespace detail
}  // namespace colmap

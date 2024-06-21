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

#include "colmap/estimators/cost_functions.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ceres/crs_matrix.h>

namespace colmap {

Eigen::Matrix6d GetCovarianceForPoseInverse(const Eigen::Matrix6d& covar,
                                            const Rigid3d& rigid3) {
  Eigen::Matrix6d adjoint = rigid3.Adjoint();
  return adjoint * covar * adjoint.transpose();
}

std::map<image_t, Eigen::MatrixXd>
BundleAdjustmentCovarianceEstimator::EstimatePoseCovarianceCeresBackend(
    ceres::Problem* problem, Reconstruction* reconstruction) {
  THROW_CHECK_NOTNULL(problem);
  THROW_CHECK_NOTNULL(reconstruction);

  // Compute marginalized covariance
  ceres::Covariance::Options options;
  ceres::Covariance covariance_computer(options);
  std::vector<std::pair<const double*, const double*>> pointer_values;
  for (const auto& image : reconstruction->Images()) {
    const double* qvec = image.second.CamFromWorld().rotation.coeffs().data();
    bool qvec_valid =
        problem->HasParameterBlock(qvec) &&
        !problem->IsParameterBlockConstant(const_cast<double*>(qvec));
    const double* tvec = image.second.CamFromWorld().translation.data();
    bool tvec_valid =
        problem->HasParameterBlock(tvec) &&
        !problem->IsParameterBlockConstant(const_cast<double*>(tvec));
    if (qvec_valid) pointer_values.push_back(std::make_pair(qvec, qvec));
    if (tvec_valid) pointer_values.push_back(std::make_pair(tvec, tvec));
    if (qvec_valid && tvec_valid)
      pointer_values.push_back(std::make_pair(qvec, tvec));
  }
  covariance_computer.Compute(pointer_values, problem);

  // Construct covariance for each pose
  std::map<image_t, Eigen::MatrixXd> covs;
  for (const auto& image : reconstruction->Images()) {
    // compute number of effective parameters
    int num_params_qvec = 0;
    const double* qvec = image.second.CamFromWorld().rotation.coeffs().data();
    if (problem->HasParameterBlock(qvec) &&
        !problem->IsParameterBlockConstant(const_cast<double*>(qvec)))
      num_params_qvec = ParameterBlockTangentSize(problem, qvec);
    int num_params_tvec = 0;
    const double* tvec = image.second.CamFromWorld().translation.data();
    if (problem->HasParameterBlock(tvec) &&
        !problem->IsParameterBlockConstant(const_cast<double*>(tvec)))
      num_params_tvec = ParameterBlockTangentSize(problem, tvec);

    // get covariance
    int num_params = num_params_qvec + num_params_tvec;
    Eigen::MatrixXd covar(num_params, num_params);
    if (num_params_qvec > 0) {
      Eigen::Matrix<double, -1, -1, Eigen::RowMajor> cov_qq(num_params_qvec,
                                                            num_params_qvec);
      covariance_computer.GetCovarianceBlockInTangentSpace(
          qvec, qvec, cov_qq.data());
      covar.block(0, 0, num_params_qvec, num_params_qvec) = cov_qq;
    }
    if (num_params_tvec > 0) {
      Eigen::Matrix<double, -1, -1, Eigen::RowMajor> cov_tt(num_params_tvec,
                                                            num_params_tvec);
      covariance_computer.GetCovarianceBlockInTangentSpace(
          tvec, tvec, cov_tt.data());
      covar.block(
          num_params_qvec, num_params_qvec, num_params_tvec, num_params_tvec) =
          cov_tt;
    }
    if (num_params_qvec > 0 && num_params_tvec > 0) {
      Eigen::Matrix<double, -1, -1, Eigen::RowMajor> cov_qt(num_params_qvec,
                                                            num_params_tvec);
      covariance_computer.GetCovarianceBlockInTangentSpace(
          qvec, tvec, cov_qt.data());
      covar.block(0, num_params_qvec, num_params_qvec, num_params_tvec) =
          cov_qt;
      covar.block(num_params_qvec, 0, num_params_tvec, num_params_qvec) =
          cov_qt.transpose();
    }
    covs.insert(std::make_pair(image.second.ImageId(), covar));
  }
  return covs;
}

std::map<image_t, Eigen::MatrixXd>
BundleAdjustmentCovarianceEstimator::EstimatePoseCovariance(
    ceres::Problem* problem,
    Reconstruction* reconstruction,
    const double lambda) {
  THROW_CHECK_NOTNULL(problem);
  THROW_CHECK_NOTNULL(reconstruction);

  // Step 1: Construct ordering that is easy to index
  std::vector<const double*> parameter_blocks;
  std::vector<const double*> point_blocks;
  std::set<const double*> pose_and_point_parameter_blocks;

  // Left: parameters for poses
  int num_params_poses = 0;
  for (const auto& image : reconstruction->Images()) {
    int num_params_qvec = 0;
    const double* qvec = image.second.CamFromWorld().rotation.coeffs().data();
    if (problem->HasParameterBlock(qvec) &&
        !problem->IsParameterBlockConstant(const_cast<double*>(qvec)))
      num_params_qvec = ParameterBlockTangentSize(problem, qvec);
    if (num_params_qvec > 0) {
      parameter_blocks.push_back(qvec);
      pose_and_point_parameter_blocks.insert(qvec);
      num_params_poses += num_params_qvec;
    }

    int num_params_tvec = 0;
    const double* tvec = image.second.CamFromWorld().translation.data();
    if (problem->HasParameterBlock(tvec) &&
        !problem->IsParameterBlockConstant(const_cast<double*>(tvec)))
      num_params_tvec = ParameterBlockTangentSize(problem, tvec);
    if (num_params_tvec > 0) {
      parameter_blocks.push_back(tvec);
      pose_and_point_parameter_blocks.insert(tvec);
      num_params_poses += num_params_tvec;
    }
  }

  // Right: parameters for 3D points that we want to eliminate
  int num_params_points = 0;
  for (const auto& point3D : reconstruction->Points3D()) {
    const double* point3D_ptr = point3D.second.xyz.data();
    int num_params_point = 0;
    if (problem->HasParameterBlock(point3D_ptr) &&
        !problem->IsParameterBlockConstant(const_cast<double*>(point3D_ptr))) {
      num_params_point = ParameterBlockTangentSize(problem, point3D_ptr);
    }
    if (num_params_point > 0) {
      point_blocks.push_back(point3D_ptr);
      pose_and_point_parameter_blocks.insert(point3D_ptr);
      num_params_points += num_params_point;
    }
  }

  // Middle: parameters other than poses and 3D points
  std::vector<double*> all_parameter_blocks;
  problem->GetParameterBlocks(&all_parameter_blocks);
  for (const auto& block : all_parameter_blocks) {
    if (problem->IsParameterBlockConstant(block)) continue;
    // check if the current parameter block is in either the pose or point
    // parameter blocks
    if (pose_and_point_parameter_blocks.find(block) ==
        pose_and_point_parameter_blocks.end()) {
      parameter_blocks.push_back(block);
    }
  }
  parameter_blocks.insert(
      parameter_blocks.end(), point_blocks.begin(), point_blocks.end());

  // Step 2: Evaluate Jacobian
  LOG(INFO) << "Evaluate jacobians";
  ceres::Problem::EvaluateOptions eval_options;
  eval_options.parameter_blocks.clear();
  for (const auto& block : parameter_blocks) {
    eval_options.parameter_blocks.push_back(const_cast<double*>(block));
  }
  ceres::CRSMatrix J_full_crs;
  problem->Evaluate(eval_options, nullptr, nullptr, nullptr, &J_full_crs);
  int num_residuals = J_full_crs.num_rows;
  int num_params = J_full_crs.num_cols;
  Eigen::Map<Eigen::SparseMatrix<double, Eigen::RowMajor>> J_full(
      J_full_crs.num_rows,
      J_full_crs.num_cols,
      J_full_crs.values.size(),
      J_full_crs.rows.data(),
      J_full_crs.cols.data(),
      J_full_crs.values.data());

  // Step 3: Schur elimination on points
  LOG(INFO) << "Schur elimination on points";
  Eigen::SparseMatrix<double> J_c =
      J_full.block(0, 0, num_residuals, num_params - num_params_points);
  Eigen::SparseMatrix<double> J_p = J_full.block(
      0, num_params - num_params_points, num_residuals, num_params_points);
  Eigen::SparseMatrix<double> H_cc = J_c.transpose() * J_c;
  Eigen::SparseMatrix<double> H_cp = J_c.transpose() * J_p;
  Eigen::SparseMatrix<double> H_pc = H_cp.transpose();
  Eigen::SparseMatrix<double> H_pp = J_p.transpose() * J_p;
  // in-place computation of H_pp_inv
  Eigen::SparseMatrix<double> H_pp_inv = H_pp;  // actually a shallow copy
  int counter_p = 0;
  for (const auto& point3D : reconstruction->Points3D()) {
    const double* point3D_ptr = point3D.second.xyz.data();
    int num_params_point = 0;
    if (problem->HasParameterBlock(point3D_ptr) &&
        !problem->IsParameterBlockConstant(const_cast<double*>(point3D_ptr))) {
      num_params_point = ParameterBlockTangentSize(problem, point3D_ptr);
    }
    if (num_params_point == 0) continue;
    Eigen::SparseMatrix<double> subMatrix_sparse =
        H_pp.block(counter_p, counter_p, num_params_point, num_params_point);
    Eigen::MatrixXd subMatrix = subMatrix_sparse;
    subMatrix +=
        lambda * Eigen::MatrixXd::Identity(subMatrix.rows(), subMatrix.cols());
    Eigen::MatrixXd subMatrix_inv = subMatrix.inverse();
    // update matrix
    for (int i = 0; i < num_params_point; ++i) {
      for (int j = 0; j < num_params_point; ++j) {
        H_pp_inv.coeffRef(counter_p + i, counter_p + j) = subMatrix_inv(i, j);
      }
    }
    H_pp_inv.makeCompressed();
    counter_p += num_params_point;
  }
  Eigen::SparseMatrix<double> S = H_cc - H_cp * H_pp_inv * H_pc;

  // Step 4: Schur elimination on other variables to get pose covariance
  Eigen::MatrixXd S_dense = S;
  int num_params_variables = S_dense.rows() - num_params_poses;
  LOG(INFO) << StringPrintf("Schur elimination on other variables (n = %d)",
                            num_params_variables);
  Eigen::MatrixXd S_aa =
      S_dense.block(0, 0, num_params_poses, num_params_poses);
  Eigen::MatrixXd S_ab = S_dense.block(
      0, num_params_poses, num_params_poses, num_params_variables);
  Eigen::MatrixXd S_ba = S_ab.transpose();
  Eigen::MatrixXd S_bb = S_dense.block(num_params_poses,
                                       num_params_poses,
                                       num_params_variables,
                                       num_params_variables);
  Eigen::LDLT<Eigen::MatrixXd> ldltOfS_bb(S_bb);
  Eigen::MatrixXd S_poses = S_aa - S_ab * ldltOfS_bb.solve(S_ba);

  // Step 5: Compute covariance
  LOG(INFO) << StringPrintf(
      "Inverting Schur complement for pose parameters (n = %d)",
      num_params_poses);
  Eigen::FullPivLU<Eigen::MatrixXd> luOfS_poses(S_poses);
  if (luOfS_poses.rank() < S_poses.rows()) {
    LOG(FATAL_THROW) << StringPrintf(
        "Error! The Schur complement on pose parameters is rank "
        "deficient. Number of columns: %d, rank: %d. This is likely due to "
        "the poses being underconstrained with Gauge ambiguity.",
        S_poses.rows(),
        luOfS_poses.rank());
  }
  Eigen::MatrixXd cov_poses = S_poses.inverse();

  // Step 6: Construct output
  std::map<image_t, Eigen::MatrixXd> covs;
  int counter_c = 0;
  for (const auto& image : reconstruction->Images()) {
    int num_params_qvec = 0;
    const double* qvec = image.second.CamFromWorld().rotation.coeffs().data();
    if (problem->HasParameterBlock(qvec) &&
        !problem->IsParameterBlockConstant(const_cast<double*>(qvec)))
      num_params_qvec = ParameterBlockTangentSize(problem, qvec);
    int num_params_tvec = 0;
    const double* tvec = image.second.CamFromWorld().translation.data();
    if (problem->HasParameterBlock(tvec) &&
        !problem->IsParameterBlockConstant(const_cast<double*>(tvec)))
      num_params_tvec = ParameterBlockTangentSize(problem, tvec);
    int num_params_pose = num_params_qvec + num_params_tvec;
    if (num_params_pose == 0) continue;

    // get covariance
    Eigen::MatrixXd covar =
        cov_poses.block(counter_c, counter_c, num_params_pose, num_params_pose);
    counter_c += num_params_pose;
    covs.insert(std::make_pair(image.second.ImageId(), covar));
  }
  return covs;
}

}  // namespace colmap

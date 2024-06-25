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

#include <ceres/crs_matrix.h>

namespace colmap {

BundleAdjustmentCovarianceEstimatorBase::
    BundleAdjustmentCovarianceEstimatorBase(ceres::Problem* problem,
                                            Reconstruction* reconstruction) {
  THROW_CHECK_NOTNULL(problem);
  THROW_CHECK_NOTNULL(reconstruction);
  problem_ = problem;
  reconstruction_ = reconstruction;
  Setup();
}

void BundleAdjustmentCovarianceEstimatorBase::Setup() {
  // Parse parameter blocks
  std::set<const double*> pose_and_point_parameter_blocks;
  // Parse parameter blocks for poses
  pose_blocks_.clear();
  num_params_poses_ = 0;
  for (const auto& image : reconstruction_->Images()) {
    const double* qvec = image.second.CamFromWorld().rotation.coeffs().data();
    if (problem_->HasParameterBlock(qvec) &&
        !problem_->IsParameterBlockConstant(const_cast<double*>(qvec))) {
      int num_params_qvec = ParameterBlockTangentSize(problem_, qvec);
      pose_blocks_.push_back(qvec);
      map_block_to_index_.emplace(qvec, num_params_poses_);
      num_params_poses_ += num_params_qvec;
      pose_and_point_parameter_blocks.insert(qvec);
    }

    const double* tvec = image.second.CamFromWorld().translation.data();
    if (problem_->HasParameterBlock(tvec) &&
        !problem_->IsParameterBlockConstant(const_cast<double*>(tvec))) {
      int num_params_tvec = ParameterBlockTangentSize(problem_, tvec);
      pose_blocks_.push_back(tvec);
      map_block_to_index_.emplace(tvec, num_params_poses_);
      num_params_poses_ += num_params_tvec;
      pose_and_point_parameter_blocks.insert(tvec);
    }
  }

  // Parse parameter blocks for 3D points
  point_blocks_.clear();
  num_params_points_ = 0;
  for (const auto& point3D : reconstruction_->Points3D()) {
    const double* point3D_ptr = point3D.second.xyz.data();
    if (problem_->HasParameterBlock(point3D_ptr) &&
        !problem_->IsParameterBlockConstant(const_cast<double*>(point3D_ptr))) {
      int num_params_point = ParameterBlockTangentSize(problem_, point3D_ptr);
      point_blocks_.push_back(point3D_ptr);
      num_params_points_ += num_params_point;
      pose_and_point_parameter_blocks.insert(point3D_ptr);
    }
  }

  // Parse parameter blocks for other variables
  other_variables_blocks_.clear();
  num_params_other_variables_ = 0;
  std::vector<double*> all_parameter_blocks;
  problem_->GetParameterBlocks(&all_parameter_blocks);
  for (double* block : all_parameter_blocks) {
    if (problem_->IsParameterBlockConstant(block)) continue;
    // check if the current parameter block is in either the pose or point
    // parameter blocks
    if (pose_and_point_parameter_blocks.find(block) !=
        pose_and_point_parameter_blocks.end()) {
      continue;
    }
    other_variables_blocks_.push_back(block);
    map_block_to_index_.insert(
        std::make_pair(block, num_params_poses_ + num_params_other_variables_));
    int num_params_block = ParameterBlockTangentSize(problem_, block);
    num_params_other_variables_ += num_params_block;
  }
}

bool BundleAdjustmentCovarianceEstimatorBase::HasBlock(
    const double* params) const {
  return map_block_to_index_.find(params) != map_block_to_index_.end();
}

bool BundleAdjustmentCovarianceEstimatorBase::HasPose(image_t image_id) const {
  const double* qvec =
      reconstruction_->Image(image_id).CamFromWorld().rotation.coeffs().data();
  const double* tvec =
      reconstruction_->Image(image_id).CamFromWorld().translation.data();
  return HasBlock(qvec) && HasBlock(tvec);
}

int BundleAdjustmentCovarianceEstimatorBase::GetBlockIndex(
    const double* params) const {
  THROW_CHECK(HasBlock(params));
  return map_block_to_index_.at(params);
}

int BundleAdjustmentCovarianceEstimatorBase::GetBlockTangentSize(
    const double* params) const {
  THROW_CHECK(HasBlock(params));
  return ParameterBlockTangentSize(problem_, params);
}

int BundleAdjustmentCovarianceEstimatorBase::GetPoseIndex(
    image_t image_id) const {
  const double* qvec =
      reconstruction_->Image(image_id).CamFromWorld().rotation.coeffs().data();
  return GetBlockIndex(qvec);
}

int BundleAdjustmentCovarianceEstimatorBase::GetPoseTangentSize(
    image_t image_id) const {
  THROW_CHECK(HasPose(image_id));
  const double* qvec =
      reconstruction_->Image(image_id).CamFromWorld().rotation.coeffs().data();
  const double* tvec =
      reconstruction_->Image(image_id).CamFromWorld().translation.data();
  return GetBlockTangentSize(qvec) + GetBlockTangentSize(tvec);
}

bool BundleAdjustmentCovarianceEstimatorBase::HasValidPoseCovariance() const {
  return cov_poses_.size() != 0;
}

bool BundleAdjustmentCovarianceEstimatorBase::HasValidFullCovariance() const {
  return cov_variables_.size() != 0;
}

Eigen::MatrixXd BundleAdjustmentCovarianceEstimatorBase::GetPoseCovariance()
    const {
  THROW_CHECK(HasValidPoseCovariance());
  return cov_poses_;
}

Eigen::MatrixXd BundleAdjustmentCovarianceEstimatorBase::GetPoseCovariance(
    image_t image_id) const {
  THROW_CHECK(HasValidPoseCovariance());
  THROW_CHECK(HasPose(image_id));
  int index = GetPoseIndex(image_id);
  int num_params_pose = GetPoseTangentSize(image_id);
  return cov_poses_.block(index, index, num_params_pose, num_params_pose);
}

Eigen::MatrixXd BundleAdjustmentCovarianceEstimatorBase::GetPoseCovariance(
    const std::vector<image_t>& image_ids) const {
  THROW_CHECK(HasValidPoseCovariance());
  std::vector<int> indices;
  for (const auto& image_id : image_ids) {
    THROW_CHECK(HasPose(image_id));
    int index = GetPoseIndex(image_id);
    int num_params_pose = GetPoseTangentSize(image_id);
    for (int i = 0; i < num_params_pose; ++i) {
      indices.push_back(index + i);
    }
  }
  size_t n_indices = indices.size();
  Eigen::MatrixXd output(n_indices, n_indices);
  for (size_t i = 0; i < n_indices; ++i) {
    for (size_t j = 0; j < n_indices; ++j) {
      output(i, j) = cov_poses_(indices[i], indices[j]);
    }
  }
  return output;
}

Eigen::MatrixXd BundleAdjustmentCovarianceEstimatorBase::GetPoseCovariance(
    image_t image_id1, image_t image_id2) const {
  THROW_CHECK(HasValidPoseCovariance());
  THROW_CHECK(HasPose(image_id1));
  THROW_CHECK(HasPose(image_id2));
  int index1 = GetPoseIndex(image_id1);
  int num_params_pose1 = GetPoseTangentSize(image_id1);
  int index2 = GetPoseIndex(image_id2);
  int num_params_pose2 = GetPoseTangentSize(image_id2);
  return cov_poses_.block(index1, index2, num_params_pose1, num_params_pose2);
}

Eigen::MatrixXd BundleAdjustmentCovarianceEstimatorBase::GetCovariance(
    double* params) const {
  THROW_CHECK(HasValidFullCovariance());
  THROW_CHECK(HasBlock(params));
  int index = GetBlockIndex(params);
  int num_params = GetBlockTangentSize(params);
  return cov_variables_.block(index, index, num_params, num_params);
}

Eigen::MatrixXd BundleAdjustmentCovarianceEstimatorBase::GetCovariance(
    const std::vector<double*>& blocks) const {
  THROW_CHECK(HasValidFullCovariance());
  std::vector<int> indices;
  for (const double* block : blocks) {
    THROW_CHECK(HasBlock(block));
    int index = GetBlockIndex(block);
    int num_params_pose = GetBlockTangentSize(block);
    for (int i = 0; i < num_params_pose; ++i) {
      indices.push_back(index + i);
    }
  }
  size_t n_indices = indices.size();
  Eigen::MatrixXd output(n_indices, n_indices);
  for (size_t i = 0; i < n_indices; ++i) {
    for (size_t j = 0; j < n_indices; ++j) {
      output(i, j) = cov_variables_(indices[i], indices[j]);
    }
  }
  return output;
}

Eigen::MatrixXd BundleAdjustmentCovarianceEstimatorBase::GetCovariance(
    double* params1, double* params2) const {
  THROW_CHECK(HasValidFullCovariance());
  THROW_CHECK(HasBlock(params1));
  THROW_CHECK(HasBlock(params2));
  int index1 = GetBlockIndex(params1);
  int num_params_block1 = GetBlockTangentSize(params1);
  int index2 = GetBlockIndex(params2);
  int num_params_block2 = GetBlockTangentSize(params2);
  return cov_variables_.block(
      index1, index2, num_params_block1, num_params_block2);
}

bool BundleAdjustmentCovarianceEstimatorCeresBackend::ComputeFull() {
  ceres::Covariance::Options options;
  ceres::Covariance covariance_computer(options);
  std::vector<const double*> parameter_blocks;
  parameter_blocks.insert(
      parameter_blocks.end(), pose_blocks_.begin(), pose_blocks_.end());
  parameter_blocks.insert(parameter_blocks.end(),
                          other_variables_blocks_.begin(),
                          other_variables_blocks_.end());
  if (!covariance_computer.Compute(parameter_blocks, problem_)) return false;
  int num_params = num_params_poses_ + num_params_other_variables_;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> covs(
      num_params, num_params);
  covariance_computer.GetCovarianceMatrixInTangentSpace(parameter_blocks,
                                                        covs.data());
  cov_variables_ = covs;
  cov_poses_ = cov_variables_.block(0, 0, num_params_poses_, num_params_poses_);
  return true;
}

bool BundleAdjustmentCovarianceEstimatorCeresBackend::Compute() {
  ceres::Covariance::Options options;
  ceres::Covariance covariance_computer(options);
  std::vector<const double*> parameter_blocks;
  parameter_blocks.insert(
      parameter_blocks.end(), pose_blocks_.begin(), pose_blocks_.end());
  if (!covariance_computer.Compute(parameter_blocks, problem_)) return false;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> covs(
      num_params_poses_, num_params_poses_);
  covariance_computer.GetCovarianceMatrixInTangentSpace(parameter_blocks,
                                                        covs.data());
  cov_poses_ = covs;
  return true;
}

void BundleAdjustmentCovarianceEstimator::ComputeSchurComplement() {
  // Evaluate jacobian
  LOG(INFO) << "Evaluate jacobian matrix";
  ceres::Problem::EvaluateOptions eval_options;
  eval_options.parameter_blocks.clear();
  for (const double* block : pose_blocks_) {
    eval_options.parameter_blocks.push_back(const_cast<double*>(block));
  }
  for (const double* block : other_variables_blocks_) {
    eval_options.parameter_blocks.push_back(const_cast<double*>(block));
  }
  for (const double* block : point_blocks_) {
    eval_options.parameter_blocks.push_back(const_cast<double*>(block));
  }
  ceres::CRSMatrix J_full_crs;
  problem_->Evaluate(eval_options, nullptr, nullptr, nullptr, &J_full_crs);
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
      J_full.block(0, 0, num_residuals, num_params - num_params_points_);
  Eigen::SparseMatrix<double> J_p = J_full.block(
      0, num_params - num_params_points_, num_residuals, num_params_points_);
  Eigen::SparseMatrix<double> H_cc = J_c.transpose() * J_c;
  Eigen::SparseMatrix<double> H_cp = J_c.transpose() * J_p;
  Eigen::SparseMatrix<double> H_pc = H_cp.transpose();
  Eigen::SparseMatrix<double> H_pp = J_p.transpose() * J_p;
  // in-place computation of H_pp_inv
  Eigen::SparseMatrix<double>& H_pp_inv = H_pp;  // reference
  int counter_p = 0;
  for (const double* block : point_blocks_) {
    int num_params_point = ParameterBlockTangentSize(problem_, block);
    Eigen::SparseMatrix<double> subMatrix_sparse =
        H_pp.block(counter_p, counter_p, num_params_point, num_params_point);
    Eigen::MatrixXd subMatrix = subMatrix_sparse;
    subMatrix +=
        lambda_ * Eigen::MatrixXd::Identity(subMatrix.rows(), subMatrix.cols());
    Eigen::MatrixXd subMatrix_inv = subMatrix.inverse();
    // update matrix
    for (int i = 0; i < num_params_point; ++i) {
      for (int j = 0; j < num_params_point; ++j) {
        H_pp_inv.coeffRef(counter_p + i, counter_p + j) = subMatrix_inv(i, j);
      }
    }
    counter_p += num_params_point;
  }
  H_pp_inv.makeCompressed();
  S_matrix_ = H_cc - H_cp * H_pp_inv * H_pc;
}

bool BundleAdjustmentCovarianceEstimator::HasValidSchurComplement() const {
  return S_matrix_.size() != 0;
}

bool BundleAdjustmentCovarianceEstimator::ComputeFull() {
  if (!HasValidSchurComplement()) {
    ComputeSchurComplement();
  }
  LOG(INFO) << StringPrintf(
      "Inverting Schur complement for all variables except for 3D points (n = "
      "%d)",
      num_params_poses_ + num_params_other_variables_);
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldltOfS(S_matrix_);
  int rank = 0;
  for (int i = 0; i < S_matrix_.rows(); ++i) {
    if (ldltOfS.vectorD().coeff(i) != 0.0) rank++;
  }
  if (rank < S_matrix_.rows()) {
    LOG(INFO) << StringPrintf(
        "Unable to compute covariance. The Schur complement on all variables "
        "(except for 3D points) is rank deficient. Number of columns: %d, "
        "rank: %d.",
        S_matrix_.rows(),
        rank);
    return false;
  }
  Eigen::SparseMatrix<double> I(S_matrix_.rows(), S_matrix_.cols());
  I.setIdentity();
  Eigen::SparseMatrix<double> S_inv = ldltOfS.solve(I);
  cov_variables_ = S_inv;  // convert to dense matrix
  cov_poses_ = cov_variables_.block(0, 0, num_params_poses_, num_params_poses_);
  return true;
}

bool BundleAdjustmentCovarianceEstimator::Compute() {
  if (!HasValidSchurComplement()) {
    ComputeSchurComplement();
  }

  // Schur elimination on other variables
  LOG(INFO) << StringPrintf("Schur elimination on other variables (n = %d)",
                            num_params_other_variables_);
  Eigen::SparseMatrix<double> S_aa =
      S_matrix_.block(0, 0, num_params_poses_, num_params_poses_);
  Eigen::SparseMatrix<double> S_ab = S_matrix_.block(
      0, num_params_poses_, num_params_poses_, num_params_other_variables_);
  Eigen::SparseMatrix<double> S_ba = S_ab.transpose();
  Eigen::SparseMatrix<double> S_bb =
      S_matrix_.block(num_params_poses_,
                      num_params_poses_,
                      num_params_other_variables_,
                      num_params_other_variables_);
  for (int i = 0; i < S_bb.rows(); ++i) {
    if (S_bb.coeff(i, i) == 0.0)
      S_bb.coeffRef(i, i) = lambda_;
    else
      S_bb.coeffRef(i, i) += lambda_;
  }
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> lltOfS_bb(S_bb);
  Eigen::SparseMatrix<double> S_poses = S_aa - S_ab * lltOfS_bb.solve(S_ba);

  // Compute pose covariance
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldltOfS_poses(S_poses);
  int rank = 0;
  for (int i = 0; i < S_matrix_.rows(); ++i) {
    if (ldltOfS_poses.vectorD().coeff(i) != 0.0) rank++;
  }
  if (rank < S_poses.rows()) {
    LOG(INFO) << StringPrintf(
        "Unable to compute covariance. The Schur complement on pose parameters "
        "is rank deficient. Number of columns: %d, rank: %d. This is likely "
        "due to the poses being underconstrained with Gauge ambiguity.",
        S_poses.rows(),
        rank);
    return false;
  }
  Eigen::SparseMatrix<double> I(S_poses.rows(), S_poses.cols());
  I.setIdentity();
  Eigen::SparseMatrix<double> S_poses_inv = ldltOfS_poses.solve(I);
  cov_poses_ = S_poses_inv;  // convert to dense matrix
  return true;
}

bool EstimatePoseCovarianceCeresBackend(
    ceres::Problem* problem,
    Reconstruction* reconstruction,
    std::map<image_t, Eigen::MatrixXd>& image_id_to_covar) {
  BundleAdjustmentCovarianceEstimatorCeresBackend estimator(problem,
                                                            reconstruction);
  if (!estimator.Compute()) return false;
  image_id_to_covar.clear();
  for (const auto& image : reconstruction->Images()) {
    image_t image_id = image.first;
    if (!estimator.HasPose(image_id)) continue;
    image_id_to_covar.emplace(image_id, estimator.GetPoseCovariance(image_id));
  }
  return true;
}

bool EstimatePoseCovariance(
    ceres::Problem* problem,
    Reconstruction* reconstruction,
    std::map<image_t, Eigen::MatrixXd>& image_id_to_covar,
    double lambda) {
  BundleAdjustmentCovarianceEstimator estimator(
      problem, reconstruction, lambda);
  if (!estimator.Compute()) return false;
  image_id_to_covar.clear();
  for (const auto& image : reconstruction->Images()) {
    image_t image_id = image.first;
    if (!estimator.HasPose(image_id)) continue;
    image_id_to_covar.emplace(image_id, estimator.GetPoseCovariance(image_id));
  }
  return true;
}

}  // namespace colmap

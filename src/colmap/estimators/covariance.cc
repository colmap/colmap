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

#include <ceres/crs_matrix.h>

namespace colmap {

BundleAdjustmentCovarianceEstimatorBase::
    BundleAdjustmentCovarianceEstimatorBase(ceres::Problem* problem,
                                            Reconstruction* reconstruction)
    : problem_(THROW_CHECK_NOTNULL(problem)),
      reconstruction_(THROW_CHECK_NOTNULL(reconstruction)) {
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
    }

    const double* tvec = image.second.CamFromWorld().translation.data();
    if (problem_->HasParameterBlock(tvec) &&
        !problem_->IsParameterBlockConstant(const_cast<double*>(tvec))) {
      int num_params_tvec = ParameterBlockTangentSize(problem_, tvec);
      pose_blocks_.push_back(tvec);
      map_block_to_index_.emplace(tvec, num_params_poses_);
      num_params_poses_ += num_params_tvec;
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
    }
  }

  // Parse parameter blocks for other variables
  SetUpOtherVariablesBlocks();
}

BundleAdjustmentCovarianceEstimatorBase::
    BundleAdjustmentCovarianceEstimatorBase(
        ceres::Problem* problem,
        const std::vector<const double*>& pose_blocks,
        const std::vector<const double*>& point_blocks)
    : problem_(THROW_CHECK_NOTNULL(problem)) {
  // Parse parameter blocks for 3D points
  point_blocks_.clear();
  num_params_points_ = 0;
  for (const double* block : point_blocks) {
    THROW_CHECK(
        problem_->HasParameterBlock(block) &&
        !problem_->IsParameterBlockConstant(const_cast<double*>(block)));
    int num_params_point = ParameterBlockTangentSize(problem_, block);
    point_blocks_.push_back(block);
    num_params_points_ += num_params_point;
  }

  // Parse parameter blocks for poses
  SetPoseBlocks(pose_blocks);
}

void BundleAdjustmentCovarianceEstimatorBase::SetPoseBlocks(
    const std::vector<const double*>& pose_blocks) {
  // Reset block indexing map
  map_block_to_index_.clear();

  // Parse parameter blocks for poses
  pose_blocks_.clear();
  num_params_poses_ = 0;
  for (const double* block : pose_blocks) {
    THROW_CHECK(
        problem_->HasParameterBlock(block) &&
        !problem_->IsParameterBlockConstant(const_cast<double*>(block)));

    int num_params_block = ParameterBlockTangentSize(problem_, block);
    pose_blocks_.push_back(block);
    map_block_to_index_.emplace(block, num_params_poses_);
    num_params_poses_ += num_params_block;
  }

  // Parse parameter blocks for other variables
  SetUpOtherVariablesBlocks();
}

void BundleAdjustmentCovarianceEstimatorBase::SetUpOtherVariablesBlocks() {
  // Construct a set for excluding pose and point parameter blocks
  std::set<const double*> pose_and_point_parameter_blocks;
  for (const double* block : pose_blocks_)
    pose_and_point_parameter_blocks.insert(block);
  for (const double* block : point_blocks_)
    pose_and_point_parameter_blocks.insert(block);

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
    map_block_to_index_.emplace(
        block, num_params_poses_ + num_params_other_variables_);
    int num_params_block = ParameterBlockTangentSize(problem_, block);
    num_params_other_variables_ += num_params_block;
  }
}

bool BundleAdjustmentCovarianceEstimatorBase::HasReconstruction() const {
  return reconstruction_ != nullptr;
}

bool BundleAdjustmentCovarianceEstimatorBase::HasBlock(
    const double* params) const {
  return map_block_to_index_.find(params) != map_block_to_index_.end();
}

bool BundleAdjustmentCovarianceEstimatorBase::HasPoseBlock(
    const double* params) const {
  const auto it = map_block_to_index_.find(params);
  if (it == map_block_to_index_.end()) return false;
  return it->second < num_params_poses_;
}

bool BundleAdjustmentCovarianceEstimatorBase::HasPose(image_t image_id) const {
  THROW_CHECK(HasReconstruction());
  const double* qvec =
      reconstruction_->Image(image_id).CamFromWorld().rotation.coeffs().data();
  const double* tvec =
      reconstruction_->Image(image_id).CamFromWorld().translation.data();
  return HasPoseBlock(qvec) && HasPoseBlock(tvec);
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
  THROW_CHECK(HasReconstruction());
  const double* qvec =
      reconstruction_->Image(image_id).CamFromWorld().rotation.coeffs().data();
  return GetBlockIndex(qvec);
}

int BundleAdjustmentCovarianceEstimatorBase::GetPoseTangentSize(
    image_t image_id) const {
  THROW_CHECK(HasReconstruction());
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

double BundleAdjustmentCovarianceEstimatorBase::GetPoseCovarianceByIndex(
    int row, int col) const {
  THROW_CHECK(HasValidPoseCovariance());
  return cov_poses_(row, col);
}

Eigen::MatrixXd
BundleAdjustmentCovarianceEstimatorBase::GetPoseCovarianceBlockOperation(
    int row_start,
    int col_start,
    int row_block_size,
    int col_block_size) const {
  THROW_CHECK(HasValidPoseCovariance());
  return cov_poses_.block(row_start, col_start, row_block_size, col_block_size);
}

double BundleAdjustmentCovarianceEstimatorBase::GetCovarianceByIndex(
    int row, int col) const {
  THROW_CHECK(HasValidFullCovariance());
  return cov_variables_(row, col);
}

Eigen::MatrixXd
BundleAdjustmentCovarianceEstimatorBase::GetCovarianceBlockOperation(
    int row_start,
    int col_start,
    int row_block_size,
    int col_block_size) const {
  THROW_CHECK(HasValidFullCovariance());
  return cov_variables_.block(
      row_start, col_start, row_block_size, col_block_size);
}

Eigen::MatrixXd BundleAdjustmentCovarianceEstimatorBase::GetPoseCovariance()
    const {
  THROW_CHECK(HasValidPoseCovariance());
  return cov_poses_;
}

Eigen::MatrixXd BundleAdjustmentCovarianceEstimatorBase::GetPoseCovariance(
    image_t image_id) const {
  THROW_CHECK(HasReconstruction());
  THROW_CHECK(HasPose(image_id));
  int index = GetPoseIndex(image_id);
  int num_params_pose = GetPoseTangentSize(image_id);
  return GetPoseCovarianceBlockOperation(
      index, index, num_params_pose, num_params_pose);
}

Eigen::MatrixXd BundleAdjustmentCovarianceEstimatorBase::GetPoseCovariance(
    const std::vector<image_t>& image_ids) const {
  THROW_CHECK(HasReconstruction());
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
      output(i, j) = GetPoseCovarianceByIndex(indices[i], indices[j]);
    }
  }
  return output;
}

Eigen::MatrixXd BundleAdjustmentCovarianceEstimatorBase::GetPoseCovariance(
    image_t image_id1, image_t image_id2) const {
  THROW_CHECK(HasReconstruction());
  THROW_CHECK(HasPose(image_id1));
  THROW_CHECK(HasPose(image_id2));
  int index1 = GetPoseIndex(image_id1);
  int num_params_pose1 = GetPoseTangentSize(image_id1);
  int index2 = GetPoseIndex(image_id2);
  int num_params_pose2 = GetPoseTangentSize(image_id2);
  return GetPoseCovarianceBlockOperation(
      index1, index2, num_params_pose1, num_params_pose2);
}

Eigen::MatrixXd BundleAdjustmentCovarianceEstimatorBase::GetPoseCovariance(
    double* params) const {
  THROW_CHECK(HasPoseBlock(params));
  int index = GetBlockIndex(params);
  int num_params = GetBlockTangentSize(params);
  return GetPoseCovarianceBlockOperation(index, index, num_params, num_params);
}

Eigen::MatrixXd BundleAdjustmentCovarianceEstimatorBase::GetPoseCovariance(
    const std::vector<double*>& blocks) const {
  std::vector<int> indices;
  for (const double* block : blocks) {
    THROW_CHECK(HasPoseBlock(block));
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
      output(i, j) = GetPoseCovarianceByIndex(indices[i], indices[j]);
    }
  }
  return output;
}

Eigen::MatrixXd BundleAdjustmentCovarianceEstimatorBase::GetPoseCovariance(
    double* params1, double* params2) const {
  THROW_CHECK(HasPoseBlock(params1));
  THROW_CHECK(HasPoseBlock(params2));
  int index1 = GetBlockIndex(params1);
  int num_params_block1 = GetBlockTangentSize(params1);
  int index2 = GetBlockIndex(params2);
  int num_params_block2 = GetBlockTangentSize(params2);
  return GetPoseCovarianceBlockOperation(
      index1, index2, num_params_block1, num_params_block2);
}

Eigen::MatrixXd BundleAdjustmentCovarianceEstimatorBase::GetCovariance(
    double* params) const {
  THROW_CHECK(HasBlock(params));
  int index = GetBlockIndex(params);
  int num_params = GetBlockTangentSize(params);
  return GetCovarianceBlockOperation(index, index, num_params, num_params);
}

Eigen::MatrixXd BundleAdjustmentCovarianceEstimatorBase::GetCovariance(
    const std::vector<double*>& blocks) const {
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
      output(i, j) = GetCovarianceByIndex(indices[i], indices[j]);
    }
  }
  return output;
}

Eigen::MatrixXd BundleAdjustmentCovarianceEstimatorBase::GetCovariance(
    double* params1, double* params2) const {
  THROW_CHECK(HasBlock(params1));
  THROW_CHECK(HasBlock(params2));
  int index1 = GetBlockIndex(params1);
  int num_params_block1 = GetBlockTangentSize(params1);
  int index2 = GetBlockIndex(params2);
  int num_params_block2 = GetBlockTangentSize(params2);
  return GetCovarianceBlockOperation(
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
  if (!covariance_computer.Compute(pose_blocks_, problem_)) return false;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> covs(
      num_params_poses_, num_params_poses_);
  covariance_computer.GetCovarianceMatrixInTangentSpace(pose_blocks_,
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

  // Schur elimination on points
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
    Eigen::SparseMatrix<double> sub_matrix_sparse =
        H_pp.block(counter_p, counter_p, num_params_point, num_params_point);
    Eigen::MatrixXd sub_matrix = sub_matrix_sparse;
    sub_matrix += lambda_ * Eigen::MatrixXd::Identity(sub_matrix.rows(),
                                                      sub_matrix.cols());
    Eigen::MatrixXd sub_matrix_inv = sub_matrix.inverse();
    // update matrix
    for (int i = 0; i < num_params_point; ++i) {
      for (int j = 0; j < num_params_point; ++j) {
        H_pp_inv.coeffRef(counter_p + i, counter_p + j) = sub_matrix_inv(i, j);
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

bool BundleAdjustmentCovarianceEstimator::HasValidPoseFactorization() const {
  return L_matrix_poses_inv_.size() != 0;
}

double BundleAdjustmentCovarianceEstimator::GetPoseCovarianceByIndex(
    int row, int col) const {
  THROW_CHECK(HasValidPoseCovariance() || HasValidPoseFactorization());
  if (HasValidPoseCovariance())
    return cov_poses_(row, col);
  else
    return L_matrix_poses_inv_.col(row).dot(L_matrix_poses_inv_.col(col));
}

Eigen::MatrixXd
BundleAdjustmentCovarianceEstimator::GetPoseCovarianceBlockOperation(
    int row_start,
    int col_start,
    int row_block_size,
    int col_block_size) const {
  THROW_CHECK(HasValidPoseCovariance() || HasValidPoseFactorization());
  if (HasValidPoseCovariance()) {
    return cov_poses_.block(
        row_start, col_start, row_block_size, col_block_size);
  }
  // HasValidPoseRefactorization() == true
  Eigen::MatrixXd output(row_block_size, col_block_size);
  for (int row = 0; row < row_block_size; ++row) {
    for (int col = 0; col < col_block_size; ++col) {
      output(row, col) = L_matrix_poses_inv_.col(row_start + row)
                             .dot(L_matrix_poses_inv_.col(col_start + col));
    }
  }
  return output;
}

bool BundleAdjustmentCovarianceEstimator::HasValidFullFactorization() const {
  return L_matrix_variables_inv_.size() != 0;
}

double BundleAdjustmentCovarianceEstimator::GetCovarianceByIndex(
    int row, int col) const {
  THROW_CHECK(HasValidFullCovariance() || HasValidFullFactorization());
  if (HasValidFullCovariance())
    return cov_variables_(row, col);
  else
    return L_matrix_variables_inv_.col(row).dot(
        L_matrix_variables_inv_.col(col));
}

Eigen::MatrixXd
BundleAdjustmentCovarianceEstimator::GetCovarianceBlockOperation(
    int row_start,
    int col_start,
    int row_block_size,
    int col_block_size) const {
  THROW_CHECK(HasValidFullCovariance() || HasValidFullFactorization());
  if (HasValidFullCovariance()) {
    return cov_variables_.block(
        row_start, col_start, row_block_size, col_block_size);
  }
  // HasValidRefactorization() == true
  Eigen::MatrixXd output(row_block_size, col_block_size);
  for (int row = 0; row < row_block_size; ++row) {
    for (int col = 0; col < col_block_size; ++col) {
      output(row, col) = L_matrix_variables_inv_.col(row_start + row)
                             .dot(L_matrix_variables_inv_.col(col_start + col));
    }
  }
  return output;
}

bool BundleAdjustmentCovarianceEstimator::FactorizeFull() {
  if (!HasValidSchurComplement()) {
    ComputeSchurComplement();
  }
  LOG(INFO) << StringPrintf(
      "Inverting Schur complement for all variables except for 3D points (n = "
      "%d)",
      num_params_poses_ + num_params_other_variables_);
  LOG(INFO) << StringPrintf("Start sparse Cholesky decomposition (n = %d)",
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
  LOG(INFO) << "Finish sparse Cholesky decomposition.";
  // construct the inverse of the L_matrix
  Eigen::SparseMatrix<double> L = ldltOfS.matrixL();
  for (int i = 0; i < S_matrix_.rows(); ++i) {
    for (int k = L.outerIndexPtr()[i]; k < L.outerIndexPtr()[i + 1]; ++k) {
      L.valuePtr()[i] *= sqrt(std::max(ldltOfS.vectorD().coeff(i), 0.));
    }
  }
  Eigen::MatrixXd L_dense = L;
  L_matrix_variables_inv_ = L_dense.triangularView<Eigen::Lower>().solve(
      Eigen::MatrixXd::Identity(L_dense.rows(), L_dense.cols()));
  LOG(INFO) << "Finish factorization by having the lower triangular matrix L "
               "inverted.";
  return true;
}

bool BundleAdjustmentCovarianceEstimator::ComputeFull() {
  if (!HasValidSchurComplement()) {
    ComputeSchurComplement();
  }
  LOG(INFO) << StringPrintf(
      "Inverting Schur complement for all variables except for 3D points (n = "
      "%d)",
      num_params_poses_ + num_params_other_variables_);
  LOG(INFO) << StringPrintf("Start sparse Cholesky decomposition (n = %d)",
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
  LOG(INFO) << "Finish sparse Cholesky decomposition.";
  Eigen::SparseMatrix<double> I(S_matrix_.rows(), S_matrix_.cols());
  I.setIdentity();
  Eigen::SparseMatrix<double> S_inv = ldltOfS.solve(I);
  cov_variables_ = S_inv;  // convert to dense matrix
  cov_poses_ = cov_variables_.block(0, 0, num_params_poses_, num_params_poses_);
  return true;
}

bool BundleAdjustmentCovarianceEstimator::Factorize() {
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
  LOG(INFO) << StringPrintf("Start sparse Cholesky decomposition (n = %d)",
                            num_params_poses_);
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldltOfS_poses(S_poses);
  int rank = 0;
  for (int i = 0; i < S_poses.rows(); ++i) {
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
  LOG(INFO) << "Finish sparse Cholesky decomposition.";
  // construct the inverse of the L_matrix
  Eigen::SparseMatrix<double> L_poses = ldltOfS_poses.matrixL();
  for (int i = 0; i < L_poses.rows(); ++i) {
    for (int k = L_poses.outerIndexPtr()[i]; k < L_poses.outerIndexPtr()[i + 1];
         ++k) {
      L_poses.valuePtr()[k] *=
          sqrt(std::max(ldltOfS_poses.vectorD().coeff(i), 0.));
    }
  }
  Eigen::MatrixXd L_poses_dense = L_poses;
  L_matrix_poses_inv_ = L_poses_dense.triangularView<Eigen::Lower>().solve(
      Eigen::MatrixXd::Identity(L_poses_dense.rows(), L_poses_dense.cols()));
  LOG(INFO) << "Finish factorization by having the lower triangular matrix L "
               "inverted.";
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
  LOG(INFO) << StringPrintf("Start sparse Cholesky decomposition (n = %d)",
                            num_params_poses_);
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldltOfS_poses(S_poses);
  int rank = 0;
  for (int i = 0; i < S_poses.rows(); ++i) {
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
  LOG(INFO) << "Finish sparse Cholesky decomposition.";
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
  if (!estimator.Factorize()) return false;
  image_id_to_covar.clear();
  for (const auto& image : reconstruction->Images()) {
    image_t image_id = image.first;
    if (!estimator.HasPose(image_id)) continue;
    image_id_to_covar.emplace(image_id, estimator.GetPoseCovariance(image_id));
  }
  return true;
}

}  // namespace colmap

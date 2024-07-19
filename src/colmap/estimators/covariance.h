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

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/scene/reconstruction.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ceres/ceres.h>

namespace colmap {

// Covariance estimation for bundle adjustment (or extended) problem.
// The interface is applicable to all ceres problem extended on top of bundle
// adjustment. The Schur complement is computed explicitly to eliminate the
// hessian block for all the 3D points, which is essential to avoid Jacobian
// rank deficiency for large-scale reconstruction
class BundleAdjustmentCovarianceEstimatorBase {
 public:
  // Construct with a COLMAP reconstruction
  BundleAdjustmentCovarianceEstimatorBase(ceres::Problem* problem,
                                          Reconstruction* reconstruction);
  // Construct by specifying pose blocks and point blocks
  BundleAdjustmentCovarianceEstimatorBase(
      ceres::Problem* problem,
      const std::vector<const double*>& pose_blocks,
      const std::vector<const double*>& point_blocks);
  virtual ~BundleAdjustmentCovarianceEstimatorBase() = default;

  // Manually set pose blocks that are interested
  void SetPoseBlocks(const std::vector<const double*>& pose_blocks);

  // Compute covariance for all parameters (except for 3D points).
  // Store the full matrix at cov_variables_ and the subblock copy at
  // cov_poses_;
  virtual bool ComputeFull() = 0;

  // Compute covariance for pose paramters.
  // Stored at cov_poses_;
  virtual bool Compute() = 0;

  // Interfaces
  // test if the block corresponds to any parameter in the problem except for 3D
  // points
  bool HasBlock(const double* params) const;
  // test if the block corresponds to any parameter in the pose_blocks
  bool HasPoseBlock(const double* params) const;
  // test if the estimator is constructed with a COLMAP reconstruction
  bool HasReconstruction() const;
  // test if the pose is inside the problem as non-constant variables
  bool HasPose(image_t image_id) const;

  // pose parameters
  Eigen::MatrixXd GetPoseCovariance() const;
  Eigen::MatrixXd GetPoseCovariance(image_t image_id) const;
  Eigen::MatrixXd GetPoseCovariance(
      const std::vector<image_t>& image_ids) const;
  Eigen::MatrixXd GetPoseCovariance(image_t image_id1, image_t image_id2) const;
  Eigen::MatrixXd GetPoseCovariance(double* parameter_block) const;
  Eigen::MatrixXd GetPoseCovariance(
      const std::vector<double*>& parameter_blocks) const;
  Eigen::MatrixXd GetPoseCovariance(double* parameter_block1,
                                    double* parameter_block2) const;

  // all parameters (except for 3D points)
  Eigen::MatrixXd GetCovariance(double* parameter_block) const;
  Eigen::MatrixXd GetCovariance(
      const std::vector<double*>& parameter_blocks) const;
  Eigen::MatrixXd GetCovariance(double* parameter_block1,
                                double* parameter_block2) const;

  // test if either ``ComputeFull()`` or ``Compute()`` has been called
  bool HasValidPoseCovariance() const;
  // test if ``ComputeFull()`` has been called
  bool HasValidFullCovariance() const;

 protected:
  // indexing the covariance matrix
  virtual double GetCovarianceByIndex(int row, int col) const;
  virtual Eigen::MatrixXd GetCovarianceBlockOperation(int row_start,
                                                      int col_start,
                                                      int row_block_size,
                                                      int col_block_size) const;
  virtual double GetPoseCovarianceByIndex(int row, int col) const;
  virtual Eigen::MatrixXd GetPoseCovarianceBlockOperation(
      int row_start,
      int col_start,
      int row_block_size,
      int col_block_size) const;

  // blocks parsed from reconstruction (initialized at construction)
  std::vector<const double*> pose_blocks_;
  int num_params_poses_ = 0;
  std::vector<const double*> other_variables_blocks_;
  int num_params_other_variables_ = 0;
  std::vector<const double*> point_blocks_;
  int num_params_points_ = 0;

  // get the starting index of the parameter block in the matrix
  // orders: [pose_blocks, other_variables_blocks, point_blocks]
  std::map<const double*, int> map_block_to_index_;

  int GetBlockIndex(const double* params) const;
  int GetBlockTangentSize(const double* params) const;
  int GetPoseIndex(image_t image_id) const;
  int GetPoseTangentSize(image_t image_id) const;

  // covariance for all parameters (except for 3D points)
  Eigen::MatrixXd cov_variables_;

  // covariance for pose parameters
  Eigen::MatrixXd cov_poses_;

  // ceres problem
  ceres::Problem* problem_;

  // reconstruction
  Reconstruction* reconstruction_ = nullptr;

 private:
  // set up parameter blocks
  void SetUpOtherVariablesBlocks();
};

class BundleAdjustmentCovarianceEstimatorCeresBackend
    : public BundleAdjustmentCovarianceEstimatorBase {
 public:
  BundleAdjustmentCovarianceEstimatorCeresBackend(
      ceres::Problem* problem, Reconstruction* reconstruction)
      : BundleAdjustmentCovarianceEstimatorBase(problem, reconstruction) {}

  BundleAdjustmentCovarianceEstimatorCeresBackend(
      ceres::Problem* problem,
      const std::vector<const double*>& pose_blocks,
      const std::vector<const double*>& point_blocks)
      : BundleAdjustmentCovarianceEstimatorBase(
            problem, pose_blocks, point_blocks) {}

  bool ComputeFull() override;
  bool Compute() override;
};

class BundleAdjustmentCovarianceEstimator
    : public BundleAdjustmentCovarianceEstimatorBase {
 public:
  BundleAdjustmentCovarianceEstimator(ceres::Problem* problem,
                                      Reconstruction* reconstruction,
                                      double lambda = 1e-8)
      : BundleAdjustmentCovarianceEstimatorBase(problem, reconstruction),
        lambda_(lambda) {}
  BundleAdjustmentCovarianceEstimator(
      ceres::Problem* problem,
      const std::vector<const double*>& pose_blocks,
      const std::vector<const double*>& point_blocks,
      double lambda = 1e-8)
      : BundleAdjustmentCovarianceEstimatorBase(
            problem, pose_blocks, point_blocks),
        lambda_(lambda) {}

  bool ComputeFull() override;
  bool Compute() override;

  // factorization
  bool FactorizeFull();
  bool Factorize();
  bool HasValidFullFactorization() const;
  bool HasValidPoseFactorization() const;

 private:
  // indexing the covariance matrix
  double GetCovarianceByIndex(int row, int col) const override;
  Eigen::MatrixXd GetCovarianceBlockOperation(
      int row_start,
      int col_start,
      int row_block_size,
      int col_block_size) const override;
  double GetPoseCovarianceByIndex(int row, int col) const override;
  Eigen::MatrixXd GetPoseCovarianceBlockOperation(
      int row_start,
      int col_start,
      int row_block_size,
      int col_block_size) const override;

  // The Schur complement for all parameters (except for 3D points) after Schur
  // elimination
  Eigen::SparseMatrix<double> S_matrix_;

  // The damping factor to avoid rank deficiency
  const double lambda_ = 1e-8;

  // Compute the Schur complement for poses and other variables by eliminating
  // 3D points
  void ComputeSchurComplement();
  bool HasValidSchurComplement() const;

  // The inverse of L matrix after Cholesky factorization
  Eigen::MatrixXd L_matrix_variables_inv_;
  Eigen::MatrixXd L_matrix_poses_inv_;
};

// The covariance for each image is in the order [R, t] with both of them
// potentially on manifold (R is always at least parameterized with
// ceres::QuaternionManifold on Lie Algebra). As a result, the covariance is
// only computed on the non-constant part for each variable. If the full parts
// of both the rotation and translation are in the problem, the covariance
// matrix will be 6x6.
bool EstimatePoseCovarianceCeresBackend(
    ceres::Problem* problem,
    Reconstruction* reconstruction,
    std::map<image_t, Eigen::MatrixXd>& image_id_to_covar);

// Similar to the convention above for ``EstimatePoseCovarianceCeresBackend``.
bool EstimatePoseCovariance(
    ceres::Problem* problem,
    Reconstruction* reconstruction,
    std::map<image_t, Eigen::MatrixXd>& image_id_to_covar,
    double lambda = 1e-8);

}  // namespace colmap

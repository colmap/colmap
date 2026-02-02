// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/math/math.h"
#include "colmap/optim/ransac.h"

#include <ceres/ceres.h>

namespace colmap {

// Ceres-specific bundle adjustment options.
struct CeresBundleAdjustmentOptions {
  // Loss function types: Trivial (non-robust) and robust loss functions.
  enum class LossFunctionType { TRIVIAL, SOFT_L1, CAUCHY, HUBER };
  LossFunctionType loss_function_type = LossFunctionType::TRIVIAL;

  // Scaling factor determines residual at which robustification takes place.
  double loss_function_scale = 1.0;

  // Whether to use Ceres' CUDA linear algebra library, if available.
  bool use_gpu = false;
  std::string gpu_index = "-1";

  // Ceres-Solver options.
  ceres::Solver::Options solver_options;

  // Heuristic threshold to switch from CPU to GPU based solvers.
  // Typically, the GPU is faster for large problems but the overhead of
  // transferring memory from the CPU to the GPU leads to better CPU performance
  // for small problems. This depends on the specific problem and hardware.
  int min_num_images_gpu_solver = 50;

  // Heuristic threshold on the minimum number of residuals to enable
  // multi-threading. Note that single-threaded is typically better for small
  // bundle adjustment problems due to the overhead of threading.
  int min_num_residuals_for_cpu_multi_threading = 50000;

  // Heuristic thresholds to switch between direct, sparse, and iterative
  // solvers. These thresholds may not be optimal for all types of problems.
  int max_num_images_direct_dense_cpu_solver = 50;
  int max_num_images_direct_sparse_cpu_solver = 1000;
  int max_num_images_direct_dense_gpu_solver = 200;
  int max_num_images_direct_sparse_gpu_solver = 4000;

  // Whether to automatically select solver type based on problem size.
  // When false, uses the linear_solver_type and preconditioner_type
  // from solver_options directly.
  bool auto_select_solver_type = true;

  CeresBundleAdjustmentOptions();

  // Create loss function for given options.
  std::unique_ptr<ceres::LossFunction> CreateLossFunction() const;

  // Create options tailored for given bundle adjustment config and problem.
  ceres::Solver::Options CreateSolverOptions(
      const BundleAdjustmentConfig& config,
      const ceres::Problem& problem) const;

  bool Check() const;
};

// Ceres-specific bundle adjustment summary with access to full solver details.
struct CeresBundleAdjustmentSummary : public BundleAdjustmentSummary {
  ceres::Solver::Summary ceres_summary;

  std::string BriefReport() const override;

  static std::shared_ptr<CeresBundleAdjustmentSummary> Create(
      const ceres::Solver::Summary& ceres_summary);
};

// Ceres-specific pose prior bundle adjustment options.
struct CeresPosePriorBundleAdjustmentOptions {
  // Loss function for prior position loss.
  CeresBundleAdjustmentOptions::LossFunctionType
      prior_position_loss_function_type =
          CeresBundleAdjustmentOptions::LossFunctionType::TRIVIAL;

  // Threshold on the residual for the robust loss.
  double prior_position_loss_scale = std::sqrt(kChiSquare95ThreeDof);

  bool Check() const;
};

// Ceres-specific bundle adjuster with access to the underlying problem.
class CeresBundleAdjuster : public BundleAdjuster {
 public:
  using BundleAdjuster::BundleAdjuster;

  virtual std::shared_ptr<ceres::Problem>& Problem() = 0;
};

std::unique_ptr<BundleAdjuster> CreateDefaultCeresBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    Reconstruction& reconstruction);

std::unique_ptr<BundleAdjuster> CreatePosePriorCeresBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const PosePriorBundleAdjustmentOptions& prior_options,
    const BundleAdjustmentConfig& config,
    std::vector<PosePrior> pose_priors,
    Reconstruction& reconstruction);

void PrintSolverSummary(const ceres::Solver::Summary& summary,
                        const std::string& header);

}  // namespace colmap

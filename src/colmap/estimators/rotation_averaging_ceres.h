// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/estimators/ceres_loss_function.h"
#include "colmap/estimators/rotation_averaging.h"
#include "colmap/math/math.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/types.h"

#include <memory>
#include <optional>

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

class PoseGraph;
class Reconstruction;

// Ceres-specific rotation averaging options. Solver-agnostic options (e.g.,
// reweighting, skip_initialization, refine_sensor_from_rig) are in
// RotationEstimatorOptions.
struct CeresRotationAveragerOptions {
  // Robust loss function applied to the relative-rotation residuals.
  CeresLossFunctionType loss_function_type = CeresLossFunctionType::CAUCHY;
  // Loss function scale in degrees.
  double loss_function_scale = 0.5;

  // Loss function scale with COVARIANCE reweighting, in standard deviations
  // of the whitened (unitless) residuals. Replaces loss_function_scale.
  double covariance_loss_scale = 3.0;

  // Number of solver iterations of a Huber-loss warm-start (with the same
  // loss_function_scale) run by RotationEstimator before the main solve. The
  // linear tails of the Huber loss widen the basin of convergence from the
  // maximum spanning tree initialization, which narrow redescending losses
  // can otherwise get trapped in. With COVARIANCE reweighting, the warm-start
  // still operates on angular residuals with UNIFORM reweighting, as
  // confidently wrong relative rotations (e.g., from symmetric structures)
  // would otherwise dominate the early iterations. Set to 0 to disable.
  int max_num_warm_start_iterations = 5;

  ceres::Solver::Options solver_options;

  CeresRotationAveragerOptions();
};

// Optimizes frame and sensor rotations in place. The reconstruction must
// outlive this object. Unlike the RunRotationAveraging() pipeline, it has no
// gravity/pose priors, outlier filtering, or frame deregistration. Uses the
// Ceres-specific options in options.ceres and the solver-agnostic options
// reweighting, skip_initialization, and refine_sensor_from_rig. With COVARIANCE
// reweighting, every valid edge must already have a positive-definite
// cam2_from_cam1_rotation_cov; RotationEstimator is responsible for ensuring
// this condition.
class CeresRotationAverager {
 public:
  CeresRotationAverager(const RotationEstimatorOptions& options,
                        const PoseGraph& pose_graph,
                        Reconstruction& reconstruction);
  CeresRotationAverager(const CeresRotationAverager&) = delete;
  CeresRotationAverager& operator=(const CeresRotationAverager&) = delete;

  ceres::Solver::Options solver_options;

  ceres::Solver::Summary Solve();
  ceres::Problem& Problem();
  const ceres::Problem& Problem() const;
  // Frame and sensor rotations for both images must be configured in Problem().
  // If given, the residual is whitened with the covariance of the relative
  // rotation, expressed in the frame of camera 1 (right perturbation, see
  // RelativeRotationCostFunctor).
  void AddRelativeRotationResidual(
      image_t image_id1,
      image_t image_id2,
      const Eigen::Quaterniond& cam2_from_cam1,
      const std::shared_ptr<ceres::LossFunction>& loss_function,
      const std::optional<Eigen::Matrix3d>& cam2_from_cam1_cov = std::nullopt);

 private:
  void InitializeRotations(const RotationEstimatorOptions& options,
                           const PoseGraph& pose_graph,
                           const FlatHashSet<image_t>& image_ids);
  void SetupParameterBlocks(const RotationEstimatorOptions& options,
                            const FlatHashSet<image_t>& image_ids);

  Reconstruction& reconstruction_;
  // Keep losses alive until the problem is destroyed.
  FlatHashMap<ceres::LossFunction*, std::shared_ptr<ceres::LossFunction>>
      losses_;
  std::unique_ptr<ceres::Problem> problem_;
};

// Calibrated sensor rotations are fixed by default and can be made variable
// through Problem(). MST initialization requires sensor rotations for all rigs
// in the reconstruction when sensor refinement is disabled.
std::unique_ptr<CeresRotationAverager> CreateDefaultCeresRotationAverager(
    const RotationEstimatorOptions& options,
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction);

}  // namespace colmap

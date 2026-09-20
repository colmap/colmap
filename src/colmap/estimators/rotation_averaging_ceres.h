// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/estimators/ceres_loss_function.h"
#include "colmap/estimators/rotation_averaging.h"
#include "colmap/math/math.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/types.h"

#include <memory>

#include <ceres/ceres.h>

namespace colmap {

class PoseGraph;
class Reconstruction;

struct CeresRotationAveragerOptions {
  CeresLossFunctionType loss_function_type = CeresLossFunctionType::HUBER;
  // Loss function scale in radians.
  double loss_function_scale = DegToRad(5.0);
  RotationAveragingReweighting reweighting =
      RotationAveragingReweighting::UNIFORM;

  ceres::Solver::Options solver_options;

  // Flag to skip maximum spanning tree initialization.
  bool skip_initialization = false;

  // Refine uncalibrated sensor rotations (missing pose or NaN translation).
  // Fully calibrated sensor rotations stay constant unless the user flips
  // them via Problem().
  bool refine_sensor_from_rig = true;

  CeresRotationAveragerOptions();
};

// Optimizes frame and sensor rotations in place. The reconstruction must
// outlive this object. Unlike the RunRotationAveraging() pipeline, it has no
// gravity/pose priors, outlier filtering, or frame deregistration.
class CeresRotationAverager {
 public:
  CeresRotationAverager(const CeresRotationAveragerOptions& options,
                        const PoseGraph& pose_graph,
                        Reconstruction& reconstruction);
  CeresRotationAverager(const CeresRotationAverager&) = delete;
  CeresRotationAverager& operator=(const CeresRotationAverager&) = delete;

  ceres::Solver::Options solver_options;

  ceres::Solver::Summary Solve();
  ceres::Problem& Problem();
  const ceres::Problem& Problem() const;
  // Frame and sensor rotations for both images must be configured in Problem().
  void AddRelativeRotationResidual(
      image_t image_id1,
      image_t image_id2,
      const Eigen::Quaterniond& cam2_from_cam1,
      std::shared_ptr<ceres::LossFunction> loss_function);

 private:
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
    const CeresRotationAveragerOptions& options,
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction);

}  // namespace colmap

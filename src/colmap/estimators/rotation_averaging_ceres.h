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
  double loss_function_scale = DegToRad(5.0);
  RotationAveragingReweighting reweighting =
      RotationAveragingReweighting::UNIFORM;

  ceres::Solver::Options solver_options;

  // Flag to skip maximum spanning tree initialization.
  bool skip_initialization = false;

  // When false, treat each non-ref sensor's cam_from_rig rotation as a
  // pre-calibrated constant
  bool refine_sensor_from_rig = true;

  CeresRotationAveragerOptions();
};

// Optimizes rotations directly in the reconstruction, which must outlive
// this object.
class CeresRotationAverager {
 public:
  CeresRotationAverager(const CeresRotationAveragerOptions& options,
                        const PoseGraph& pose_graph,
                        Reconstruction& reconstruction);
  CeresRotationAverager(const CeresRotationAverager&) = delete;
  CeresRotationAverager& operator=(const CeresRotationAverager&) = delete;

  ceres::Solver::Summary Solve();
  ceres::Problem& Problem();
  const ceres::Problem& Problem() const;
  const ceres::Solver::Options& SolverOptions() const;
  // Frame and sensor rotations for both images must be configured in Problem().
  void AddRelativeRotationResidual(
      image_t image_id1,
      image_t image_id2,
      const Eigen::Quaterniond& cam2_from_cam1,
      std::shared_ptr<ceres::LossFunction> loss_function);

 private:
  ceres::Solver::Options solver_options_;
  Reconstruction& reconstruction_;
  FlatHashMap<ceres::LossFunction*, std::shared_ptr<ceres::LossFunction>>
      losses_;
  // Keep losses alive until the problem is destroyed.
  std::unique_ptr<ceres::Problem> problem_;
};

// Calibrated sensor rotations are fixed by default and can be made variable
// through Problem().
std::unique_ptr<CeresRotationAverager> CreateDefaultCeresRotationAverager(
    const CeresRotationAveragerOptions& options,
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction);

}  // namespace colmap

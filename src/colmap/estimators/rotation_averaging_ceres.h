#pragma once

#include "colmap/estimators/ceres_loss.h"
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

  // Ceres-Solver options.
  ceres::Solver::Options solver_options;

  // Match RotationEstimatorOptions initialization behavior.
  bool skip_initialization = false;

  CeresRotationAveragerOptions();
};

// Owns exactly one Ceres problem. Parameter blocks in a default problem update
// Reconstruction rotations directly, so the reconstruction must outlive the
// owner.
class CeresRotationAverager {
 public:
  CeresRotationAverager(const CeresRotationAverager&) = delete;
  CeresRotationAverager& operator=(const CeresRotationAverager&) = delete;

  ceres::Solver::Summary Solve();
  ceres::Problem& Problem();
  const ceres::Problem& Problem() const;
  const ceres::Solver::Options& SolverOptions() const;
  void AddRelativeRotationResidual(
      image_t image_id1,
      image_t image_id2,
      const Eigen::Quaterniond& cam2_from_cam1,
      std::shared_ptr<ceres::LossFunction> loss_function);

 private:
  CeresRotationAverager(std::unique_ptr<ceres::Problem> problem,
                        ceres::Solver::Options solver_options,
                        Reconstruction& reconstruction);

  ceres::Solver::Options solver_options_;
  Reconstruction& reconstruction_;
  FlatHashMap<ceres::LossFunction*, std::shared_ptr<ceres::LossFunction>>
      losses_;
  // Destroy the problem before the retained DO_NOT_TAKE_OWNERSHIP losses.
  std::unique_ptr<ceres::Problem> problem_;

  friend std::unique_ptr<CeresRotationAverager>
  CreateDefaultCeresRotationAverager(const CeresRotationAveragerOptions&,
                                     const PoseGraph&,
                                     Reconstruction&);
};

std::unique_ptr<CeresRotationAverager> CreateDefaultCeresRotationAverager(
    const CeresRotationAveragerOptions& options,
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction);

}  // namespace colmap

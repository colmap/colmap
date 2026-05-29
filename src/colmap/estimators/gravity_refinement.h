#pragma once

#include "colmap/geometry/pose_prior.h"
#include "colmap/math/math.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"

#include <ceres/ceres.h>

namespace colmap {

struct GravityRefinerOptions {
  // The minimal ratio that the gravity vector should be consistent with
  double max_outlier_ratio = 0.5;
  // The maximum allowed angle error in degree
  double max_gravity_error = 1.;
  // Only refine the gravity of the images with more than min_neighbors
  int min_num_neighbors = 7;

  // The options for the solver
  ceres::Solver::Options solver_options;

  GravityRefinerOptions() {
    solver_options.num_threads = -1;
    solver_options.max_num_iterations = 100;
    solver_options.function_tolerance = 1e-5;
  }

  std::shared_ptr<ceres::LossFunction> CreateLossFunction() {
    return std::make_shared<ceres::ArctanLoss>(
        1 - std::cos(DegToRad(max_gravity_error)));
  }
};

class GravityRefiner {
 public:
  explicit GravityRefiner(const GravityRefinerOptions& options)
      : options_(options) {}

  void RefineGravity(const PoseGraph& pose_graph,
                     const Reconstruction& reconstruction,
                     std::vector<PosePrior>& pose_priors);

 private:
  void IdentifyErrorProneGravity(
      const PoseGraph& pose_graph,
      const Reconstruction& reconstruction,
      std::unordered_map<image_t, PosePrior*>& image_to_pose_prior,
      std::unordered_set<frame_t>& error_prone_frames);
  GravityRefinerOptions options_;
  std::shared_ptr<ceres::LossFunction> loss_function_;
};

// Refine gravity stored in pose priors using relative rotations from the pose
// graph.
void RunGravityRefinement(const GravityRefinerOptions& options,
                          const PoseGraph& pose_graph,
                          const Reconstruction& reconstruction,
                          std::vector<PosePrior>& pose_priors);

}  // namespace colmap

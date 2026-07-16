#pragma once

#include "colmap/geometry/pose_prior.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/enum_utils.h"
#include "colmap/util/hash_containers.h"

#include <string>
#include <vector>

#include <ceres/ceres.h>

namespace colmap {

// off: no pose priors are used.
// initialize: seed covered rig-frame centers from pose priors; ordinary
//   post-positioning normalization still runs, so this is not a metric
//   gauge, only a warm start.
// optimize: robustly constrain the solve with pose-prior residuals; skips
//   the gauge-destroying normalization only if it actually engaged.
// Enumerator names are lowercase so MAKE_ENUM_CLASS's generated ToString/
// FromString round-trip the exact CLI/project-file spelling required by the
// contract, without a separate case-conversion step.
MAKE_ENUM_CLASS(PosePriorPositionMode, 0, off, initialize, optimize);

struct GlobalPositionerOptions {
  // Whether to initialize the camera and track positions randomly.
  bool generate_random_positions = true;
  bool generate_random_points = true;
  // Whether to initialize the camera scales to a constant 1 or derive them from
  // the initialized camera and point positions.
  bool generate_scales = true;

  // Flags for which parameters to optimize
  bool optimize_positions = true;
  bool optimize_points = true;
  bool optimize_scales = true;

  // When false, treat sensor_from_rig as a fixed (pre-calibrated) parameter.
  bool refine_sensor_from_rig = true;

  bool use_gpu = true;
  std::string gpu_index = "-1";
  int min_num_images_gpu_solver = 50;

  // Constrain the minimum number of views per track
  int min_num_view_per_track = 3;

  // PRNG seed for random initialization.
  // If -1 (default), uses non-deterministic random_device seeding.
  // If >= 0, uses deterministic seeding with the given value.
  int random_seed = -1;

  // Scaling factor for the loss function
  double loss_function_scale = 0.1;

  // Whether/how to use pose priors (e.g. GPS) to seed or constrain frame
  // positions. See PosePriorPositionMode.
  PosePriorPositionMode pose_prior_position_mode = PosePriorPositionMode::off;
  // Robust loss scale for whitened pose-prior position residuals in
  // `optimize` mode: sqrt(7.815), the 95% chi-square threshold for 3 DoF.
  double pose_prior_position_loss_scale = 2.7955;
  // Fallback position standard deviation (metres) used only when a prior has
  // no position covariance.
  double pose_prior_position_fallback_stddev = 1.0;

  // Whether to use custom parameter block ordering for Schur-based solvers.
  // Disable for deterministic behavior when using a fixed random seed.
  bool use_parameter_block_ordering = true;

  // The options for the solver
  ceres::Solver::Options solver_options;

  GlobalPositionerOptions() {
    solver_options.num_threads = -1;
    solver_options.max_num_iterations = 100;
    solver_options.function_tolerance = 1e-5;
  }

  std::shared_ptr<ceres::LossFunction> CreateLossFunction() {
    return std::make_shared<ceres::HuberLoss>(loss_function_scale);
  }
};

// Summary of pose-prior engagement during global positioning. Populated
// regardless of mode so callers can log/report requested-vs-engaged status.
struct PosePriorPositionSummary {
  PosePriorPositionMode requested = PosePriorPositionMode::off;
  bool engaged = false;
  int num_usable_priors = 0;
  int num_covered_frames = 0;
  int num_fallback_frames = 0;
};

class GlobalPositioner {
 public:
  explicit GlobalPositioner(const GlobalPositionerOptions& options);

  // Returns true if the optimization was a success, false if there was a
  // failure.
  // Assume tracks here are already filtered. `pose_priors` is only consulted
  // when options.pose_prior_position_mode != OFF; if provided, `summary` is
  // filled with requested/engaged status.
  bool Solve(const PoseGraph& pose_graph,
            Reconstruction& reconstruction,
            const std::vector<PosePrior>& pose_priors = {},
            PosePriorPositionSummary* summary = nullptr);

  GlobalPositionerOptions& GetOptions() { return options_; }

 protected:
  void SetupProblem(const PoseGraph& pose_graph,
                    const Reconstruction& reconstruction);

  // Initialize all cameras to be random, optionally seeding covered rig-frame
  // centers from pose priors when options.pose_prior_position_mode ==
  // INITIALIZE.
  void InitializeRandomPositions(const PoseGraph& pose_graph,
                                 Reconstruction& reconstruction,
                                 const std::vector<PosePrior>& pose_priors,
                                 PosePriorPositionSummary* summary);

  // Add tracks to the problem
  void AddPointToCameraConstraints(Reconstruction& reconstruction);

  // Add a single point3D to the problem
  void AddPoint3DToProblem(point3D_t point3D_id,
                           Reconstruction& reconstruction);

  // Set the parameter groups
  void AddCamerasAndPointsToParameterGroups(Reconstruction& reconstruction);

  // Parameterize the variables, set some variables to be constant if desired
  void ParameterizeVariables(Reconstruction& reconstruction);

  // During the optimization, the camera translation is set to be the camera
  // center Convert the results back to camera poses
  void ConvertBackResults(Reconstruction& reconstruction);

  GlobalPositionerOptions options_;

  std::unique_ptr<ceres::Problem> problem_;

  // Loss functions for reweighted terms.
  std::shared_ptr<ceres::LossFunction> loss_function_;
  std::shared_ptr<ceres::LossFunction> loss_function_ptcam_uncalibrated_;
  std::shared_ptr<ceres::LossFunction> loss_function_ptcam_calibrated_;

  // Auxiliary scale variables.
  std::vector<double> scales_;

  // Temporary storage for frame centers (world coordinates) during
  // optimization. This allows keeping RigFromWorld().translation() in
  // cam_from_world convention.
  NodeHashMap<frame_t, Eigen::Vector3d> frame_centers_;

  // Temporary storage for camera-in-rig positions when cam_from_rig is unknown
  // and needs to be estimated.
  NodeHashMap<sensor_t, Eigen::Vector3d> cams_in_rig_;
};

// Solve global positioning using point-to-camera constraints.
bool RunGlobalPositioning(const GlobalPositionerOptions& options,
                          const PoseGraph& pose_graph,
                          Reconstruction& reconstruction,
                          const std::vector<PosePrior>& pose_priors = {},
                          PosePriorPositionSummary* summary = nullptr);

}  // namespace colmap

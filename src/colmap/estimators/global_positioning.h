#pragma once

#include "colmap/geometry/pose_prior.h"
#include "colmap/geometry/sim3.h"
#include "colmap/math/math.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"

#include <string>
#include <vector>

#include <ceres/ceres.h>

namespace colmap {

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

  // Constrain frame positions toward pose priors through a jointly optimized
  // similarity gauge that maps the global positioning frame to the prior frame.
  bool use_prior_position = false;
  bool use_robust_loss_on_prior_position = true;
  double prior_position_loss_scale = std::sqrt(kChiSquare95ThreeDof);
  double prior_position_fallback_stddev = 1.0;

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

class GlobalPositioner {
 public:
  explicit GlobalPositioner(const GlobalPositionerOptions& options);

  // Returns true if the optimization was a success, false if there was a
  // failure.
  // Assume tracks here are already filtered
  bool Solve(const PoseGraph& pose_graph,
             Reconstruction& reconstruction,
             const std::vector<PosePrior>& pose_priors = {});

  GlobalPositionerOptions& GetOptions() { return options_; }

 protected:
  void SetupProblem(const PoseGraph& pose_graph,
                    const Reconstruction& reconstruction);

  // Initialize all cameras to be random.
  void InitializeRandomPositions(const PoseGraph& pose_graph,
                                 Reconstruction& reconstruction);

  // Add tracks to the problem
  void AddPointToCameraConstraints(Reconstruction& reconstruction);

  // Add a single point3D to the problem
  void AddPoint3DToProblem(point3D_t point3D_id,
                           Reconstruction& reconstruction);

  // Set the parameter groups
  void AddCamerasAndPointsToParameterGroups(Reconstruction& reconstruction);

  // Estimate the similarity gauge mapping the current frame centers to the
  // prior frame. Returns false if there are not enough valid priors.
  bool InitializeGauge(const Reconstruction& reconstruction);

  // Add prior-position residuals on frame centers through the gauge.
  void AddPosePriorConstraints(const Reconstruction& reconstruction);

  // RMSE of frame centers mapped through the gauge against their priors.
  double PriorPositionRMSE(const Reconstruction& reconstruction) const;

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
  std::shared_ptr<ceres::LossFunction> prior_loss_function_;

  // Camera position priors and the gauge mapping the global positioning frame
  // to the prior frame, jointly optimized once the gauge is initialized.
  std::vector<PosePrior> pose_priors_;
  Sim3d prior_from_gp_;
  bool gauge_initialized_ = false;
  int frame_param_group_ = -1;

  // Auxiliary scale variables.
  std::vector<double> scales_;

  // Temporary storage for frame centers (world coordinates) during
  // optimization. This allows keeping RigFromWorld().translation() in
  // cam_from_world convention.
  std::unordered_map<frame_t, Eigen::Vector3d> frame_centers_;

  // Temporary storage for camera-in-rig positions when cam_from_rig is unknown
  // and needs to be estimated.
  std::unordered_map<sensor_t, Eigen::Vector3d> cams_in_rig_;
};

// Solve global positioning using point-to-camera constraints.
bool RunGlobalPositioning(const GlobalPositionerOptions& options,
                          const PoseGraph& pose_graph,
                          Reconstruction& reconstruction,
                          const std::vector<PosePrior>& pose_priors = {});

}  // namespace colmap

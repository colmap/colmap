#pragma once

#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/track.h"
#include "colmap/util/hash_containers.h"

#include <memory>
#include <string>
#include <vector>

#include <ceres/ceres.h>

namespace colmap {

// Per-observation covariance matrices in world coordinates for default BATA
// residuals.
using ObservationCovarianceMap =
    FlatHashMap<Point3DTrackElementKey, Eigen::Matrix3d, PairHash>;

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
  // Whether to fix the first active observation scale to remove the gauge
  // ambiguity when optimizing scales.
  bool fix_observation_scale_gauge = true;
  // Whether to down-weight observations from cameras without a focal-length
  // prior, preserving the stock global-positioning objective by default.
  bool downweight_uncalibrated_observations = true;

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
  // Solve the prepared problem and publish its results.
  ceres::Solver::Summary Solve();

  ceres::Problem& Problem();
  const ceres::Solver::Options& SolverOptions() const;
  // Returns the temporary frame-center block used by the prepared problem,
  // allowing additional residuals to constrain the same variable. Returns
  // nullptr if the frame is not active.
  double* FrameCenterParameterBlock(frame_t frame_id);
  bool Finalize(const ceres::Solver::Summary& summary);
  // Rebuilds the solver ordering after extending Problem(), ensuring parameter
  // blocks introduced after construction are included before solving.
  void SetParameterBlockOrdering();

 protected:
  explicit GlobalPositioner(const GlobalPositionerOptions& options);

  // Construct the problem without solving it.
  void Prepare(const PoseGraph& pose_graph,
               Reconstruction& reconstruction,
               const ObservationCovarianceMap& observation_covariances,
               std::shared_ptr<ceres::LossFunction> loss_function);

  void SetupProblem(std::shared_ptr<ceres::LossFunction> loss_function);

  // Initialize all cameras to be random.
  void InitializeRandomPositions(const PoseGraph& pose_graph,
                                 Reconstruction& reconstruction);

  // Add regular constraints with optional keyed covariances.
  void AddPointToCameraConstraints(
      Reconstruction& reconstruction,
      const ObservationCovarianceMap& observation_covariances);

  // Add a single point3D to the problem.
  void AddPoint3DToProblem(
      point3D_t point3D_id,
      Reconstruction& reconstruction,
      const ObservationCovarianceMap& observation_covariances);

  // Parameterize the variables, set some variables to be constant if desired
  void ParameterizeVariables(Reconstruction& reconstruction);

  // During the optimization, the camera translation is set to be the camera
  // center Convert the results back to camera poses
  void ConvertBackResults(Reconstruction& reconstruction);

  GlobalPositionerOptions options_;

 private:
  std::vector<double> scales_;

 protected:
  // Loss functions for reweighted terms.
  std::shared_ptr<ceres::LossFunction> loss_function_;
  std::shared_ptr<ceres::LossFunction> loss_function_ptcam_uncalibrated_;
  std::shared_ptr<ceres::LossFunction> loss_function_ptcam_calibrated_;

  // Temporary storage for frame centers (world coordinates) during
  // optimization. This allows keeping RigFromWorld().translation() in
  // cam_from_world convention.
  NodeHashMap<frame_t, Eigen::Vector3d> frame_centers_;

  // Temporary storage for camera-in-rig positions when cam_from_rig is unknown
  // and needs to be estimated.
  NodeHashMap<sensor_t, Eigen::Vector3d> cams_in_rig_;

  // Retained for late parameter ordering and Finalize().
  Reconstruction* reconstruction_ = nullptr;
  std::unique_ptr<ceres::Problem> problem_;

 private:
  friend std::unique_ptr<GlobalPositioner> CreateDefaultGlobalPositioner(
      const GlobalPositionerOptions&,
      const PoseGraph&,
      Reconstruction*,
      const ObservationCovarianceMap&,
      std::shared_ptr<ceres::LossFunction>);
};

// The reconstruction must outlive the returned positioner.
std::unique_ptr<GlobalPositioner> CreateDefaultGlobalPositioner(
    const GlobalPositionerOptions& options,
    const PoseGraph& pose_graph,
    Reconstruction* reconstruction,
    const ObservationCovarianceMap& observation_covariances = {},
    std::shared_ptr<ceres::LossFunction> loss_function = nullptr);

// Solve global positioning using point-to-camera constraints.
bool RunGlobalPositioning(const GlobalPositionerOptions& options,
                          const PoseGraph& pose_graph,
                          Reconstruction& reconstruction);

}  // namespace colmap

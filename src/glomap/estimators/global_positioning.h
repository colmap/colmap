#pragma once

#include "colmap/scene/frame.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/rig.h"
#include "colmap/sensor/models.h"

#include "glomap/scene/view_graph.h"

#include <string>

#include <ceres/ceres.h>

namespace glomap {

struct GlobalPositionerOptions {
  // ONLY_POINTS is recommended
  enum ConstraintType {
    // only include camera to point constraints
    ONLY_POINTS,
    // only include camera to camera constraints
    ONLY_CAMERAS,
    // the points and cameras are reweighted to have similar total contribution
    POINTS_AND_CAMERAS_BALANCED,
    // treat each contribution from camera to point and camera to camera equally
    POINTS_AND_CAMERAS,
  };

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

  bool use_gpu = true;
  std::string gpu_index = "-1";
  int min_num_images_gpu_solver = 50;

  // Constrain the minimum number of views per track
  int min_num_view_per_track = 3;

  // PRNG seed for random initialization.
  // If -1 (default), uses non-deterministic random_device seeding.
  // If >= 0, uses deterministic seeding with the given value.
  int random_seed = -1;

  // the type of global positioning
  ConstraintType constraint_type = ONLY_POINTS;
  double constraint_reweight_scale =
      1.0;  // only relevant for POINTS_AND_CAMERAS_BALANCED

  // Scaling factor for the loss function
  double loss_function_scale = 0.1;

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
  bool Solve(const ViewGraph& view_graph,
             colmap::Reconstruction& reconstruction);

  GlobalPositionerOptions& GetOptions() { return options_; }

 protected:
  void SetupProblem(const ViewGraph& view_graph,
                    const colmap::Reconstruction& reconstruction);

  // Initialize all cameras to be random.
  void InitializeRandomPositions(const ViewGraph& view_graph,
                                 colmap::Reconstruction& reconstruction);

  // Creates camera to camera constraints from relative translations. (3D)
  void AddCameraToCameraConstraints(const ViewGraph& view_graph,
                                    colmap::Reconstruction& reconstruction);

  // Add tracks to the problem
  void AddPointToCameraConstraints(colmap::Reconstruction& reconstruction);

  // Add a single point3D to the problem
  void AddPoint3DToProblem(point3D_t point3D_id,
                           colmap::Reconstruction& reconstruction);

  // Set the parameter groups
  void AddCamerasAndPointsToParameterGroups(
      colmap::Reconstruction& reconstruction);

  // Parameterize the variables, set some variables to be constant if desired
  void ParameterizeVariables(colmap::Reconstruction& reconstruction);

  // During the optimization, the camera translation is set to be the camera
  // center Convert the results back to camera poses
  void ConvertBackResults(colmap::Reconstruction& reconstruction);

  GlobalPositionerOptions options_;

  std::unique_ptr<ceres::Problem> problem_;

  // Loss functions for reweighted terms.
  std::shared_ptr<ceres::LossFunction> loss_function_;
  std::shared_ptr<ceres::LossFunction> loss_function_ptcam_uncalibrated_;
  std::shared_ptr<ceres::LossFunction> loss_function_ptcam_calibrated_;

  // Auxiliary scale variables.
  std::vector<double> scales_;

  std::unordered_map<rig_t, double> rig_scales_;

  // Temporary storage for frame centers (world coordinates) during
  // optimization. This allows keeping RigFromWorld().translation in
  // cam_from_world convention.
  std::unordered_map<frame_t, Eigen::Vector3d> frame_centers_;

  // Temporary storage for camera-in-rig positions when cam_from_rig is unknown
  // and needs to be estimated.
  std::unordered_map<sensor_t, Eigen::Vector3d> cams_in_rig_;
};

}  // namespace glomap

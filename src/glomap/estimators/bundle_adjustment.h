#pragma once

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/scene/reconstruction.h"

#include <string>
#include <thread>

#include <ceres/ceres.h>

namespace glomap {

struct BundleAdjusterOptions {
  // Flags for which parameters to optimize
  bool optimize_rig_poses = false;  // Whether to optimize the rig poses
  bool optimize_rotations = true;
  bool optimize_translation = true;
  bool optimize_intrinsics = true;
  bool optimize_principal_point = false;
  bool optimize_points = true;

  bool use_gpu = true;
  std::string gpu_index = "-1";
  int min_num_images_gpu_solver = 50;

  // Constrain the minimum number of views per track
  int min_num_view_per_track = 3;

  // Scaling factor for the loss function
  double loss_function_scale = 1.0;

  // The options for the solver
  ceres::Solver::Options solver_options;

  BundleAdjusterOptions() {
    solver_options.num_threads = std::thread::hardware_concurrency();
    solver_options.max_num_iterations = 200;
    solver_options.minimizer_progress_to_stdout = false;
    solver_options.function_tolerance = 1e-5;
  }

  // Convert to colmap BundleAdjustmentOptions
  colmap::BundleAdjustmentOptions ToColmapOptions() const;
};

// Run bundle adjustment using colmap's implementation.
// constant_rotation: if true, keep rotation constant (only optimize
// translation)
bool RunBundleAdjustment(const BundleAdjusterOptions& options,
                         bool constant_rotation,
                         colmap::Reconstruction& reconstruction);

}  // namespace glomap

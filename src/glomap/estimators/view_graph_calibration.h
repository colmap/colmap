#pragma once

#include "colmap/scene/reconstruction.h"

#include "glomap/scene/view_graph.h"

#include <memory>

#include <ceres/ceres.h>

namespace glomap {

struct ViewGraphCalibratorOptions {
  // Random seed for RANSAC-based estimation (-1 for random).
  int random_seed = -1;

  // Whether to cross-validate prior focal lengths by checking the ratio of
  // calibrated vs uncalibrated pairs per camera. When enabled, UNCALIBRATED
  // pairs are converted to CALIBRATED if both cameras have reliable priors.
  bool cross_validate_prior_focal_lengths = true;

  // Whether to re-estimate relative poses after focal length calibration.
  bool reestimate_relative_pose = true;

  // The minimum ratio of the estimated focal length to the prior focal length.
  double min_focal_length_ratio = 0.1;
  // The maximum ratio of the estimated focal length to the prior focal length.
  double max_focal_length_ratio = 10;

  // The maximum calibration error for an image pair.
  double max_calibration_error = 2.;

  // Scaling factor for the loss function
  double loss_function_scale = 0.01;

  // The options for the solver
  ceres::Solver::Options solver_options;

  // Options for relative pose re-estimation.
  double relpose_max_error = 1.0;
  int relpose_min_num_inliers = 30;
  double relpose_min_inlier_ratio = 0.25;

  ViewGraphCalibratorOptions() {
    solver_options.num_threads = -1;
    solver_options.max_num_iterations = 100;
    solver_options.function_tolerance = 1e-5;
  }

  // Create loss function for given options.
  std::unique_ptr<ceres::LossFunction> CreateLossFunction() const {
    return std::make_unique<ceres::CauchyLoss>(loss_function_scale);
  }
};

// Calibrate the view graph by estimating focal lengths from fundamental
// matrices. Filters image pairs with high calibration errors.
// Then re-estimates relative poses using the calibrated cameras.
bool CalibrateViewGraph(const ViewGraphCalibratorOptions& options,
                        ViewGraph& view_graph,
                        colmap::Reconstruction& reconstruction);

}  // namespace glomap

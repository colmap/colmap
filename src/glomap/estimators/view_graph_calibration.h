#pragma once

#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/models.h"

#include "glomap/estimators/relpose_estimation.h"
#include "glomap/processors/image_pair_inliers.h"
#include "glomap/scene/view_graph.h"

#include <memory>

#include <ceres/ceres.h>

namespace glomap {

struct ViewGraphCalibratorOptions {
  // Whether to cross-validate prior focal lengths by checking the ratio of
  // calibrated vs uncalibrated pairs per camera. When enabled, UNCALIBRATED
  // pairs are converted to CALIBRATED if both cameras have reliable priors.
  bool cross_validate_prior_focal_lengths = true;

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

class ViewGraphCalibrator {
 public:
  explicit ViewGraphCalibrator(const ViewGraphCalibratorOptions& options)
      : options_(options) {}

  // Entry point for the calibration
  bool Solve(ViewGraph& view_graph, colmap::Reconstruction& reconstruction);

 private:
  // Initialize focal lengths from reconstruction
  void InitializeFocalsFromReconstruction(
      const colmap::Reconstruction& reconstruction);

  // Add the image pairs to the problem
  void AddImagePairsToProblem(const ViewGraph& view_graph,
                              const colmap::Reconstruction& reconstruction);

  // Set the cameras to be constant if they have prior intrinsics
  size_t ParameterizeCameras(const colmap::Reconstruction& reconstruction);

  // Convert the results back to the camera
  void ConvertBackResults(colmap::Reconstruction& reconstruction);

  // Evaluate and filter the image pairs based on the calibration results
  size_t EvaluateAndFilterImagePairs(ViewGraph& view_graph) const;

  ViewGraphCalibratorOptions options_;
  std::unique_ptr<ceres::Problem> problem_;
  std::unique_ptr<ceres::LossFunction> loss_function_;
  std::unordered_map<camera_t, double> focals_;
};

// Calibrate the view graph by estimating focal lengths from fundamental
// matrices. Filters image pairs with high calibration errors.
// Then re-estimates relative poses using the calibrated cameras.
bool CalibrateViewGraph(const ViewGraphCalibratorOptions& options,
                        const RelativePoseEstimationOptions& relpose_options,
                        const InlierThresholdOptions& inlier_thresholds,
                        ViewGraph& view_graph,
                        colmap::Reconstruction& reconstruction);

}  // namespace glomap

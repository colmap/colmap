#pragma once

#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/models.h"

#include "glomap/scene/view_graph.h"

#include <memory>

#include <ceres/ceres.h>

namespace glomap {

struct ViewGraphCalibratorOptions {
  // The minimal ratio of the estimated focal length against the prior focal
  // length
  double thres_lower_ratio = 0.1;
  // The maximal ratio of the estimated focal length against the prior focal
  // length
  double thres_higher_ratio = 10;

  // The threshold for the corresponding error in the problem for an image pair
  double thres_two_view_error = 2.;

  // Scaling factor for the loss function
  double loss_function_scale = 0.01;

  // The options for the solver
  ceres::Solver::Options solver_options;

  ViewGraphCalibratorOptions() {
    solver_options.num_threads = -1;
    solver_options.max_num_iterations = 100;
    solver_options.function_tolerance = 1e-5;
  }

  std::shared_ptr<ceres::LossFunction> CreateLossFunction() {
    return std::make_shared<ceres::CauchyLoss>(loss_function_scale);
  }
};

class ViewGraphCalibrator {
 public:
  explicit ViewGraphCalibrator(const ViewGraphCalibratorOptions& options)
      : options_(options) {}

  // Entry point for the calibration
  bool Solve(ViewGraph& view_graph, colmap::Reconstruction& reconstruction);

 private:
  // Reset the problem
  void Reset(const colmap::Reconstruction& reconstruction);

  // Add the image pairs to the problem
  void AddImagePairsToProblem(const ViewGraph& view_graph,
                              const colmap::Reconstruction& reconstruction);

  // Add a single image pair to the problem
  void AddImagePair(image_t image_id1,
                    image_t image_id2,
                    const ImagePair& image_pair,
                    const colmap::Reconstruction& reconstruction);

  // Set the cameras to be constant if they have prior intrinsics
  size_t ParameterizeCameras(const colmap::Reconstruction& reconstruction);

  // Convert the results back to the camera
  void CopyBackResults(colmap::Reconstruction& reconstruction);

  // Filter the image pairs based on the calibration results
  size_t FilterImagePairs(ViewGraph& view_graph) const;

  ViewGraphCalibratorOptions options_;
  std::unique_ptr<ceres::Problem> problem_;
  std::unordered_map<camera_t, double> focals_;
  std::shared_ptr<ceres::LossFunction> loss_function_;
};

}  // namespace glomap

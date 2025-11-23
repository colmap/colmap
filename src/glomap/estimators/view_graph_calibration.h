#pragma once

#include "glomap/estimators/optimization_base.h"
#include "glomap/scene/types_sfm.h"

#include <memory>

namespace glomap {

struct ViewGraphCalibratorOptions : public OptimizationBaseOptions {
  // The minimal ratio of the estimated focal length against the prior focal
  // length
  double thres_lower_ratio = 0.1;
  // The maximal ratio of the estimated focal length against the prior focal
  // length
  double thres_higher_ratio = 10;

  // The threshold for the corresponding error in the problem for an image pair
  double thres_two_view_error = 2.;

  ViewGraphCalibratorOptions() : OptimizationBaseOptions() {
    thres_loss_function = 1e-2;
  }

  std::shared_ptr<ceres::LossFunction> CreateLossFunction() {
    return std::make_shared<ceres::CauchyLoss>(thres_loss_function);
  }
};

class ViewGraphCalibrator {
 public:
  ViewGraphCalibrator(const ViewGraphCalibratorOptions& options)
      : options_(options) {}

  // Entry point for the calibration
  bool Solve(ViewGraph& view_graph,
             std::unordered_map<camera_t, Camera>& cameras,
             std::unordered_map<image_t, Image>& images);

 private:
  // Reset the problem
  void Reset(const std::unordered_map<camera_t, Camera>& cameras);

  // Add the image pairs to the problem
  void AddImagePairsToProblem(
      const ViewGraph& view_graph,
      const std::unordered_map<camera_t, Camera>& cameras,
      const std::unordered_map<image_t, Image>& images);

  // Add a single image pair to the problem
  void AddImagePair(const ImagePair& image_pair,
                    const std::unordered_map<camera_t, Camera>& cameras,
                    const std::unordered_map<image_t, Image>& images);

  // Set the cameras to be constant if they have prior intrinsics
  size_t ParameterizeCameras(
      const std::unordered_map<camera_t, Camera>& cameras);

  // Convert the results back to the camera
  void CopyBackResults(std::unordered_map<camera_t, Camera>& cameras);

  // Filter the image pairs based on the calibration results
  size_t FilterImagePairs(ViewGraph& view_graph) const;

  ViewGraphCalibratorOptions options_;
  std::unique_ptr<ceres::Problem> problem_;
  std::unordered_map<camera_t, double> focals_;
  std::shared_ptr<ceres::LossFunction> loss_function_;
};

}  // namespace glomap

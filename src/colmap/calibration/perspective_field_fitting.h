// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/scene/camera.h"
#include "colmap/sensor/models.h"

#include <limits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colmap {

// Dense or subsampled perspective field (2D up-vectors and latitude angles)
// with per-pixel confidences for a single image.
struct PerspectiveField {
  // Dimensions of the perspective field grid (0 if unstructured/subsampled).
  int width = 0;
  int height = 0;

  // 2D pixel coordinates in the original image: shape (N, 2).
  Eigen::MatrixX2d points2D_in_img;
  // Unit 2D up-vectors in the original image: shape (N, 2).
  Eigen::MatrixX2d up_in_img;
  // Latitude angles in radians in [-pi/2, pi/2]: shape (N,).
  Eigen::VectorXd latitude;
  // Confidence weights in [0, 1] for the 2D up-vectors: shape (N,).
  Eigen::VectorXd up_confidence;
  // Confidence weights in [0, 1] for the latitude angles: shape (N,).
  Eigen::VectorXd latitude_confidence;

  size_t NumPoints() const { return points2D_in_img.rows(); }
  bool Check() const;
};

struct PerspectiveFieldCameraInput {
  const PerspectiveField* field = nullptr;
  Camera* camera = nullptr;
  // Rotation from the rig coordinate frame to this camera's sensor frame.
  // Identity for single-camera frames or the reference sensor of a rig.
  Eigen::Quaterniond cam_from_rig = Eigen::Quaterniond::Identity();
  // Whether to refine this camera's intrinsics during optimization.
  bool refine_camera = false;
};

struct PerspectiveFieldFittingOptions {
  // Maximum number of Ceres solver iterations for nonlinear refinement.
  int max_num_iterations = 100;

  // Robust Huber loss scale for 2D up-vector and sin(latitude) residuals.
  // Set to <= 0 to use standard L2 loss. Matches GeoCalib's default (0.01).
  double loss_function_scale = 0.01;

  // Spatial window size (stride x stride) for subsampling rays from a dense
  // perspective field. Set to 1 to use all rays.
  int stride = 4;

  // If true, select the ray with maximum combined confidence within each
  // stride x stride spatial window. If false, use uniform striding.
  bool max_pool_confidence = true;

  // Minimum combined confidence sqrt(up_confidence * latitude_confidence) to
  // include a ray.
  double min_confidence = 0.0;

  // Whether to weight residuals by their predicted confidences.
  bool use_confidence = true;

  // Which camera parameter groups to refine when refine_camera is true.
  bool refine_focal_length = true;
  bool refine_principal_point = false;
  bool refine_extra_params = true;

  // Whether to print the Ceres solver summary.
  bool print_summary = false;

  bool Check() const;
};

struct FittedPerspectiveFields {
  // Estimated unit gravity vector (down direction in sensor/rig coordinates,
  // following COLMAP's PosePrior::gravity convention) in the rig frame.
  Eigen::Vector3d gravity_in_rig = Eigen::Vector3d::Zero();
  bool success = false;
  double initial_cost = std::numeric_limits<double>::infinity();
  double final_cost = std::numeric_limits<double>::infinity();
};

// Subsample a perspective field using spatial window max-pooling (when
// max_pool_confidence is true and the field has valid grid dimensions) or
// uniform striding, filtering out rays below min_confidence.
PerspectiveField SubsamplePerspectiveField(const PerspectiveField& field,
                                           int stride,
                                           bool max_pool_confidence = true,
                                           double min_confidence = 0.0);

// Estimate a single unit gravity vector in the rig frame (and optionally
// refine camera intrinsics for inputs with refine_camera == true) from one or
// more perspective fields belonging to the same frame/rig.
FittedPerspectiveFields FitPerspectiveFields(
    const PerspectiveFieldFittingOptions& options,
    const std::vector<PerspectiveFieldCameraInput>& inputs);

// Convenience overload for a single image/camera.
FittedPerspectiveFields FitPerspectiveField(
    const PerspectiveFieldFittingOptions& options,
    const PerspectiveField& field,
    Camera* camera,
    bool refine_camera);

// Synthesize a noiseless perspective field on a pixel grid for a given camera
// and sensor-frame gravity vector (down direction, COLMAP convention). Useful
// for testing and evaluation.
PerspectiveField ComputePerspectiveFieldFromCameraAndGravity(
    const Camera& camera,
    const Eigen::Vector3d& gravity_in_cam,
    int grid_width = 0,
    int grid_height = 0);

}  // namespace colmap

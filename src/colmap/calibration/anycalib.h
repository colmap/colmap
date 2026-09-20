// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/calibration/ray_fitting.h"
#include "colmap/calibration/resources.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/sensor/bitmap.h"

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace colmap {

class CameraCalibrator;
struct CameraCalibrationOptions;

struct AnyCalibCalibrationOptions {
  // Path or download URI of the exported AnyCalib ONNX model.
  std::string model_path = kDefaultAnyCalibGenUri;

  // Fitting options shared with future ray-based backends.
  RayFittingOptions fitting;

  bool Check() const;
};

// Network input plus the transform mapping fitted intrinsics back to the
// original image, mirroring `AnyCalib.set_im_size` with target aspect ratio 1
// (upsample small images, center-crop to square, downsample to 322x322).
struct AnyCalibInput {
  // RGB in [0, 1], row-major [C, H, W].
  std::vector<float> data;
  // Per-axis scale and shift of the intrinsics digitizing transform.
  Eigen::Vector2d scale_xy = Eigen::Vector2d::Ones();
  Eigen::Vector2d shift_xy = Eigen::Vector2d::Zero();
  int upright_width = 0;
  int upright_height = 0;
  int image_rot90 = 0;

  // Map network-image coordinates and camera rays back to the original image
  // orientation and resolution.
  Eigen::Vector2d ImagePointToOriginal(const Eigen::Vector2d& point) const;
  Eigen::Vector3d CameraRayToOriginal(const Eigen::Vector3d& ray) const;
};

AnyCalibInput PrepareAnyCalibInput(const Bitmap& bitmap,
                                   const PosePrior& pose_prior = PosePrior());

std::unique_ptr<CameraCalibrator> CreateAnyCalibCalibrator(
    const CameraCalibrationOptions& options);

}  // namespace colmap

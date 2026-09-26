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

class MonocularCalibrator;
struct MonocularCalibrationOptions;

struct AnyCalibOptions {
  // Path or download URI of the exported AnyCalib ONNX model.
  std::string model_path = kDefaultAnyCalibGenUri;

  // Fitting options shared with future ray-based backends.
  RayFittingOptions fitting;

  bool Check() const;
};

// Network input plus the transform mapping fitted intrinsics back to the
// original image, mirroring `AnyCalib.set_im_size` with target aspect ratio 1
// (rotate upright if the pose prior provides gravity, upsample small images,
// center-crop to square, downsample to 322x322).
struct AnyCalibInput {
  // RGB in [0, 1], row-major [C, H, W].
  std::vector<float> data;
  // Per-axis scale and shift of the intrinsics digitizing transform.
  Eigen::Vector2d scale_xy = Eigen::Vector2d::Ones();
  Eigen::Vector2d shift_xy = Eigen::Vector2d::Zero();
  // Dimensions of the upright (rotation-corrected) image, for mapping network
  // coordinates back through the gravity rotation to the original image.
  int upright_width = 0;
  int upright_height = 0;
  // Counter-clockwise quarter turns applied to upright the image (0-3).
  int image_rot90 = 0;

  // Map network-image coordinates and camera rays back to the original image
  // orientation and resolution.
  Eigen::Vector2d ImgToOrig(const Eigen::Vector2d& point) const;
  Eigen::Vector3d CamToOrig(const Eigen::Vector3d& ray) const;
};

AnyCalibInput PrepareAnyCalibInput(const Bitmap& bitmap,
                                   const PosePrior& pose_prior = PosePrior());

std::unique_ptr<MonocularCalibrator> CreateAnyCalibCalibrator(
    const MonocularCalibrationOptions& options);

}  // namespace colmap

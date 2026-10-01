// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/calibration/perspective_field_fitting.h"
#include "colmap/calibration/resources.h"
#include "colmap/sensor/bitmap.h"

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace colmap {

struct GeoCalibOptions {
  // Path or download URI of the exported GeoCalib ONNX model.
  std::string model_path = kDefaultGeoCalibUri;

  // Target size of the shorter image edge before center-cropping to a multiple
  // of 32. Matches GeoCalib's default preprocessor (`resize = 320`).
  int image_size = 320;

  // If true, center-crop the resized image to a square (image_size x
  // image_size) instead of preserving the aspect ratio.
  bool force_square = false;

  // Number of CPU threads for ONNX inference (-1 uses all available cores).
  int num_threads = -1;

  // Whether to use GPU for ONNX inference.
  bool use_gpu = true;
  std::string gpu_index = "-1";

  // Options for Ceres perspective field fitting.
  PerspectiveFieldFittingOptions fitting;

  bool Check() const;
};

struct GeoCalibInput {
  // RGB in [0, 1], row-major [C, H, W].
  std::vector<float> data;
  int width = 0;
  int height = 0;
  // Per-axis scale and shift mapping original image coordinates to network
  // coordinates: p_net = p_orig.cwiseProduct(scale_xy) + shift_xy.
  Eigen::Vector2d scale_xy = Eigen::Vector2d::Ones();
  Eigen::Vector2d shift_xy = Eigen::Vector2d::Zero();

  Eigen::Vector2d ImgToOrig(const Eigen::Vector2d& point) const;
  Eigen::Vector2d UpToOrig(const Eigen::Vector2d& up) const;
};

GeoCalibInput PrepareGeoCalibInput(const Bitmap& bitmap,
                                   int image_size = 320,
                                   bool force_square = false);

class GeoCalib {
 public:
  virtual ~GeoCalib() = default;

  static std::unique_ptr<GeoCalib> Create(const GeoCalibOptions& options);

  // Predict the dense perspective field (mapped back to the original image's
  // pixel coordinate system) from a single image.
  virtual PerspectiveField PredictPerspectiveField(
      const Bitmap& bitmap) const = 0;

  // Convenience method to predict the perspective field for a single image and
  // fit sensor-frame gravity (and optionally camera intrinsics).
  virtual FittedPerspectiveFields Calibrate(
      const Bitmap& bitmap,
      Camera* camera,
      bool refine_camera = false) const = 0;
};

}  // namespace colmap

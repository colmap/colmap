// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/calibration/geocalib.h"
#include "colmap/util/threading.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace colmap {

struct GravityEstimationOptions {
  // Target COLMAP camera model for the fitted intrinsics. Any perspective
  // model is supported; the fitting refines that model's parameters directly.
  // Empty preserves each camera's existing model instead of converting it.
  std::string camera_model;

  // Whether to refine camera intrinsics when !camera.has_prior_focal_length.
  bool refine_intrinsics = true;

  // If true, refine intrinsics even for cameras with has_prior_focal_length.
  bool force_refine_intrinsics = false;

  // If true, overwrite existing valid gravity priors in the database.
  bool overwrite_gravity = true;

  // Plausibility bounds on the calibrated intrinsics, shared with the
  // incremental mapper and single-view calibration defaults.
  double min_focal_length_ratio = kDefaultMinFocalLengthRatio;
  double max_focal_length_ratio = kDefaultMaxFocalLengthRatio;
  double max_extra_param = kDefaultMaxExtraParam;

  // GeoCalib inference and Ceres perspective field fitting options.
  GeoCalibOptions geocalib;

  bool Check() const;
};

using GeoCalibFactory =
    std::function<std::unique_ptr<GeoCalib>(const GeoCalibOptions& options)>;

// Estimate per-frame gravity (and optionally camera intrinsics) for images in
// the database using GeoCalib, writing sensor-frame gravity vectors into
// PosePrior entries and aggregated camera parameters into Camera entries.
std::unique_ptr<Thread> CreateGravityEstimationController(
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const GravityEstimationOptions& options,
    const std::vector<std::string>& image_names = {},
    GeoCalibFactory geocalib_factory = {});

}  // namespace colmap

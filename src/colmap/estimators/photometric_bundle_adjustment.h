// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/ceres_loss_function.h"
#include "colmap/image/image_pyramid.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/hash_containers.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace colmap {

// Options for sparse photometric bundle adjustment (LSPBA + Least-Squares NCC).
struct PhotometricBundleAdjustmentOptions {
  // Side length P of the square corner-grid patch (2 <= P <= 8, M = P * P
  // samples per residual). Default 4x4 (16 samples) matches LSPBA.
  int patch_size = 4;

  // Comma-separated coarse-to-fine power-of-two image pyramid levels
  // (0 = base resolution, 1 = 1/2, 2 = 1/4, ...).
  std::string pyramid_levels = "2,1,0";

  // Optional maximum dimension cap when loading images into RAM (-1 = full
  // original resolution).
  int max_image_size = -1;

  // Maximum Ceres iterations for the structure-only warmup stage (optimizing
  // landmark planes with cameras held constant before joint optimization at
  // each pyramid level; 0 disables warmup).
  int structure_warmup_max_num_iterations = 20;

  // Minimum per-pixel intensity standard deviation (on [0, 255] grayscale)
  // required for a patch to be considered textured. For a 4x4 patch (M = 16),
  // min_patch_std = 2.0 corresponds to ||I - mean||_2 >= 8.0 = 0.5 * M,
  // matching LSPBA §3.4.
  double min_patch_std = 2.0;

  // Robust loss applied to each patch's squared LS-NCC residual ||r||^2 in
  // [0, 4]. Default Geman-McClure with scale 0.5 matches LSPBA §3.2.
  CeresLossFunctionType loss_function_type =
      CeresLossFunctionType::GEMAN_MCCLURE;
  double loss_function_scale = 0.5;

  // Dimensionless relative weight for VCE-scaled full-track geometric
  // reprojection regularization (0.0 = pure photometric BA). When > 0 and
  // camera poses or intrinsics are variable, geometric reprojection residuals
  // (using BundleAdjustmentOptions::ceres loss function) are weighted by
  //   w_geo = geometric_reprojection_weight *
  //           (median(||r_photo||^2) / median(||r_geo||^2)),
  // equalizing per-observation block variance across modalities at 1.0.
  double geometric_reprojection_weight = 1.0;

  // Number of CPU threads for image loading, pyramid construction, and source
  // selection (-1 = all available cores). When > 0, also overrides the Ceres
  // solver thread count; when -1, Ceres threading follows
  // BundleAdjustmentOptions::ceres.
  int num_threads = -1;

  std::vector<int> ParsePyramidLevels() const;
  bool Check() const;
};

// Summary returned by PhotometricBundleAdjuster::Solve().
struct PhotometricBundleAdjustmentSummary : public BundleAdjustmentSummary {
  int num_tracks_total = 0;
  int num_tracks_kept = 0;
  int num_tracks_filtered_short = 0;
  int num_tracks_filtered_invalid = 0;
  int num_tracks_filtered_low_texture = 0;
  double image_load_seconds = 0.0;
  double optimization_seconds = 0.0;
};

// Create a photometric bundle adjuster that loads images from `image_path`.
std::unique_ptr<BundleAdjuster> CreatePhotometricBundleAdjuster(
    const BundleAdjustmentOptions& ba_options,
    const PhotometricBundleAdjustmentOptions& photometric_options,
    const BundleAdjustmentConfig& config,
    const std::filesystem::path& image_path,
    Reconstruction& reconstruction);

// Create a photometric bundle adjuster operating on preloaded image pyramids
// (useful for unit tests and in-memory pipelines).
std::unique_ptr<BundleAdjuster> CreatePhotometricBundleAdjuster(
    const BundleAdjustmentOptions& ba_options,
    const PhotometricBundleAdjustmentOptions& photometric_options,
    const BundleAdjustmentConfig& config,
    std::shared_ptr<const FlatHashMap<image_t, GrayscaleImagePyramid>> pyramids,
    Reconstruction& reconstruction);

}  // namespace colmap

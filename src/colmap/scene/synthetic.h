// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/feature/types.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/models.h"

#include <filesystem>

namespace colmap {

struct SyntheticDatasetOptions {
  int num_rigs = 2;
  int num_cameras_per_rig = 1;
  int num_frames_per_rig = 5;
  int num_points3D = 100;
  // Target track length per 3D point. If -1 (default), all images observe all
  // points (dense visibility). If > 0, observations are pruned to exactly this
  // many per point. Must be -1 or >= 2.
  int track_length = -1;

  double sensor_from_rig_translation_stddev = 0.05;
  // Random rotation in degrees around the z-axis of the sensor.
  double sensor_from_rig_rotation_stddev = 5.;

  int camera_width = 1024;
  int camera_height = 768;
  CameraModelId camera_model_id = SimpleRadialCameraModel::model_id;
  std::vector<double> camera_params = {1280, 512, 384, 0.05};
  bool camera_has_prior_focal_length = false;

  // The type of feature descriptors to synthesize.
  FeatureExtractorType feature_type = FeatureExtractorType::SIFT;

  int num_points2D_without_point3D = 10;

  double inlier_match_ratio = 1.0;

  // Whether to include decomposed relative poses in two-view geometries.
  bool two_view_geometry_has_relative_pose = false;

  enum class MatchConfig {
    // Exhaustive matches between all pairs of observations of a 3D point.
    EXHAUSTIVE = 1,
    // Chain of matches between images with consecutive identifiers, i.e.,
    // there are only matches between image pairs (image_id, image_id+1).
    CHAINED = 2,
    // Sparse matches with controllable sparsity, removing edges randomly while
    // maintaining view graph connectivity.
    SPARSE = 3,
  };
  MatchConfig match_config = MatchConfig::EXHAUSTIVE;

  // Sparsity parameter for SPARSE match config, in range [0, 1].
  // 0 = fully connected view graph, equivalent to EXHAUSTIVE (all edges)
  // 1 = empty view graph (no edges)
  double match_sparsity = 0.0;

  bool prior_position = false;
  PosePrior::CoordinateSystem prior_position_coordinate_system =
      PosePrior::CoordinateSystem::CARTESIAN;
  bool prior_gravity = false;
  Eigen::Vector3d prior_gravity_in_world = Eigen::Vector3d::UnitY();

  // The synthesized image file extension.
  std::string image_extension = ".png";
};

void SynthesizeDataset(const SyntheticDatasetOptions& options,
                       Reconstruction* reconstruction,
                       Database* database = nullptr);

struct SyntheticNoiseOptions {
  double rig_from_world_translation_stddev = 0.0;
  // Random rotation in degrees around the z-axis of the rig.
  double rig_from_world_rotation_stddev = 0.0;
  double point3D_stddev = 0.0;
  double point2D_stddev = 0.0;

  // Translational standard deviation of the prior position in meters.
  double prior_position_stddev = 1.5;
  // Rotational standard deviation of the prior gravity in degrees.
  double prior_gravity_stddev = 1.0;
};

void SynthesizeNoise(const SyntheticNoiseOptions& options,
                     Reconstruction* reconstruction,
                     Database* database = nullptr);

struct SyntheticImageOptions {
  int feature_peak_radius = 2;
  int feature_patch_radius = 15;
  int feature_patch_max_brightness = 128;
};

// Generates patches with a dark background and a bright feature peak for each
// 2D point in an image. The color of the peak and the pattern of the background
// is unique per 3D point. Notice that this approach does not result in perfect
// feature detections and matches due to overlapping patches, etc.
void SynthesizeImages(const SyntheticImageOptions& options,
                      const Reconstruction& reconstruction,
                      const std::filesystem::path& image_path);

}  // namespace colmap

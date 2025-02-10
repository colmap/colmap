// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/models.h"
#include "colmap/util/types.h"

namespace colmap {

struct SyntheticDatasetOptions {
  int num_cameras = 2;
  int num_images = 10;
  int num_points3D = 100;

  int camera_width = 1024;
  int camera_height = 768;
  CameraModelId camera_model_id = SimpleRadialCameraModel::model_id;
  std::vector<double> camera_params = {1280, 512, 384, 0.05};

  int num_points2D_without_point3D = 10;
  double point2D_stddev = 0.0;

  double inlier_match_ratio = 1.0;

  enum class MatchConfig {
    // Exhaustive matches between all pairs of observations of a 3D point.
    EXHAUSTIVE = 1,
    // Chain of matches with random start/end observations.
    CHAINED = 2,
  };
  MatchConfig match_config = MatchConfig::EXHAUSTIVE;

  bool use_prior_position = false;
  bool use_geographic_coords_prior = false;
  double prior_position_stddev = 1.5;
};

void SynthesizeDataset(const SyntheticDatasetOptions& options,
                       Reconstruction* reconstruction,
                       Database* database = nullptr);

}  // namespace colmap

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

#include "colmap/scene/database.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/sfm/observation_manager.h"

namespace colmap {

// Algorithm class for incremental mapper to make it easier to extend
class IncrementalMapperImpl {
 public:
  // Find seed images for incremental reconstruction. Suitable seed images have
  // a large number of correspondences and have camera calibration priors. The
  // returned list is ordered such that most suitable images are in the front.
  static std::vector<image_t> FindFirstInitialImage(
      const IncrementalMapper::Options& options,
      const CorrespondenceGraph& correspondence_graph,
      const Reconstruction& reconstruction,
      const std::unordered_map<image_t, size_t>& init_num_reg_trials,
      const std::unordered_map<image_t, size_t>& num_registrations);

  // For a given first seed image, find other images that are connected to the
  // first image. Suitable second images have a large number of correspondences
  // to the first image and have camera calibration priors. The returned list is
  // ordered such that most suitable images are in the front.
  static std::vector<image_t> FindSecondInitialImage(
      const IncrementalMapper::Options& options,
      image_t image_id1,
      const CorrespondenceGraph& correspondence_graph,
      const Reconstruction& reconstruction,
      const std::unordered_map<image_t, size_t>& num_registrations);

  // Implement IncrementalMapper::FindInitialImagePair
  static bool FindInitialImagePair(
      const IncrementalMapper::Options& options,
      const DatabaseCache& database_cache,
      const Reconstruction& reconstruction,
      const std::unordered_map<image_t, size_t>& init_num_reg_trials,
      const std::unordered_map<image_t, size_t>& num_registrations,
      std::unordered_set<image_pair_t>& init_image_pairs,
      image_t& image_id1,
      image_t& image_id2,
      Rigid3d& cam2_from_cam1);

  // Implement IncrementalMapper::FindNextImages
  static std::vector<image_t> FindNextImages(
      const IncrementalMapper::Options& options,
      const ObservationManager& obs_manager,
      const std::unordered_set<image_t>& filtered_images,
      std::unordered_map<image_t, size_t>& num_reg_trials);

  // Implement IncrementalMapper::FindLocalBundle
  static std::vector<image_t> FindLocalBundle(
      const IncrementalMapper::Options& options,
      image_t image_id,
      const Reconstruction& reconstruction);

  // Implement IncrementalMapper::EstimateInitialTwoViewGeometry
  static bool EstimateInitialTwoViewGeometry(
      const IncrementalMapper::Options& options,
      const DatabaseCache& database_cache,
      image_t image_id1,
      image_t image_id2,
      Rigid3d& cam2_from_cam1);
};

}  // namespace colmap

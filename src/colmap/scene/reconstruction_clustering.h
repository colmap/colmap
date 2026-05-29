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

#include "colmap/scene/reconstruction.h"
#include "colmap/util/logging.h"
#include "colmap/util/types.h"

#include <unordered_map>

namespace colmap {

// Options for reconstruction clustering based on frame covisibility.
struct ReconstructionClusteringOptions {
  // Minimum number of shared 3D points between two frames to consider them
  // connected in the covisibility graph.
  int min_covisibility_count = 5;

  // Minimum edge weight threshold for clustering. If the adaptive threshold
  // (median - MAD) falls below this, this value is used instead.
  double min_edge_weight_threshold = 20.0;

  // Minimum number of registered frames required for a cluster to be kept.
  // Clusters with fewer frames will be discarded.
  int min_num_reg_frames = 3;

  void Check() const {
    THROW_CHECK_GE(min_covisibility_count, 1);
    THROW_CHECK_GT(min_edge_weight_threshold, 0.0);
    THROW_CHECK_GE(min_num_reg_frames, 2);
  }
};

// Clusters frames based on 3D point covisibility and removes weakly connected
// frames.
//
// Covisibility is the number of 3D points visible in both frames. Frames with
// high covisibility likely have reliable relative pose estimates, while weakly
// connected frames may have less reliable geometry.
//
// Algorithm:
//   1. Build a covisibility graph where edges connect frames sharing >=
//      min_covisibility_count points.
//   2. Compute an adaptive edge weight threshold using median minus median
//      absolute deviation (MAD).
//   3. Cluster frames using union-find: merge strongly connected frames.
//   4. Assign cluster IDs sorted by number of frames in descending order
//      (i.e., cluster ID 0 is the largest cluster).
//
// Args:
//   options: Configuration options for clustering.
//   reconstruction: The reconstruction containing frames and 3D points.
//
// Returns:
//   Map from frame_id to cluster_id for all registered frames. Cluster IDs are
//   sorted by number of frames (largest cluster has ID 0).
std::unordered_map<frame_t, int> ClusterReconstructionFrames(
    const ReconstructionClusteringOptions& options,
    Reconstruction& reconstruction);

}  // namespace colmap

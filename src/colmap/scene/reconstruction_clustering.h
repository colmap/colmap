// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/scene/reconstruction.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"
#include "colmap/util/types.h"

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
NodeHashMap<frame_t, int> ClusterReconstructionFrames(
    const ReconstructionClusteringOptions& options,
    Reconstruction& reconstruction);

}  // namespace colmap

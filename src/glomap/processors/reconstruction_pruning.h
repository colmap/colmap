#pragma once

#include "colmap/scene/reconstruction.h"

#include "glomap/scene/types.h"

namespace glomap {

// Prunes weakly connected frames and clusters the remainder based on 3D point
// covisibility.
//
// Covisibility is the number of 3D points visible in both frames. Frames with
// high covisibility likely have reliable relative pose estimates, while weakly
// connected frames may have less reliable geometry.
//
// Algorithm:
//   1. Build a covisibility graph where edges connect frames sharing >= 5
//      points.
//   2. Compute an adaptive edge weight threshold using median minus median
//      absolute deviation (MAD).
//   3. Cluster frames using union-find: first merge strongly connected frames,
//      then iteratively merge clusters connected by two weaker (0.75 *
//      threshold) edges.
//
// Args:
//   reconstruction: The reconstruction containing frames and 3D points.
//
// Returns:
//   Map from frame_id to cluster_id for all registered frames
std::unordered_map<frame_t, int> PruneWeaklyConnectedFrames(
    colmap::Reconstruction& reconstruction);

}  // namespace glomap

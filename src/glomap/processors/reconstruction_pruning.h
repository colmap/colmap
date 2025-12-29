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
//   2. Find the largest connected component and de-register all frames outside
//      it. This removes isolated or poorly connected parts of the
//      reconstruction.
//   3. Compute an adaptive edge weight threshold using median minus maximum
//   absolute deviation (MAD).
//   4. Cluster frames using union-find: first merge strongly connected frames,
//      then iteratively merge clusters connected by two weaker (0.75 *
//      threshold) edges.
//
// Args:
//   reconstruction: The reconstruction containing frames and 3D points.
//      Frames outside the largest connected component will be de-registered.
//
// Returns:
//   Map from frame_id to cluster_id for frames in the largest component.
std::unordered_map<frame_t, int> PruneWeaklyConnectedFrames(
    colmap::Reconstruction& reconstruction);

}  // namespace glomap

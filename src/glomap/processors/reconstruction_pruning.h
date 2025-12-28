#pragma once

#include "colmap/scene/reconstruction.h"

#include "glomap/scene/types.h"

namespace glomap {

// Clusters frames based on 3D point covisibility to identify weakly connected
// components in the reconstruction.
//
// Covisibility is the number of 3D points visible in both frames. Frames with
// high covisibility likely have reliable relative pose estimates, while weakly
// connected frames may have less reliable geometry.
//
// The clustering uses an adaptive threshold (median - MAD) to handle varying
// covisibility distributions. Frames are grouped using union-find, with
// iterative refinement to merge clusters connected by multiple weaker edges.
//
// Args:
//   reconstruction: The reconstruction containing frames and 3D points.
//
// Returns:
//   Map from frame_id to cluster_id.
std::unordered_map<frame_t, int> PruneWeaklyConnectedFrames(
    colmap::Reconstruction& reconstruction);

}  // namespace glomap

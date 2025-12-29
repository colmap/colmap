#pragma once

#include "colmap/scene/reconstruction.h"

#include "glomap/types.h"

namespace glomap {

// Extract a subset of the reconstruction for a specific cluster.
// Returns a new Reconstruction containing only frames/images/points from the
// specified cluster. If cluster_id is -1, returns a copy of the full
// reconstruction.
colmap::Reconstruction SubReconstructionByClusterId(
    const colmap::Reconstruction& reconstruction,
    const std::unordered_map<frame_t, int>& cluster_ids,
    int cluster_id);

// Write a reconstruction to disk, optionally splitting by cluster.
// If cluster_ids is empty, writes the full reconstruction to path/0/.
// If cluster_ids is provided, writes each cluster to path/{cluster_id}/.
void WriteReconstructionsByClusters(
    const std::string& reconstruction_path,
    const colmap::Reconstruction& reconstruction,
    const std::unordered_map<frame_t, int>& cluster_ids = {},
    const std::string& output_format = "bin",
    const std::string& image_path = "");

}  // namespace glomap

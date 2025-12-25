#pragma once

#include "colmap/scene/reconstruction.h"

#include "glomap/scene/types.h"

namespace glomap {

// Write a reconstruction to disk, optionally splitting by cluster.
// If cluster_ids is empty, writes the full reconstruction to path/0/.
// If cluster_ids is provided, writes each cluster to path/{cluster_id}/.
void WriteGlomapReconstruction(
    const std::string& reconstruction_path,
    const colmap::Reconstruction& reconstruction,
    const std::unordered_map<frame_t, int>& cluster_ids = {},
    const std::string& output_format = "bin",
    const std::string& image_path = "");

void WriteColmapReconstruction(const std::string& reconstruction_path,
                               const colmap::Reconstruction& reconstruction,
                               const std::string& output_format = "bin");

}  // namespace glomap

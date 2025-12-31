#pragma once

#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"

#include "glomap/scene/types.h"
#include "glomap/scene/view_graph.h"

namespace glomap {

// Initialize an empty reconstruction from the database.
// This adds cameras, rigs, frames, and images (without 3D points).
void InitializeEmptyReconstructionFromDatabase(
    const colmap::Database& database, colmap::Reconstruction& reconstruction);

// Initialize the view graph from the database.
void InitializeViewGraphFromDatabase(const colmap::Database& database,
                                     ViewGraph& view_graph);

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

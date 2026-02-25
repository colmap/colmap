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

#include "colmap/scene/reconstruction_clustering.h"

#include "colmap/math/connected_components.h"
#include "colmap/math/math.h"
#include "colmap/math/union_find.h"
#include "colmap/util/logging.h"

namespace colmap {
namespace {

// Alias for clarity: we're working with frame pairs, not image pairs.
using frame_pair_t = image_pair_t;

// Clusters nodes using union-find based on edge weights.
//
// Algorithm:
//   Merge nodes connected by strong edges (weight > threshold).
//
std::unordered_map<frame_t, int> EstablishStrongClusters(
    const ReconstructionClusteringOptions& options,
    const std::unordered_set<frame_t>& nodes,
    const std::unordered_map<frame_pair_t, int>& edge_weights,
    double edge_weight_threshold) {
  UnionFind<frame_t> uf;
  uf.Reserve(nodes.size());

  // Create initial clusters from strong edges (weight > threshold).
  // TODO(lpanaf): use different thresholds for different edges based on local
  // statistics.
  for (const auto& [pair_id, weight] : edge_weights) {
    if (weight >= edge_weight_threshold) {
      const auto [frame_id1, frame_id2] = PairIdToImagePair(pair_id);
      uf.Union(frame_id1, frame_id2);
    }
  }

  // Assign sequential cluster IDs (largest cluster gets ID 0).
  uf.Compress();
  std::unordered_map<frame_t, std::vector<frame_t>> root_to_nodes;
  for (const auto& [node, root] : uf.Parents()) {
    root_to_nodes[root].push_back(node);
  }

  // Phase 3: Collect nodes by their union-find roots, sort by number of
  // frames, and assign sequential cluster IDs (largest cluster gets ID 0).
  std::vector<std::vector<frame_t>> sorted_clusters;
  sorted_clusters.reserve(root_to_nodes.size());
  for (auto& [root, cluster_nodes] : root_to_nodes) {
    sorted_clusters.push_back(std::move(cluster_nodes));
  }
  std::sort(sorted_clusters.begin(),
            sorted_clusters.end(),
            [](const auto& a, const auto& b) { return a.size() > b.size(); });

  // Assign cluster IDs based on sorted order.
  std::unordered_map<frame_t, int> cluster_ids;
  int num_valid_clusters = 0;
  for (size_t cluster_id = 0; cluster_id < sorted_clusters.size();
       ++cluster_id) {
    const auto& cluster_nodes = sorted_clusters[cluster_id];
    if (cluster_nodes.size() >= size_t(options.min_num_reg_frames)) {
      for (const frame_t node : cluster_nodes) {
        cluster_ids[node] = static_cast<int>(num_valid_clusters);
      }
      num_valid_clusters++;
    } else {
      // Clusters smaller than min_num_reg_frames are discarded.
      for (const frame_t node : cluster_nodes) {
        cluster_ids[node] = -1;
      }
    }
  }

  // Ensure all nodes are assigned a cluster ID.
  for (const auto node : nodes) {
    if (!uf.FindIfExists(node).has_value()) cluster_ids[node] = -1;
  }

  LOG(INFO) << "Frames are grouped into " << num_valid_clusters
            << " clusters (size >= " << options.min_num_reg_frames << ")";

  return cluster_ids;
}

}  // namespace

std::unordered_map<frame_t, int> ClusterReconstructionFrames(
    const ReconstructionClusteringOptions& options,
    Reconstruction& reconstruction) {
  options.Check();

  // Step 1: Compute covisibility counts between all frame pairs.
  // For each 3D point, increment the count for every pair of frames that sees
  // it.
  std::unordered_map<frame_pair_t, int> frame_covisibility_count;
  std::unordered_set<frame_t> nodes;
  // Insert all registered frames to the nodes set.
  for (const frame_t frame_id : reconstruction.RegFrameIds()) {
    nodes.insert(frame_id);
  }

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <= 2) continue;

    for (size_t i = 0; i < point3D.track.Length(); i++) {
      const image_t image_id1 = point3D.track.Element(i).image_id;
      const frame_t frame_id1 = reconstruction.Image(image_id1).FrameId();

      nodes.insert(frame_id1);
      // TODO: This may over-count frame pairs when multiple images from the
      // same frame appear in a track (e.g., rig cameras). Figure out if
      // intended.
      for (size_t j = i + 1; j < point3D.track.Length(); j++) {
        const image_t image_id2 = point3D.track.Element(j).image_id;
        const frame_t frame_id2 = reconstruction.Image(image_id2).FrameId();
        if (frame_id1 == frame_id2) continue;
        const frame_pair_t pair_id = ImagePairToPairId(frame_id1, frame_id2);
        frame_covisibility_count[pair_id]++;
      }
    }
  }

  // Filter edges to keep only reliable connections.
  std::unordered_map<frame_pair_t, int> edge_weights;
  for (const auto& [pair_id, count] : frame_covisibility_count) {
    if (count < options.min_covisibility_count) continue;
    edge_weights[pair_id] = count;
  }
  LOG(INFO) << "Established visibility graph with " << edge_weights.size()
            << " pairs";

  if (edge_weights.empty()) {
    LOG(WARNING) << "No valid frame pairs found for clustering";
    return {};
  }

  // Compute adaptive threshold using median minus median absolute
  // deviation (MAD).
  std::vector<int> weight_values;
  weight_values.reserve(edge_weights.size());
  for (const auto& [pair_id, weight] : edge_weights) {
    weight_values.push_back(weight);
  }
  const auto [median, mad] = MedianAbsoluteDeviation(std::move(weight_values));
  const double threshold =
      std::max(median - mad, options.min_edge_weight_threshold);
  LOG(INFO) << "Threshold for strong cluster: " << threshold;

  // Cluster frames based on covisibility weights.
  return EstablishStrongClusters(options, nodes, edge_weights, threshold);
}

}  // namespace colmap

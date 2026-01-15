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

// Clusters nodes using union-find based on edge weights.
//
// Algorithm:
//   1. Merge nodes connected by strong edges (weight > threshold).
//   2. Iteratively merge clusters connected by at least min_weak_edges_to_merge
//      weaker edges (weight >= weak_edge_multiplier * threshold).
//   3. Assign sequential cluster IDs based on union-find roots.
//
// The iterative refinement helps avoid over-segmentation when the connection
// between two groups of nodes is distributed across multiple weaker edges.
std::unordered_map<frame_t, int> EstablishStrongClusters(
    const ReconstructionClusteringOptions& options,
    const std::unordered_set<frame_t>& nodes,
    const std::unordered_map<image_pair_t, int>& edge_weights,
    double edge_weight_threshold) {
  std::unordered_map<frame_t, int> cluster_ids;
  UnionFind<frame_t> uf;
  uf.Reserve(nodes.size());

  // Phase 1: Create initial clusters from strong edges (weight > threshold).
  for (const auto& [pair_id, weight] : edge_weights) {
    if (weight >= edge_weight_threshold) {
      const auto [frame_id1, frame_id2] = PairIdToImagePair(pair_id);
      uf.Union(frame_id1, frame_id2);
    }
  }

  // Phase 2: Iteratively merge clusters connected by multiple weaker edges.
  // Two clusters are merged if they share >= min_weak_edges_to_merge edges with
  // weight >= weak_edge_multiplier * threshold. This continues until no more
  // merges occur (or max max_clustering_iterations iterations).
  bool changed = true;
  int iteration = 0;
  while (changed) {
    changed = false;
    iteration++;

    if (iteration > options.max_clustering_iterations) {
      break;
    }

    // Count edges between each pair of cluster roots.
    std::unordered_map<frame_t, std::unordered_map<frame_t, int>> num_pairs;
    for (const auto& [pair_id, weight] : edge_weights) {
      if (weight < options.weak_edge_multiplier * edge_weight_threshold)
        continue;

      const auto [frame_id1, frame_id2] = PairIdToImagePair(pair_id);
      frame_t root1 = uf.Find(frame_id1);
      frame_t root2 = uf.Find(frame_id2);

      if (root1 == root2) continue;  // Already in same cluster.

      num_pairs[root1][root2]++;
      num_pairs[root2][root1]++;
    }

    // Merge clusters that share >= min_weak_edges_to_merge connecting edges.
    for (const auto& [root1, counter] : num_pairs) {
      for (const auto& [root2, count] : counter) {
        if (root1 <= root2) continue;  // Process each pair once.

        if (count >= options.min_weak_edges_to_merge) {
          changed = true;
          uf.Union(root1, root2);
        }
      }
    }
  }

  // Phase 3: Assign sequential cluster IDs based on union-find roots.
  std::unordered_map<frame_t, int> root_to_cluster;
  int next_cluster_id = 0;

  for (const frame_t node : nodes) {
    frame_t root = uf.Find(node);
    auto it = root_to_cluster.find(root);
    if (it == root_to_cluster.end()) {
      root_to_cluster[root] = next_cluster_id++;
    }
    cluster_ids[node] = root_to_cluster[root];
  }

  // Count clusters with at least kMinClusterSize frames.
  constexpr int kMinClusterSize = 2;
  std::unordered_map<int, int> cluster_sizes;
  for (const auto& [node, cluster_id] : cluster_ids) {
    cluster_sizes[cluster_id]++;
  }
  int num_valid_clusters = 0;
  for (const auto& [cluster_id, size] : cluster_sizes) {
    if (size >= kMinClusterSize) {
      num_valid_clusters++;
    }
  }

  LOG(INFO) << "Clustering took " << iteration << " iterations. "
            << "Frames are grouped into " << num_valid_clusters
            << " clusters (size >= " << kMinClusterSize << ")";

  return cluster_ids;
}

}  // namespace

std::unordered_map<frame_t, int> ClusterReconstructionFrames(
    const ReconstructionClusteringOptions& options,
    Reconstruction& reconstruction) {
  // Step 1: Compute covisibility counts between all frame pairs.
  // For each 3D point, increment the count for every pair of frames that sees
  // it.
  std::unordered_map<image_pair_t, int> frame_covisibility_count;
  std::unordered_set<frame_t> nodes;
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
        const image_pair_t pair_id = ImagePairToPairId(frame_id1, frame_id2);
        frame_covisibility_count[pair_id]++;
      }
    }
  }

  // Filter edges to keep only reliable connections.
  std::unordered_map<image_pair_t, int> edge_weights;
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
  LOG(INFO) << "Threshold for Strong Clustering: " << threshold;

  // Cluster frames based on covisibility weights.
  return EstablishStrongClusters(options, nodes, edge_weights, threshold);
}

}  // namespace colmap

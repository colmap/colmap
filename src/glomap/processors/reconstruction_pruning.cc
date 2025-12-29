#include "glomap/processors/reconstruction_pruning.h"

#include "colmap/math/connected_components.h"
#include "colmap/math/math.h"
#include "colmap/math/union_find.h"

namespace glomap {
namespace {

// Alias for clarity: we're working with frame pairs, not image pairs.
using frame_pair_t = image_pair_t;

// Clustering parameters.
constexpr int kMinCovisibilityCount = 5;
constexpr double kMinEdgeWeightThreshold = 20.0;
constexpr double kWeakEdgeMultiplier = 0.75;
constexpr int kMinWeakEdgesToMerge = 2;
constexpr int kMaxClusteringIterations = 10;

// Clusters nodes using union-find based on edge weights.
//
// Algorithm:
//   1. Merge nodes connected by strong edges (weight > threshold).
//   2. Iteratively merge clusters connected by at least kMinWeakEdgesToMerge
//      weaker edges (weight >= kWeakEdgeMultiplier * threshold).
//   3. Assign sequential cluster IDs based on union-find roots.
//
// The iterative refinement helps avoid over-segmentation when the connection
// between two groups of nodes is distributed across multiple weaker edges.
std::unordered_map<frame_t, int> EstablishStrongClusters(
    const std::unordered_set<frame_t>& nodes,
    const std::unordered_map<frame_pair_t, int>& edge_weights,
    double edge_weight_threshold) {
  std::unordered_map<frame_t, int> cluster_ids;
  colmap::UnionFind<frame_t> uf;
  uf.Reserve(nodes.size());

  // Phase 1: Create initial clusters from strong edges (weight > threshold).
  for (const auto& [pair_id, weight] : edge_weights) {
    if (weight > edge_weight_threshold) {
      const auto [frame_id1, frame_id2] = colmap::PairIdToImagePair(pair_id);
      uf.Union(frame_id1, frame_id2);
    }
  }

  // Phase 2: Iteratively merge clusters connected by multiple weaker edges.
  // Two clusters are merged if they share >= kMinWeakEdgesToMerge edges with
  // weight >= kWeakEdgeMultiplier * threshold. This continues until no more
  // merges occur (or max kMaxClusteringIterations iterations).
  bool changed = true;
  int iteration = 0;
  while (changed) {
    changed = false;
    iteration++;

    if (iteration > kMaxClusteringIterations) {
      break;
    }

    // Count edges between each pair of cluster roots.
    std::unordered_map<frame_t, std::unordered_map<frame_t, int>> num_pairs;
    for (const auto& [pair_id, weight] : edge_weights) {
      if (weight < kWeakEdgeMultiplier * edge_weight_threshold) continue;

      const auto [frame_id1, frame_id2] = colmap::PairIdToImagePair(pair_id);
      frame_t root1 = uf.Find(frame_id1);
      frame_t root2 = uf.Find(frame_id2);

      if (root1 == root2) continue;  // Already in same cluster.

      num_pairs[root1][root2]++;
      num_pairs[root2][root1]++;
    }

    // Merge clusters that share >= 2 connecting edges.
    for (const auto& [root1, counter] : num_pairs) {
      for (const auto& [root2, count] : counter) {
        if (root1 <= root2) continue;  // Process each pair once.

        if (count >= kMinWeakEdgesToMerge) {
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

std::unordered_map<frame_t, int> PruneWeaklyConnectedFrames(
    colmap::Reconstruction& reconstruction) {
  // Step 1: Compute covisibility counts between all frame pairs.
  // For each 3D point, increment the count for every pair of frames that sees
  // it.
  std::unordered_map<frame_pair_t, int> frame_covisibility_count;
  std::unordered_set<frame_t> nodes;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <= 2) continue;

    for (size_t i = 0; i < point3D.track.Length(); i++) {
      const image_t image_id1 = point3D.track.Element(i).image_id;
      const frame_t frame_id1 = reconstruction.Image(image_id1).FrameId();

      nodes.insert(frame_id1);
      for (size_t j = i + 1; j < point3D.track.Length(); j++) {
        const image_t image_id2 = point3D.track.Element(j).image_id;
        const frame_t frame_id2 = reconstruction.Image(image_id2).FrameId();
        if (frame_id1 == frame_id2) continue;
        const frame_pair_t pair_id =
            colmap::ImagePairToPairId(frame_id1, frame_id2);
        frame_covisibility_count[pair_id]++;
      }
    }
  }

  // Step 2: Filter edges to keep only reliable connections.
  std::unordered_map<frame_pair_t, int> edge_weights;
  for (const auto& [pair_id, count] : frame_covisibility_count) {
    if (count < kMinCovisibilityCount) continue;
    edge_weights[pair_id] = count;
  }
  LOG(INFO) << "Established visibility graph with " << edge_weights.size()
            << " pairs";

  if (edge_weights.empty()) {
    LOG(WARNING) << "No valid frame pairs found for clustering";
    return {};
  }

  // Step 3: Keep only the largest connected component and de-register the rest.
  std::vector<std::pair<frame_t, frame_t>> edges;
  edges.reserve(edge_weights.size());
  for (const auto& [pair_id, weight] : edge_weights) {
    const auto [frame_id1, frame_id2] = colmap::PairIdToImagePair(pair_id);
    edges.emplace_back(frame_id1, frame_id2);
  }
  const std::vector<frame_t> largest_cc_vec =
      colmap::FindLargestConnectedComponent(nodes, edges);
  const std::unordered_set<frame_t> largest_cc(largest_cc_vec.begin(),
                                               largest_cc_vec.end());
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (largest_cc.count(frame_id) == 0 && frame.HasPose()) {
      reconstruction.DeRegisterFrame(frame_id);
    }
  }
  LOG(INFO) << "Kept " << largest_cc.size() << " frames in largest component";

  // Filter to keep only edges within the largest component.
  for (auto it = edge_weights.begin(); it != edge_weights.end();) {
    const auto [frame_id1, frame_id2] = colmap::PairIdToImagePair(it->first);
    if (largest_cc.count(frame_id1) == 0 || largest_cc.count(frame_id2) == 0) {
      it = edge_weights.erase(it);
    } else {
      ++it;
    }
  }

  // Step 4: Compute adaptive threshold using median minus median absolute
  // deviation (MAD). Extract weight values after filtering to largest CC.
  std::vector<int> weight_values;
  weight_values.reserve(edge_weights.size());
  for (const auto& [pair_id, weight] : edge_weights) {
    weight_values.push_back(weight);
  }
  const auto [median, mad] =
      colmap::MedianAbsoluteDeviation(std::move(weight_values));
  const double threshold = std::max(median - mad, kMinEdgeWeightThreshold);
  LOG(INFO) << "Threshold for Strong Clustering: " << threshold;

  // Step 5: Cluster frames based on covisibility weights.
  return EstablishStrongClusters(largest_cc, edge_weights, threshold);
}

}  // namespace glomap

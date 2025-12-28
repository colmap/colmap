#include "glomap/processors/reconstruction_pruning.h"

#include "colmap/math/union_find.h"

namespace glomap {
namespace {

// Alias for clarity: we're working with frame pairs, not image pairs.
using frame_pair_t = image_pair_t;

// Computes the Median Absolute Deviation of a sorted vector.
double ComputeMedianAbsoluteDeviation(const std::vector<int>& sorted_values) {
  const double median = sorted_values[sorted_values.size() / 2];
  std::vector<int> abs_deviations(sorted_values.size());
  for (size_t i = 0; i < sorted_values.size(); i++) {
    abs_deviations[i] = std::abs(sorted_values[i] - static_cast<int>(median));
  }
  std::sort(abs_deviations.begin(), abs_deviations.end());
  return abs_deviations[abs_deviations.size() / 2];
}

// Finds connected components and returns the largest one.
std::unordered_set<frame_t> FindLargestConnectedComponent(
    const std::unordered_set<frame_t>& nodes,
    const std::unordered_map<frame_pair_t, int>& edge_weights) {
  colmap::UnionFind<frame_t> uf;
  uf.Reserve(nodes.size());

  // Connect all nodes that share an edge.
  for (const auto& [pair_id, weight] : edge_weights) {
    const auto [frame_id1, frame_id2] = colmap::PairIdToImagePair(pair_id);
    uf.Union(frame_id1, frame_id2);
  }

  // Count sizes of each component.
  std::unordered_map<frame_t, std::vector<frame_t>> components;
  for (const frame_t node : nodes) {
    components[uf.Find(node)].push_back(node);
  }

  // Find the largest component.
  frame_t largest_root = 0;
  size_t largest_size = 0;
  for (const auto& [root, members] : components) {
    if (members.size() > largest_size) {
      largest_size = members.size();
      largest_root = root;
    }
  }

  return std::unordered_set<frame_t>(components[largest_root].begin(),
                                     components[largest_root].end());
}

// Clusters nodes using union-find based on edge weights.
//
// Algorithm:
//   1. Merge nodes connected by strong edges (weight > threshold).
//   2. Iteratively merge clusters connected by at least 2 weaker edges
//      (weight >= 0.75 * threshold).
//   3. Assign sequential cluster IDs based on union-find roots.
//
// The iterative refinement helps avoid over-segmentation when the connection
// between two groups of nodes is distributed across at least 2 edges.
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
  // Two clusters are merged if they share >= 2 edges with weight >= 0.75 *
  // threshold. This continues until no more merges occur (or max 10
  // iterations).
  bool changed = true;
  int iteration = 0;
  while (changed) {
    changed = false;
    iteration++;

    if (iteration > 10) {
      break;
    }

    // Count edges between each pair of cluster roots.
    std::unordered_map<frame_t, std::unordered_map<frame_t, int>> num_pairs;
    for (const auto& [pair_id, weight] : edge_weights) {
      if (weight < 0.75 * edge_weight_threshold) continue;

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

        if (count >= 2) {
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

std::unordered_map<frame_t, int> PruneWeaklyConnectedImages(
    colmap::Reconstruction& reconstruction) {
  // Step 1: Compute covisibility counts between all frame pairs.
  // For each 3D point, increment the count for every pair of frames that sees
  // it.
  std::unordered_map<frame_pair_t, int> frame_covisibility_count;
  std::unordered_set<frame_t> nodes;
  for (const auto& [track_id, track] : reconstruction.Points3D()) {
    if (track.track.Length() <= 2) continue;

    for (size_t i = 0; i < track.track.Length(); i++) {
      const image_t image_id1 = track.track.Element(i).image_id;
      const frame_t frame_id1 = reconstruction.Image(image_id1).FrameId();

      nodes.insert(frame_id1);
      for (size_t j = i + 1; j < track.track.Length(); j++) {
        const image_t image_id2 = track.track.Element(j).image_id;
        const frame_t frame_id2 = reconstruction.Image(image_id2).FrameId();
        if (frame_id1 == frame_id2) continue;
        const frame_pair_t pair_id =
            colmap::ImagePairToPairId(frame_id1, frame_id2);
        frame_covisibility_count[pair_id]++;
      }
    }
  }

  // Step 2: Filter edges to keep only reliable connections.
  // Require at least 5 shared points (needed for stable relative pose).
  std::unordered_map<frame_pair_t, int> edge_weights;
  std::vector<int> weight_values;
  for (const auto& [pair_id, count] : frame_covisibility_count) {
    if (count < 5) continue;

    edge_weights[pair_id] = count;
    weight_values.push_back(count);
  }
  LOG(INFO) << "Established visibility graph with " << edge_weights.size()
            << " pairs";

  if (weight_values.empty()) {
    LOG(WARNING) << "No valid frame pairs found for clustering";
    return {};
  }

  // Step 3: Keep only the largest connected component and de-register the rest.
  const std::unordered_set<frame_t> largest_cc =
      FindLargestConnectedComponent(nodes, edge_weights);
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (largest_cc.count(frame_id) == 0 && frame.HasPose()) {
      reconstruction.DeRegisterFrame(frame_id);
    }
  }
  LOG(INFO) << "Kept " << largest_cc.size() << " frames in largest component";

  // Filter to keep only edges within the largest component.
  std::erase_if(edge_weights, [&largest_cc](const auto& pair) {
    const auto [frame_id1, frame_id2] = colmap::PairIdToImagePair(pair.first);
    return largest_cc.count(frame_id1) == 0 || largest_cc.count(frame_id2) == 0;
  });

  // Step 4: Compute adaptive threshold using median - MAD.
  std::sort(weight_values.begin(), weight_values.end());
  const double median = weight_values[weight_values.size() / 2];
  const double threshold =
      std::max(median - ComputeMedianAbsoluteDeviation(weight_values), 20.0);
  LOG(INFO) << "Threshold for Strong Clustering: " << threshold;

  // Step 5: Cluster frames based on covisibility weights.
  return EstablishStrongClusters(largest_cc, edge_weights, threshold);
}

}  // namespace glomap

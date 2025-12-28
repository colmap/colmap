#include "glomap/math/tree.h"

#include "colmap/math/spanning_tree.h"

namespace glomap {

image_t MaximumSpanningTree(const ViewGraph& view_graph,
                            const std::unordered_map<image_t, Image>& images,
                            std::unordered_map<image_t, image_t>& parents,
                            WeightType type) {
  // Build mapping between image_id and contiguous indices.
  std::unordered_map<image_t, int> image_id_to_idx;
  std::vector<image_t> idx_to_image_id;
  image_id_to_idx.reserve(images.size());
  idx_to_image_id.reserve(images.size());

  for (const auto& [image_id, image] : images) {
    if (image.HasPose()) {
      image_id_to_idx[image_id] = static_cast<int>(idx_to_image_id.size());
      idx_to_image_id.push_back(image_id);
    }
  }

  // Build edges and weights from view graph.
  std::vector<std::pair<int, int>> edges;
  std::vector<double> weights;
  edges.reserve(view_graph.image_pairs.size());
  weights.reserve(view_graph.image_pairs.size());

  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) {
      continue;
    }
    const auto it1 = image_id_to_idx.find(image_pair.image_id1);
    const auto it2 = image_id_to_idx.find(image_pair.image_id2);
    if (it1 == image_id_to_idx.end() || it2 == image_id_to_idx.end()) {
      continue;
    }
    edges.emplace_back(it1->second, it2->second);
    weights.push_back(type == WeightType::INLIER_NUM
                          ? static_cast<double>(image_pair.inliers.size())
                          : image_pair.weight);
  }

  // Compute spanning tree using generic algorithm.
  const colmap::SpanningTree tree =
      colmap::ComputeMaximumSpanningTree(idx_to_image_id.size(), edges, weights);

  // Convert back to image_id based parent map.
  parents.clear();
  for (size_t i = 0; i < idx_to_image_id.size(); ++i) {
    if (tree.parents[i] >= 0) {
      parents[idx_to_image_id[i]] = idx_to_image_id[tree.parents[i]];
    }
  }

  return idx_to_image_id[tree.root];
}

}  // namespace glomap

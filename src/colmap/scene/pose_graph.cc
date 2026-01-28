#include "colmap/scene/pose_graph.h"

#include "colmap/math/connected_components.h"

namespace colmap {

void PoseGraph::Load(const CorrespondenceGraph& corr_graph) {
  for (const auto& [pair_id, num_matches] :
       corr_graph.NumMatchesBetweenAllImages()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const TwoViewGeometry two_view_geometry = corr_graph.ExtractTwoViewGeometry(
        image_id1, image_id2, /*extract_inlier_matches=*/false);
    if (two_view_geometry.cam2_from_cam1.has_value()) {
      Edge edge;
      edge.cam2_from_cam1 = *two_view_geometry.cam2_from_cam1;
      edge.num_matches = num_matches;
      AddEdge(image_id1, image_id2, std::move(edge));
    }
  }

  LOG(INFO) << "Loaded " << edges_.size() << " edges into pose graph";
}

std::unordered_set<frame_t> PoseGraph::ComputeLargestConnectedFrameComponent(
    const Reconstruction& reconstruction, bool filter_unregistered) const {
  std::unordered_set<frame_t> nodes;
  std::vector<std::pair<frame_t, frame_t>> graph_edges;

  for (const auto& [pair_id, edge] : ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const frame_t frame_id1 = reconstruction.Image(image_id1).FrameId();
    const frame_t frame_id2 = reconstruction.Image(image_id2).FrameId();

    // If filter_unregistered, skip pairs where either frame has no pose.
    if (filter_unregistered) {
      if (!reconstruction.Frame(frame_id1).HasPose() ||
          !reconstruction.Frame(frame_id2).HasPose()) {
        continue;
      }
    }

    nodes.insert(frame_id1);
    nodes.insert(frame_id2);
    graph_edges.emplace_back(frame_id1, frame_id2);
  }

  if (nodes.empty()) {
    return {};
  }

  const std::vector<frame_t> largest_component_vec =
      FindLargestConnectedComponent(nodes, graph_edges);
  return std::unordered_set<frame_t>(largest_component_vec.begin(),
                                     largest_component_vec.end());
}

void PoseGraph::InvalidatePairsOutsideActiveImageIds(
    const std::unordered_set<image_t>& active_image_ids) {
  for (const auto& [pair_id, edge] : edges_) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (!active_image_ids.count(image_id1) ||
        !active_image_ids.count(image_id2)) {
      SetInvalidEdge(pair_id);
    }
  }
}

int PoseGraph::MarkConnectedComponents(
    const Reconstruction& reconstruction,
    std::unordered_map<frame_t, int>& cluster_ids,
    int min_num_images) const {
  std::unordered_set<frame_t> nodes;
  std::vector<std::pair<frame_t, frame_t>> graph_edges;
  for (const auto& [pair_id, edge] : ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const frame_t frame_id1 = reconstruction.Image(image_id1).FrameId();
    const frame_t frame_id2 = reconstruction.Image(image_id2).FrameId();
    nodes.insert(frame_id1);
    nodes.insert(frame_id2);
    graph_edges.emplace_back(frame_id1, frame_id2);
  }

  const std::vector<std::vector<frame_t>> connected_components =
      FindConnectedComponents(nodes, graph_edges);
  const int num_comp = static_cast<int>(connected_components.size());

  std::vector<std::pair<int, int>> comp_num_images(num_comp);
  for (int comp = 0; comp < num_comp; comp++) {
    comp_num_images[comp] =
        std::make_pair(connected_components[comp].size(), comp);
  }
  std::sort(comp_num_images.begin(), comp_num_images.end(), std::greater<>());

  // Clear and populate cluster_ids output parameter
  cluster_ids.clear();
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    cluster_ids[frame_id] = -1;
  }

  int comp = 0;
  for (; comp < num_comp; comp++) {
    if (comp_num_images[comp].first < min_num_images) break;
    for (auto frame_id : connected_components[comp_num_images[comp].second]) {
      cluster_ids[frame_id] = comp;
    }
  }

  return comp;
}

}  // namespace colmap

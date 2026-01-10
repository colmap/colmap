#include "glomap/scene/pose_graph.h"

#include "colmap/math/connected_components.h"

namespace glomap {

void PoseGraph::Load(const colmap::DatabaseCache& cache) {
  const auto corr_graph = cache.CorrespondenceGraph();

  for (const auto& [pair_id, _] : corr_graph->NumMatchesBetweenAllImages()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    colmap::TwoViewGeometry two_view_geometry =
        corr_graph->TwoViewGeometry(image_id1, image_id2);
    if (!two_view_geometry.cam2_from_cam1.has_value()) {
      continue;
    }
    Edge edge;
    edge.cam2_from_cam1 = *two_view_geometry.cam2_from_cam1;
    edge.inlier_matches = std::move(two_view_geometry.inlier_matches);

    AddEdge(image_id1, image_id2, std::move(edge));
  }

  LOG(INFO) << "Loaded " << edges_.size() << " edges into pose graph";
}

std::unordered_set<frame_t> PoseGraph::ComputeLargestConnectedFrameComponent(
    const colmap::Reconstruction& reconstruction,
    bool filter_unregistered) const {
  std::unordered_set<frame_t> nodes;
  std::vector<std::pair<frame_t, frame_t>> graph_edges;

  for (const auto& [pair_id, edge] : ValidEdges()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
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
      colmap::FindLargestConnectedComponent(nodes, graph_edges);
  return std::unordered_set<frame_t>(largest_component_vec.begin(),
                                     largest_component_vec.end());
}

void PoseGraph::InvalidatePairsOutsideActiveImageIds(
    const std::unordered_set<image_t>& active_image_ids) {
  for (const auto& [pair_id, edge] : edges_) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    if (!active_image_ids.count(image_id1) ||
        !active_image_ids.count(image_id2)) {
      SetInvalidEdge(pair_id);
    }
  }
}

}  // namespace glomap

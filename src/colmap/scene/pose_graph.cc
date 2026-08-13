#include "colmap/scene/pose_graph.h"

#include "colmap/math/connected_components.h"
#include "colmap/util/hash_containers.h"

namespace colmap {
namespace {

struct FrameGraph {
  FlatHashSet<frame_t> nodes;
  std::vector<std::pair<frame_t, frame_t>> edges;
};

FrameGraph BuildFrameGraph(const PoseGraph& pose_graph,
                           const Reconstruction& reconstruction,
                           const bool filter_unregistered) {
  FrameGraph graph;
  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const frame_t frame_id1 = reconstruction.Image(image_id1).FrameId();
    const frame_t frame_id2 = reconstruction.Image(image_id2).FrameId();

    if (filter_unregistered && (!reconstruction.Frame(frame_id1).HasPose() ||
                                !reconstruction.Frame(frame_id2).HasPose())) {
      continue;
    }

    graph.nodes.insert(frame_id1);
    graph.nodes.insert(frame_id2);
    graph.edges.emplace_back(frame_id1, frame_id2);
  }
  return graph;
}

}  // namespace

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

std::vector<FlatHashSet<frame_t>> PoseGraph::ComputeConnectedFrameComponents(
    const Reconstruction& reconstruction, bool filter_unregistered) const {
  FrameGraph graph =
      BuildFrameGraph(*this, reconstruction, filter_unregistered);
  if (graph.nodes.empty()) {
    return {};
  }

  std::vector<std::vector<frame_t>> components =
      FindConnectedComponents(graph.nodes, graph.edges);

  std::sort(components.begin(),
            components.end(),
            [](const std::vector<frame_t>& a, const std::vector<frame_t>& b) {
              return a.size() > b.size();
            });

  std::vector<FlatHashSet<frame_t>> result;
  result.reserve(components.size());
  for (auto& component : components) {
    result.emplace_back(component.begin(), component.end());
  }
  return result;
}

std::vector<FlatHashSet<image_t>> PoseGraph::ComputeConnectedComponentImageIds(
    const Reconstruction& reconstruction, bool filter_unregistered) const {
  const std::vector<FlatHashSet<frame_t>> frame_components =
      ComputeConnectedFrameComponents(reconstruction, filter_unregistered);

  FlatHashMap<frame_t, int> frame_to_component;
  for (int comp = 0; comp < static_cast<int>(frame_components.size()); ++comp) {
    for (const frame_t frame_id : frame_components[comp]) {
      frame_to_component[frame_id] = comp;
    }
  }

  std::vector<FlatHashSet<image_t>> image_ids(frame_components.size());
  for (const auto& [image_id, image] : reconstruction.Images()) {
    const auto it = frame_to_component.find(image.FrameId());
    if (it != frame_to_component.end()) {
      image_ids[it->second].insert(image_id);
    }
  }
  return image_ids;
}

FlatHashSet<frame_t> PoseGraph::ComputeLargestConnectedFrameComponent(
    const Reconstruction& reconstruction, bool filter_unregistered) const {
  FrameGraph graph =
      BuildFrameGraph(*this, reconstruction, filter_unregistered);
  if (graph.nodes.empty()) {
    return {};
  }
  const std::vector<frame_t> largest_component =
      FindLargestConnectedComponent(graph.nodes, graph.edges);
  return {largest_component.begin(), largest_component.end()};
}

void PoseGraph::InvalidatePairsOutsideActiveImageIds(
    const FlatHashSet<image_t>& active_image_ids) {
  for (const auto& [pair_id, edge] : edges_) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (!active_image_ids.count(image_id1) ||
        !active_image_ids.count(image_id2)) {
      SetInvalidEdge(pair_id);
    }
  }
}

int PoseGraph::MarkConnectedComponents(const Reconstruction& reconstruction,
                                       NodeHashMap<frame_t, int>& cluster_ids,
                                       int min_num_images) const {
  const std::vector<FlatHashSet<frame_t>> connected_components =
      ComputeConnectedFrameComponents(reconstruction,
                                      /*filter_unregistered=*/false);

  // Clear and populate cluster_ids output parameter
  cluster_ids.clear();
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    cluster_ids[frame_id] = -1;
  }

  int comp = 0;
  for (; comp < static_cast<int>(connected_components.size()); ++comp) {
    if (static_cast<int>(connected_components[comp].size()) < min_num_images) {
      break;
    }
    for (const frame_t frame_id : connected_components[comp]) {
      cluster_ids[frame_id] = comp;
    }
  }

  return comp;
}

}  // namespace colmap

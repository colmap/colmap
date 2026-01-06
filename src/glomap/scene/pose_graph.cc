#include "glomap/scene/pose_graph.h"

#include "colmap/math/connected_components.h"

namespace glomap {

void PoseGraph::Load(const colmap::DatabaseCache& cache) {
  const auto corr_graph = cache.CorrespondenceGraph();

  for (const auto& [pair_id, cam2_from_cam1] : cache.RelativePoses()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);

    Edge edge;
    edge.cam2_from_cam1 = cam2_from_cam1;
    edge.inlier_matches =
        corr_graph->FindCorrespondencesBetweenImages(image_id1, image_id2);

    AddEdge(image_id1, image_id2, std::move(edge));
  }

  LOG(INFO) << "Loaded " << edges_.size() << " edges into pose graph";
}

int PoseGraph::KeepLargestConnectedComponents(
    colmap::Reconstruction& reconstruction) {
  std::unordered_set<frame_t> nodes;
  std::vector<std::pair<frame_t, frame_t>> graph_edges;
  for (const auto& [pair_id, edge] : ValidEdges()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    const frame_t frame_id1 = reconstruction.Image(image_id1).FrameId();
    const frame_t frame_id2 = reconstruction.Image(image_id2).FrameId();
    nodes.insert(frame_id1);
    nodes.insert(frame_id2);
    graph_edges.emplace_back(frame_id1, frame_id2);
  }

  if (nodes.empty()) {
    return 0;
  }

  const std::vector<frame_t> largest_component_vec =
      colmap::FindLargestConnectedComponent(nodes, graph_edges);
  const std::unordered_set<frame_t> largest_component(
      largest_component_vec.begin(), largest_component_vec.end());

  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (largest_component.count(frame_id) == 0 && frame.HasPose()) {
      reconstruction.DeRegisterFrame(frame_id);
    }
  }

  for (const auto& [pair_id, edge] : edges_) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    if (!reconstruction.Image(image_id1).HasPose() ||
        !reconstruction.Image(image_id2).HasPose()) {
      SetInvalidEdge(pair_id);
    }
  }

  return static_cast<int>(largest_component.size());
}

int PoseGraph::MarkConnectedComponents(
    const colmap::Reconstruction& reconstruction,
    std::unordered_map<frame_t, int>& cluster_ids,
    int min_num_images) {
  std::unordered_set<frame_t> nodes;
  std::vector<std::pair<frame_t, frame_t>> graph_edges;
  for (const auto& [pair_id, edge] : ValidEdges()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    const frame_t frame_id1 = reconstruction.Image(image_id1).FrameId();
    const frame_t frame_id2 = reconstruction.Image(image_id2).FrameId();
    nodes.insert(frame_id1);
    nodes.insert(frame_id2);
    graph_edges.emplace_back(frame_id1, frame_id2);
  }

  const std::vector<std::vector<frame_t>> connected_components =
      colmap::FindConnectedComponents(nodes, graph_edges);
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

std::unordered_map<image_t, std::unordered_set<image_t>>
PoseGraph::CreateImageAdjacencyList() const {
  std::unordered_map<image_t, std::unordered_set<image_t>> adjacency_list;
  for (const auto& [pair_id, edge] : ValidEdges()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    adjacency_list[image_id1].insert(image_id2);
    adjacency_list[image_id2].insert(image_id1);
  }
  return adjacency_list;
}

void PoseGraph::FilterByRelativeRotation(
    const colmap::Reconstruction& reconstruction, double max_angle_deg) {
  const double max_angle_rad = colmap::DegToRad(max_angle_deg);
  int num_invalid = 0;
  for (const auto& [pair_id, edge] : ValidEdges()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    const Image& image1 = reconstruction.Image(image_id1);
    const Image& image2 = reconstruction.Image(image_id2);

    if (!image1.HasPose() || !image2.HasPose()) {
      continue;
    }

    const Eigen::Quaterniond cam2_from_cam1 =
        image2.CamFromWorld().rotation *
        image1.CamFromWorld().rotation.inverse();
    if (cam2_from_cam1.angularDistance(edge.cam2_from_cam1.rotation) >
        max_angle_rad) {
      SetInvalidEdge(pair_id);
      num_invalid++;
    }
  }

  LOG(INFO) << "Marked " << num_invalid
            << " image pairs as invalid with relative rotation error > "
            << max_angle_deg << " degrees";
}

}  // namespace glomap

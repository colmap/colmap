#include "glomap/scene/view_graph.h"

#include <queue>

namespace glomap {
namespace {

void BreadthFirstSearch(
    const std::unordered_map<frame_t, std::unordered_set<frame_t>>&
        adjacency_list,
    image_t root,
    std::unordered_map<image_t, bool>& visited,
    std::unordered_set<image_t>& component) {
  std::queue<image_t> queue;
  queue.push(root);
  visited[root] = true;
  component.insert(root);

  while (!queue.empty()) {
    const image_t curr = queue.front();
    queue.pop();

    for (const image_t neighbor : adjacency_list.at(curr)) {
      if (!visited[neighbor]) {
        queue.push(neighbor);
        visited[neighbor] = true;
        component.insert(neighbor);
      }
    }
  }
}

std::vector<std::unordered_set<frame_t>> FindConnectedComponents(
    const std::unordered_map<frame_t, std::unordered_set<frame_t>>&
        adjacency_list) {
  std::vector<std::unordered_set<frame_t>> connected_components;
  std::unordered_map<frame_t, bool> visited;
  visited.reserve(adjacency_list.size());
  for (const auto& [frame_id, neighbors] : adjacency_list) {
    visited[frame_id] = false;
  }

  for (auto& [frame_id, _] : adjacency_list) {
    if (!visited[frame_id]) {
      std::unordered_set<frame_t> component;
      BreadthFirstSearch(adjacency_list, frame_id, visited, component);
      connected_components.push_back(std::move(component));
    }
  }

  return connected_components;
}

}  // namespace

int ViewGraph::KeepLargestConnectedComponents(
    std::unordered_map<frame_t, Frame>& frames,
    const std::unordered_map<image_t, Image>& images) {
  const std::vector<std::unordered_set<frame_t>> connected_components =
      FindConnectedComponents(CreateFrameAdjacencyList(images));

  int max_comp = -1;
  int max_num_reg_images = 0;
  for (int comp = 0; comp < connected_components.size(); ++comp) {
    if (connected_components[comp].size() > max_num_reg_images) {
      int num_reg_images = 0;
      for (auto& [image_id, image] : images) {
        if (image.HasPose()) {
          ++num_reg_images;
        }
      }
      if (num_reg_images > max_num_reg_images)
        max_num_reg_images = connected_components[comp].size();
      max_comp = comp;
    }
  }

  if (max_comp == -1) {
    return 0;
  }

  const std::unordered_set<frame_t>& largest_component =
      connected_components[max_comp];
  for (auto& [frame_id, frame] : frames) {
    if (largest_component.count(frame_id) == 0) {
      frame.ResetPose();
    }
  }

  for (auto& [pair_id, image_pair] : image_pairs) {
    if (!images.at(image_pair.image_id1).HasPose() ||
        !images.at(image_pair.image_id2).HasPose()) {
      image_pair.is_valid = false;
    }
  }

  return max_num_reg_images;
}

int ViewGraph::MarkConnectedComponents(
    const std::unordered_map<frame_t, Frame>& frames,
    const std::unordered_map<image_t, Image>& images,
    int min_num_images) {
  const std::vector<std::unordered_set<frame_t>> connected_components =
      FindConnectedComponents(CreateFrameAdjacencyList(images));
  const int num_comp = connected_components.size();

  std::vector<std::pair<int, int>> comp_num_images(num_comp);
  for (int comp = 0; comp < num_comp; comp++) {
    comp_num_images[comp] =
        std::make_pair(connected_components[comp].size(), comp);
  }
  std::sort(comp_num_images.begin(), comp_num_images.end(), std::greater<>());

  // Clear and populate cluster_ids member
  cluster_ids.clear();
  for (const auto& [frame_id, frame] : frames) {
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
ViewGraph::CreateImageAdjacencyList() const {
  std::unordered_map<image_t, std::unordered_set<image_t>> adjacency_list;
  for (const auto& [_, image_pair] : image_pairs) {
    if (image_pair.is_valid) {
      adjacency_list[image_pair.image_id1].insert(image_pair.image_id2);
      adjacency_list[image_pair.image_id2].insert(image_pair.image_id1);
    }
  }
  return adjacency_list;
}

std::unordered_map<frame_t, std::unordered_set<frame_t>>
ViewGraph::CreateFrameAdjacencyList(
    const std::unordered_map<image_t, Image>& images) const {
  std::unordered_map<frame_t, std::unordered_set<frame_t>> adjacency_list;
  for (const auto& [_, image_pair] : image_pairs) {
    if (image_pair.is_valid) {
      const frame_t frame_id1 = images.at(image_pair.image_id1).FrameId();
      const frame_t frame_id2 = images.at(image_pair.image_id2).FrameId();
      adjacency_list[frame_id1].insert(frame_id2);
      adjacency_list[frame_id2].insert(frame_id1);
    }
  }
  return adjacency_list;
}

}  // namespace glomap

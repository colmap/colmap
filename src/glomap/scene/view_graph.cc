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
    colmap::Reconstruction& reconstruction) {
  const std::vector<std::unordered_set<frame_t>> connected_components =
      FindConnectedComponents(
          CreateFrameAdjacencyList(reconstruction.Images()));

  int max_comp = -1;
  int max_num_reg_images = 0;
  for (int comp = 0; comp < connected_components.size(); ++comp) {
    if (connected_components[comp].size() > max_num_reg_images) {
      int num_reg_images = 0;
      for (const auto& [image_id, image] : reconstruction.Images()) {
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
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (largest_component.count(frame_id) == 0 && frame.HasPose()) {
      reconstruction.DeRegisterFrame(frame_id);
    }
  }

  for (const auto& [pair_id, image_pair] : image_pairs) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    if (!reconstruction.Image(image_id1).HasPose() ||
        !reconstruction.Image(image_id2).HasPose()) {
      SetToInvalid(pair_id);
    }
  }

  return max_num_reg_images;
}

int ViewGraph::MarkConnectedComponents(
    const colmap::Reconstruction& reconstruction,
    std::unordered_map<frame_t, int>& cluster_ids,
    int min_num_images) {
  const std::vector<std::unordered_set<frame_t>> connected_components =
      FindConnectedComponents(
          CreateFrameAdjacencyList(reconstruction.Images()));
  const int num_comp = connected_components.size();

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
ViewGraph::CreateImageAdjacencyList() const {
  std::unordered_map<image_t, std::unordered_set<image_t>> adjacency_list;
  for (const auto& [pair_id, image_pair] : ValidImagePairs()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    adjacency_list[image_id1].insert(image_id2);
    adjacency_list[image_id2].insert(image_id1);
  }
  return adjacency_list;
}

std::unordered_map<frame_t, std::unordered_set<frame_t>>
ViewGraph::CreateFrameAdjacencyList(
    const std::unordered_map<image_t, colmap::Image>& images) const {
  std::unordered_map<frame_t, std::unordered_set<frame_t>> adjacency_list;
  for (const auto& [pair_id, image_pair] : ValidImagePairs()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    const frame_t frame_id1 = images.at(image_id1).FrameId();
    const frame_t frame_id2 = images.at(image_id2).FrameId();
    adjacency_list[frame_id1].insert(frame_id2);
    adjacency_list[frame_id2].insert(frame_id1);
  }
  return adjacency_list;
}

void ViewGraph::FilterByRelativeRotation(
    const colmap::Reconstruction& reconstruction, double max_angle_deg) {
  const double max_angle_rad = colmap::DegToRad(max_angle_deg);
  int num_invalid = 0;
  for (const auto& [pair_id, image_pair] : ValidImagePairs()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    const Image& image1 = reconstruction.Image(image_id1);
    const Image& image2 = reconstruction.Image(image_id2);

    if (!image1.HasPose() || !image2.HasPose()) {
      continue;
    }

    const Eigen::Quaterniond cam2_from_cam1 =
        image2.CamFromWorld().rotation *
        image1.CamFromWorld().rotation.inverse();
    if (cam2_from_cam1.angularDistance(image_pair.cam2_from_cam1.rotation) >
        max_angle_rad) {
      SetToInvalid(pair_id);
      num_invalid++;
    }
  }

  LOG(INFO) << "Marked " << num_invalid
            << " image pairs as invalid with relative rotation error > "
            << max_angle_deg << " degrees";
}

void ViewGraph::FilterByNumInliers(int min_num_inliers) {
  int num_invalid = 0;
  for (const auto& [pair_id, image_pair] : ValidImagePairs()) {
    if (image_pair.inliers.size() < min_num_inliers) {
      SetToInvalid(pair_id);
      num_invalid++;
    }
  }

  LOG(INFO) << "Marked " << num_invalid
            << " image pairs as invalid with inlier count < "
            << min_num_inliers;
}

void ViewGraph::FilterByInlierRatio(double min_inlier_ratio) {
  int num_invalid = 0;
  for (const auto& [pair_id, image_pair] : ValidImagePairs()) {
    const double inlier_ratio = image_pair.inliers.size() /
                                static_cast<double>(image_pair.matches.rows());
    if (inlier_ratio < min_inlier_ratio) {
      SetToInvalid(pair_id);
      num_invalid++;
    }
  }

  LOG(INFO) << "Marked " << num_invalid
            << " image pairs as invalid with inlier ratio < "
            << min_inlier_ratio;
}

bool ViewGraph::IsValid(image_pair_t pair_id) const {
  return invalid_pairs_.find(pair_id) == invalid_pairs_.end();
}

void ViewGraph::SetToValid(image_pair_t pair_id) {
  invalid_pairs_.erase(pair_id);
}

void ViewGraph::SetToInvalid(image_pair_t pair_id) {
  invalid_pairs_.insert(pair_id);
}

ImagePair& ViewGraph::AddImagePair(image_t image_id1,
                                   image_t image_id2,
                                   ImagePair image_pair) {
  if (colmap::SwapImagePair(image_id1, image_id2)) {
    image_pair.Invert();
  }
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  auto [it, inserted] = image_pairs.emplace(pair_id, std::move(image_pair));
  if (!inserted) {
    throw std::runtime_error(
        "Image pair already exists: " + std::to_string(image_id1) + ", " +
        std::to_string(image_id2));
  }
  return it->second;
}

bool ViewGraph::HasImagePair(image_t image_id1, image_t image_id2) const {
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  return image_pairs.find(pair_id) != image_pairs.end();
}

std::pair<ImagePair&, bool> ViewGraph::Pair(image_t image_id1,
                                            image_t image_id2) {
  const bool swapped = colmap::SwapImagePair(image_id1, image_id2);
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  return {image_pairs.at(pair_id), swapped};
}

std::pair<const ImagePair&, bool> ViewGraph::Pair(image_t image_id1,
                                                  image_t image_id2) const {
  const bool swapped = colmap::SwapImagePair(image_id1, image_id2);
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  return {image_pairs.at(pair_id), swapped};
}

ImagePair ViewGraph::GetImagePair(image_t image_id1, image_t image_id2) const {
  const bool swapped = colmap::SwapImagePair(image_id1, image_id2);
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  ImagePair result = image_pairs.at(pair_id);
  if (swapped) {
    result.Invert();
  }
  return result;
}

bool ViewGraph::DeleteImagePair(image_t image_id1, image_t image_id2) {
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  return image_pairs.erase(pair_id) > 0;
}

void ViewGraph::UpdateImagePair(image_t image_id1,
                                image_t image_id2,
                                ImagePair image_pair) {
  if (colmap::SwapImagePair(image_id1, image_id2)) {
    image_pair.Invert();
  }
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  auto it = image_pairs.find(pair_id);
  if (it == image_pairs.end()) {
    throw std::runtime_error(
        "Image pair does not exist: " + std::to_string(image_id1) + ", " +
        std::to_string(image_id2));
  }
  it->second = std::move(image_pair);
}

}  // namespace glomap

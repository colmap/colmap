#include "glomap/scene/view_graph.h"

#include "colmap/math/connected_components.h"
#include "colmap/scene/two_view_geometry.h"

namespace glomap {

void ViewGraph::LoadFromDatabase(const colmap::Database& database,
                                 bool allow_duplicate) {
  auto all_matches = database.ReadAllMatches();
  size_t invalid_count = 0;

  for (auto& [pair_id, feature_matches] : all_matches) {
    auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);

    const bool duplicate = HasImagePair(image_id1, image_id2);
    if (duplicate) {
      if (!allow_duplicate) {
        LOG(FATAL_THROW) << "Duplicate image pair in database: " << image_id1
                         << ", " << image_id2;
      }
      LOG(WARNING) << "Duplicate image pair in database: " << image_id1 << ", "
                   << image_id2;
    }

    // Build the image pair from TwoViewGeometry
    struct ImagePair image_pair;
    static_cast<colmap::TwoViewGeometry&>(image_pair) =
        database.ReadTwoViewGeometry(image_id1, image_id2);

    const bool is_invalid =
        image_pair.config == colmap::TwoViewGeometry::UNDEFINED ||
        image_pair.config == colmap::TwoViewGeometry::DEGENERATE ||
        image_pair.config == colmap::TwoViewGeometry::WATERMARK ||
        image_pair.config == colmap::TwoViewGeometry::MULTIPLE;

    if (is_invalid) {
      invalid_count++;
    } else {
      // Collect the matches
      image_pair.matches = Eigen::MatrixXi(feature_matches.size(), 2);

      size_t count = 0;
      for (int i = 0; i < feature_matches.size(); i++) {
        colmap::point2D_t point2D_idx1 = feature_matches[i].point2D_idx1;
        colmap::point2D_t point2D_idx2 = feature_matches[i].point2D_idx2;
        if (point2D_idx1 != colmap::kInvalidPoint2DIdx &&
            point2D_idx2 != colmap::kInvalidPoint2DIdx) {
          image_pair.matches.row(count) << point2D_idx1, point2D_idx2;
          count++;
        }
      }
      image_pair.matches.conservativeResize(count, 2);
    }

    if (duplicate) {
      UpdateImagePair(image_id1, image_id2, std::move(image_pair));
    } else {
      AddImagePair(image_id1, image_id2, std::move(image_pair));
    }

    if (is_invalid) {
      SetInvalidImagePair(colmap::ImagePairToPairId(image_id1, image_id2));
    }
  }
  LOG(INFO) << "Loaded " << all_matches.size() << " image pairs, "
            << invalid_count << " invalid";
}

int ViewGraph::KeepLargestConnectedComponents(
    colmap::Reconstruction& reconstruction) {
  std::unordered_set<frame_t> nodes;
  std::vector<std::pair<frame_t, frame_t>> edges;
  for (const auto& [pair_id, image_pair] : ValidImagePairs()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    const frame_t frame_id1 = reconstruction.Image(image_id1).FrameId();
    const frame_t frame_id2 = reconstruction.Image(image_id2).FrameId();
    nodes.insert(frame_id1);
    nodes.insert(frame_id2);
    edges.emplace_back(frame_id1, frame_id2);
  }

  if (nodes.empty()) {
    return 0;
  }

  const std::vector<frame_t> largest_component_vec =
      colmap::FindLargestConnectedComponent(nodes, edges);
  const std::unordered_set<frame_t> largest_component(
      largest_component_vec.begin(), largest_component_vec.end());

  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (largest_component.count(frame_id) == 0 && frame.HasPose()) {
      reconstruction.DeRegisterFrame(frame_id);
    }
  }

  for (const auto& [pair_id, image_pair] : image_pairs_) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    if (!reconstruction.Image(image_id1).HasPose() ||
        !reconstruction.Image(image_id2).HasPose()) {
      SetInvalidImagePair(pair_id);
    }
  }

  return static_cast<int>(largest_component.size());
}

int ViewGraph::MarkConnectedComponents(
    const colmap::Reconstruction& reconstruction,
    std::unordered_map<frame_t, int>& cluster_ids,
    int min_num_images) {
  std::unordered_set<frame_t> nodes;
  std::vector<std::pair<frame_t, frame_t>> edges;
  for (const auto& [pair_id, image_pair] : ValidImagePairs()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    const frame_t frame_id1 = reconstruction.Image(image_id1).FrameId();
    const frame_t frame_id2 = reconstruction.Image(image_id2).FrameId();
    nodes.insert(frame_id1);
    nodes.insert(frame_id2);
    edges.emplace_back(frame_id1, frame_id2);
  }

  const std::vector<std::vector<frame_t>> connected_components =
      colmap::FindConnectedComponents(nodes, edges);
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
ViewGraph::CreateImageAdjacencyList() const {
  std::unordered_map<image_t, std::unordered_set<image_t>> adjacency_list;
  for (const auto& [pair_id, image_pair] : ValidImagePairs()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    adjacency_list[image_id1].insert(image_id2);
    adjacency_list[image_id2].insert(image_id1);
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
    THROW_CHECK(image_pair.cam2_from_cam1.has_value());

    const Eigen::Quaterniond cam2_from_cam1 =
        image2.CamFromWorld().rotation *
        image1.CamFromWorld().rotation.inverse();
    if (cam2_from_cam1.angularDistance(image_pair.cam2_from_cam1->rotation) >
        max_angle_rad) {
      SetInvalidImagePair(pair_id);
      num_invalid++;
    }
  }

  LOG(INFO) << "Marked " << num_invalid
            << " image pairs as invalid with relative rotation error > "
            << max_angle_deg << " degrees";
}

}  // namespace glomap

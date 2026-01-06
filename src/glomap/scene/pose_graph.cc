#include "glomap/scene/pose_graph.h"

#include "colmap/estimators/two_view_geometry.h"
#include "colmap/math/connected_components.h"
#include "colmap/scene/two_view_geometry.h"

namespace glomap {
namespace {

std::vector<Eigen::Vector2d> FeatureKeypointsToPointsVector(
    const colmap::FeatureKeypoints& keypoints) {
  std::vector<Eigen::Vector2d> points(keypoints.size());
  for (size_t i = 0; i < keypoints.size(); i++) {
    points[i] = Eigen::Vector2d(keypoints[i].x, keypoints[i].y);
  }
  return points;
}

}  // namespace

void PoseGraph::LoadFromDatabase(const colmap::Database& database,
                                 bool allow_duplicate) {
  // TODO: Move relative pose decomposition logic to the top level.
  auto all_matches = database.ReadAllMatches();

  // Read cameras and images upfront for pose decomposition.
  std::unordered_map<colmap::camera_t, colmap::Camera> cameras;
  for (auto& camera : database.ReadAllCameras()) {
    cameras.emplace(camera.camera_id, std::move(camera));
  }

  std::unordered_map<image_t, colmap::Image> images;
  for (auto& image : database.ReadAllImages()) {
    images.emplace(image.ImageId(), std::move(image));
  }

  size_t invalid_count = 0;
  size_t decompose_count = 0;
  size_t decompose_failed_count = 0;

  for (auto& [pair_id, feature_matches] : all_matches) {
    auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);

    const bool duplicate = HasEdge(image_id1, image_id2);
    if (duplicate) {
      if (!allow_duplicate) {
        LOG(FATAL_THROW) << "Duplicate image pair in database: " << image_id1
                         << ", " << image_id2;
      }
      LOG(WARNING) << "Duplicate image pair in database: " << image_id1 << ", "
                   << image_id2;
    }

    // Read TwoViewGeometry from database and extract needed fields.
    colmap::TwoViewGeometry two_view_geom =
        database.ReadTwoViewGeometry(image_id1, image_id2);

    const bool is_invalid =
        two_view_geom.config == colmap::TwoViewGeometry::UNDEFINED ||
        two_view_geom.config == colmap::TwoViewGeometry::DEGENERATE ||
        two_view_geom.config == colmap::TwoViewGeometry::WATERMARK ||
        two_view_geom.config == colmap::TwoViewGeometry::MULTIPLE;

    if (is_invalid) {
      invalid_count++;
    } else {
      // Decompose relative pose if not already present.
      if (!two_view_geom.cam2_from_cam1.has_value()) {
        const colmap::Image& image1 = images.at(image_id1);
        const colmap::Image& image2 = images.at(image_id2);
        const colmap::Camera& camera1 = cameras.at(image1.CameraId());
        const colmap::Camera& camera2 = cameras.at(image2.CameraId());

        const std::vector<Eigen::Vector2d> points1 =
            FeatureKeypointsToPointsVector(database.ReadKeypoints(image_id1));
        const std::vector<Eigen::Vector2d> points2 =
            FeatureKeypointsToPointsVector(database.ReadKeypoints(image_id2));

        decompose_count++;
        const bool success = colmap::EstimateTwoViewGeometryPose(
            camera1, points1, camera2, points2, &two_view_geom);

        if (success && two_view_geom.cam2_from_cam1.has_value()) {
          const double norm = two_view_geom.cam2_from_cam1->translation.norm();
          if (norm > 1e-12) {
            two_view_geom.cam2_from_cam1->translation /= norm;
          }
        } else {
          decompose_failed_count++;
        }
      }
    }

    // Skip pairs that don't have a valid pose.
    if (!two_view_geom.cam2_from_cam1.has_value()) {
      invalid_count++;
      continue;
    }

    // Build Edge from TwoViewGeometry.
    Edge new_edge;
    new_edge.cam2_from_cam1 = *two_view_geom.cam2_from_cam1;
    new_edge.inlier_matches = std::move(two_view_geom.inlier_matches);

    if (duplicate) {
      UpdateEdge(image_id1, image_id2, std::move(new_edge));
    } else {
      AddEdge(image_id1, image_id2, std::move(new_edge));
    }
  }

  LOG(INFO) << "Loaded " << all_matches.size() << " image pairs, "
            << invalid_count << " invalid";
  if (decompose_count > 0) {
    LOG(INFO) << "Decomposed relative pose for " << decompose_count
              << " pairs, " << decompose_failed_count << " failed";
  }
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

int PoseGraph::MarkConnectedComponents(
    const colmap::Reconstruction& reconstruction,
    std::unordered_map<frame_t, int>& cluster_ids,
    int min_num_images) const {
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

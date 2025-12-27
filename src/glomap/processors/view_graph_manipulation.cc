#include "view_graph_manipulation.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/math/union_find.h"
#include "colmap/util/threading.h"

namespace glomap {

image_t ViewGraphManipulator::EstablishStrongClusters(
    ViewGraph& view_graph,
    colmap::Reconstruction& reconstruction,
    std::unordered_map<frame_t, int>& cluster_ids,
    StrongClusterCriteria criteria,
    double min_thres,
    int min_num_images) {
  view_graph.KeepLargestConnectedComponents(reconstruction);

  // Construct the initial cluster by keeping the pairs with weight > min_thres
  colmap::UnionFind<image_pair_t> uf;
  uf.Reserve(reconstruction.NumFrames());
  // Go through the edges, and add the edge with weight > min_thres
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;

    bool status = false;
    status = status ||
             (criteria == INLIER_NUM && image_pair.inliers.size() > min_thres);
    status = status || (criteria == WEIGHT && image_pair.weight > min_thres);
    if (status) {
      uf.Union(
          image_pair_t(reconstruction.Image(image_pair.image_id1).FrameId()),
          image_pair_t(reconstruction.Image(image_pair.image_id2).FrameId()));
    }
  }

  // For every two connected components, we check the number of slightly weaker
  // pairs (> 0.75 min_thres) between them Two clusters are concatenated if the
  // number of such pairs is larger than a threshold (2)
  bool status = true;
  int iteration = 0;
  while (status) {
    status = false;
    iteration++;

    if (iteration > 10) {
      break;
    }

    std::unordered_map<image_pair_t, std::unordered_map<image_pair_t, int>>
        num_pairs;
    for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
      if (!image_pair.is_valid) continue;

      // If the number of inliers < 0.75 of the threshold, skip
      bool status = false;
      status = status || (criteria == INLIER_NUM &&
                          image_pair.inliers.size() < 0.75 * min_thres);
      status = status ||
               (criteria == WEIGHT && image_pair.weight < 0.75 * min_thres);
      if (status) continue;

      image_t image_id1 = image_pair.image_id1;
      image_t image_id2 = image_pair.image_id2;

      image_pair_t root1 =
          uf.Find(image_pair_t(reconstruction.Image(image_id1).FrameId()));
      image_pair_t root2 =
          uf.Find(image_pair_t(reconstruction.Image(image_id2).FrameId()));

      if (root1 == root2) {
        continue;
      }
      if (num_pairs.find(root1) == num_pairs.end())
        num_pairs.insert(
            std::make_pair(root1, std::unordered_map<image_pair_t, int>()));
      if (num_pairs.find(root2) == num_pairs.end())
        num_pairs.insert(
            std::make_pair(root2, std::unordered_map<image_pair_t, int>()));

      num_pairs[root1][root2]++;
      num_pairs[root2][root1]++;
    }
    // Connect the clusters progressively. If two clusters have more than 3
    // pairs, then connect them
    for (auto& [root1, counter] : num_pairs) {
      for (auto& [root2, count] : counter) {
        if (root1 <= root2) continue;

        if (count >= 2) {
          status = true;
          uf.Union(root1, root2);
        }
      }
    }
  }

  for (auto& [image_pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;

    frame_t frame_id1 = reconstruction.Image(image_pair.image_id1).FrameId();
    frame_t frame_id2 = reconstruction.Image(image_pair.image_id2).FrameId();

    if (uf.Find(image_pair_t(frame_id1)) != uf.Find(image_pair_t(frame_id2))) {
      image_pair.is_valid = false;
    }
  }
  int num_comp =
      view_graph.MarkConnectedComponents(reconstruction, cluster_ids);

  LOG(INFO) << "Clustering take " << iteration << " iterations. "
            << "Images are grouped into " << num_comp
            << " clusters after strong-clustering";

  return num_comp;
}

void ViewGraphManipulator::UpdateImagePairsConfig(
    ViewGraph& view_graph, const colmap::Reconstruction& reconstruction) {
  // For each camera, check the number of times that the camera is involved in a
  // pair with configuration 2 First: the total occurence; second: the number of
  // pairs with configuration 2
  std::unordered_map<camera_t, std::pair<int, int>> camera_counter;
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;

    const camera_t camera_id1 =
        reconstruction.Image(image_pair.image_id1).CameraId();
    const camera_t camera_id2 =
        reconstruction.Image(image_pair.image_id2).CameraId();

    const colmap::Camera& camera1 = reconstruction.Camera(camera_id1);
    const colmap::Camera& camera2 = reconstruction.Camera(camera_id2);
    if (!camera1.has_prior_focal_length || !camera2.has_prior_focal_length)
      continue;

    if (image_pair.config == colmap::TwoViewGeometry::CALIBRATED) {
      camera_counter[camera_id1].first++;
      camera_counter[camera_id2].first++;
      camera_counter[camera_id1].second++;
      camera_counter[camera_id2].second++;
    } else if (image_pair.config == colmap::TwoViewGeometry::UNCALIBRATED) {
      camera_counter[camera_id1].first++;
      camera_counter[camera_id2].first++;
    }
  }

  // Check the ratio of valid and invalid relative pair, if the majority of the
  // pairs are valid, then set the camera to valid
  std::unordered_map<camera_t, bool> camera_validity;
  for (auto& [camera_id, counter] : camera_counter) {
    if (counter.second * 1. / counter.first > 0.5) {
      camera_validity[camera_id] = true;
    } else {
      camera_validity[camera_id] = false;
    }
  }

  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;
    if (image_pair.config != colmap::TwoViewGeometry::UNCALIBRATED) continue;

    const camera_t camera_id1 =
        reconstruction.Image(image_pair.image_id1).CameraId();
    const camera_t camera_id2 =
        reconstruction.Image(image_pair.image_id2).CameraId();

    const colmap::Camera& camera1 = reconstruction.Camera(camera_id1);
    const colmap::Camera& camera2 = reconstruction.Camera(camera_id2);

    if (camera_validity[camera_id1] && camera_validity[camera_id2]) {
      image_pair.config = colmap::TwoViewGeometry::CALIBRATED;
      image_pair.F = colmap::FundamentalFromEssentialMatrix(
          camera2.CalibrationMatrix(),
          colmap::EssentialMatrixFromPose(image_pair.cam2_from_cam1),
          camera1.CalibrationMatrix());
    }
  }
}

// Decompose the relative camera postion from the camera config
void ViewGraphManipulator::DecomposeRelPose(
    ViewGraph& view_graph, colmap::Reconstruction& reconstruction) {
  std::vector<image_pair_t> image_pair_ids;
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;
    const camera_t camera_id1 =
        reconstruction.Image(image_pair.image_id1).CameraId();
    const camera_t camera_id2 =
        reconstruction.Image(image_pair.image_id2).CameraId();
    if (!reconstruction.Camera(camera_id1).has_prior_focal_length ||
        !reconstruction.Camera(camera_id2).has_prior_focal_length)
      continue;
    image_pair_ids.push_back(pair_id);
  }

  const int64_t num_image_pairs = image_pair_ids.size();
  LOG(INFO) << "Decompose relative pose for " << num_image_pairs << " pairs";

  colmap::ThreadPool thread_pool(colmap::ThreadPool::kMaxNumThreads);
  for (int64_t idx = 0; idx < num_image_pairs; idx++) {
    thread_pool.AddTask([&, idx]() {
      ImagePair& image_pair = view_graph.image_pairs.at(image_pair_ids[idx]);
      const Image& image1 = reconstruction.Image(image_pair.image_id1);
      const Image& image2 = reconstruction.Image(image_pair.image_id2);

      const camera_t camera_id1 = image1.CameraId();
      const camera_t camera_id2 = image2.CameraId();
      const colmap::Camera& camera1 = reconstruction.Camera(camera_id1);
      const colmap::Camera& camera2 = reconstruction.Camera(camera_id2);

      // Use the two-view geometry to re-estimate the relative pose
      colmap::TwoViewGeometry two_view_geometry;
      two_view_geometry.E = image_pair.E;
      two_view_geometry.F = image_pair.F;
      two_view_geometry.H = image_pair.H;
      two_view_geometry.config = image_pair.config;

      std::vector<Eigen::Vector2d> points1(image1.NumPoints2D());
      for (colmap::point2D_t point2D_idx = 0;
           point2D_idx < image1.NumPoints2D();
           point2D_idx++) {
        points1[point2D_idx] = image1.Point2D(point2D_idx).xy;
      }
      std::vector<Eigen::Vector2d> points2(image2.NumPoints2D());
      for (colmap::point2D_t point2D_idx = 0;
           point2D_idx < image2.NumPoints2D();
           point2D_idx++) {
        points2[point2D_idx] = image2.Point2D(point2D_idx).xy;
      }

      colmap::EstimateTwoViewGeometryPose(
          camera1, points1, camera2, points2, &two_view_geometry);

      // if it planar, then use the estimated relative pose
      if (image_pair.config == colmap::TwoViewGeometry::PLANAR &&
          camera1.has_prior_focal_length && camera2.has_prior_focal_length) {
        image_pair.config = colmap::TwoViewGeometry::CALIBRATED;
        return;
      } else if (!(camera1.has_prior_focal_length &&
                   camera2.has_prior_focal_length))
        return;

      image_pair.config = two_view_geometry.config;
      image_pair.cam2_from_cam1 = two_view_geometry.cam2_from_cam1;

      if (image_pair.cam2_from_cam1.translation.norm() > 1e-12) {
        image_pair.cam2_from_cam1.translation =
            image_pair.cam2_from_cam1.translation.normalized();
      }
    });
  }

  thread_pool.Wait();

  size_t counter = 0;
  for (size_t idx = 0; idx < image_pair_ids.size(); idx++) {
    ImagePair& image_pair = view_graph.image_pairs.at(image_pair_ids[idx]);
    if (image_pair.config != colmap::TwoViewGeometry::CALIBRATED &&
        image_pair.config != colmap::TwoViewGeometry::PLANAR_OR_PANORAMIC)
      counter++;
  }
  LOG(INFO) << "Decompose relative pose done. " << counter
            << " pairs are pure rotation";
}

}  // namespace glomap

#include "view_graph_manipulation.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/util/threading.h"

namespace glomap {

void ViewGraphManipulator::UpdateImagePairsConfig(
    ViewGraph& view_graph, const colmap::Reconstruction& reconstruction) {
  // For each camera, check the number of times that the camera is involved in a
  // pair with configuration 2 First: the total occurence; second: the number of
  // pairs with configuration 2
  std::unordered_map<camera_t, std::pair<int, int>> camera_counter;
  for (const auto& [pair_id, image_pair] : view_graph.ValidImagePairs()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    const camera_t camera_id1 = reconstruction.Image(image_id1).CameraId();
    const camera_t camera_id2 = reconstruction.Image(image_id2).CameraId();

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

  for (auto& [pair_id, image_pair] : view_graph.ImagePairs()) {
    if (!view_graph.IsValid(pair_id)) continue;
    if (image_pair.config != colmap::TwoViewGeometry::UNCALIBRATED) continue;

    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    const camera_t camera_id1 = reconstruction.Image(image_id1).CameraId();
    const camera_t camera_id2 = reconstruction.Image(image_id2).CameraId();

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
    ViewGraph& view_graph,
    colmap::Reconstruction& reconstruction,
    int num_threads) {
  std::vector<image_pair_t> image_pair_ids;
  for (const auto& [pair_id, image_pair] : view_graph.ValidImagePairs()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    const camera_t camera_id1 = reconstruction.Image(image_id1).CameraId();
    const camera_t camera_id2 = reconstruction.Image(image_id2).CameraId();
    if (!reconstruction.Camera(camera_id1).has_prior_focal_length ||
        !reconstruction.Camera(camera_id2).has_prior_focal_length)
      continue;
    image_pair_ids.push_back(pair_id);
  }

  const int64_t num_image_pairs = image_pair_ids.size();
  LOG(INFO) << "Decompose relative pose for " << num_image_pairs << " pairs";

  colmap::ThreadPool thread_pool(colmap::GetEffectiveNumThreads(num_threads));
  for (int64_t idx = 0; idx < num_image_pairs; idx++) {
    thread_pool.AddTask([&, idx]() {
      const image_pair_t pair_id = image_pair_ids[idx];
      const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
      ImagePair& image_pair = view_graph.ImagePair(image_id1, image_id2).first;
      const Image& image1 = reconstruction.Image(image_id1);
      const Image& image2 = reconstruction.Image(image_id2);

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
    const auto [image_id1, image_id2] =
        colmap::PairIdToImagePair(image_pair_ids[idx]);
    const ImagePair& image_pair =
        view_graph.ImagePair(image_id1, image_id2).first;
    if (image_pair.config != colmap::TwoViewGeometry::CALIBRATED &&
        image_pair.config != colmap::TwoViewGeometry::PLANAR_OR_PANORAMIC)
      counter++;
  }
  LOG(INFO) << "Decompose relative pose done. " << counter
            << " pairs are pure rotation";
}

}  // namespace glomap

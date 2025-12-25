#include "glomap/io/colmap_converter.h"

#include "colmap/geometry/essential_matrix.h"

namespace glomap {

namespace {

// Helper to create a trivial rig for a camera if it's not already in a rig.
void EnsureCameraHasRig(colmap::Reconstruction& reconstruction,
                        camera_t camera_id,
                        std::unordered_map<camera_t, rig_t>& camera_to_rig,
                        rig_t& max_rig_id) {
  if (camera_to_rig.find(camera_id) != camera_to_rig.end()) {
    return;  // Camera already has a rig
  }

  // Create a new trivial rig for this camera
  Rig rig;
  rig.SetRigId(++max_rig_id);
  rig.AddRefSensor(reconstruction.Camera(camera_id).SensorId());
  reconstruction.AddRig(rig);
  camera_to_rig[camera_id] = rig.RigId();
}

// Helper to create a trivial frame for an image if it doesn't have one.
void EnsureImageHasFrame(
    colmap::Reconstruction& reconstruction,
    image_t image_id,
    const std::unordered_map<camera_t, rig_t>& camera_to_rig,
    frame_t& max_frame_id) {
  Image& image = reconstruction.Image(image_id);
  if (image.FrameId() != colmap::kInvalidFrameId) {
    return;  // Image already has a frame
  }

  frame_t frame_id = ++max_frame_id;
  rig_t rig_id = camera_to_rig.at(image.CameraId());

  Frame frame;
  frame.SetFrameId(frame_id);
  frame.SetRigId(rig_id);
  frame.AddDataId(image.DataId());
  frame.SetRigFromWorld(Rigid3d());
  reconstruction.AddFrame(frame);

  image.SetFrameId(frame_id);
  image.SetFramePtr(&reconstruction.Frame(frame_id));
}

}  // namespace

void ConvertDatabaseToGlomap(const colmap::Database& database,
                             colmap::Reconstruction& reconstruction,
                             ViewGraph& view_graph) {
  reconstruction = colmap::Reconstruction();
  view_graph.image_pairs.clear();

  // Add all cameras
  for (auto& camera : database.ReadAllCameras()) {
    reconstruction.AddCamera(std::move(camera));
  }

  // Add all rigs from database
  rig_t max_rig_id = 0;
  std::unordered_map<camera_t, rig_t> camera_to_rig;

  for (auto& rig : database.ReadAllRigs()) {
    max_rig_id = std::max(max_rig_id, rig.RigId());
    reconstruction.AddRig(rig);

    sensor_t sensor_id = rig.RefSensorId();
    if (sensor_id.type == SensorType::CAMERA) {
      camera_to_rig[sensor_id.id] = rig.RigId();
    }
    for (const auto& [non_ref_sensor_id, sensor_pose] : rig.NonRefSensors()) {
      if (non_ref_sensor_id.type == SensorType::CAMERA) {
        camera_to_rig[non_ref_sensor_id.id] = rig.RigId();
      }
    }
  }

  // Create trivial rigs for cameras not in any rig
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    EnsureCameraHasRig(reconstruction, camera_id, camera_to_rig, max_rig_id);
  }

  // Add all images
  for (auto& image : database.ReadAllImages()) {
    const colmap::FeatureKeypoints keypoints =
        database.ReadKeypoints(image.ImageId());
    const colmap::point2D_t num_points2D = keypoints.size();
    image.Points2D().resize(num_points2D);
    for (colmap::point2D_t point2D_idx = 0; point2D_idx < num_points2D;
         point2D_idx++) {
      image.Point2D(point2D_idx).xy =
          Eigen::Vector2d(keypoints[point2D_idx].x, keypoints[point2D_idx].y);
    }

    reconstruction.AddImage(std::move(image));
  }

  LOG(INFO) << "Read " << reconstruction.NumImages() << " images";

  // Add all frames from database
  frame_t max_frame_id = 0;
  for (auto& frame : database.ReadAllFrames()) {
    frame_t frame_id = frame.FrameId();
    if (frame_id == colmap::kInvalidFrameId) continue;
    max_frame_id = std::max(max_frame_id, frame_id);

    Frame glomap_frame;
    glomap_frame.SetFrameId(frame_id);
    glomap_frame.SetRigId(frame.RigId());
    glomap_frame.SetRigFromWorld(Rigid3d());

    for (auto data_id : frame.ImageIds()) {
      image_t image_id = data_id.id;
      glomap_frame.AddDataId(data_id);

      if (reconstruction.ExistsImage(image_id)) {
        reconstruction.Image(image_id).SetFrameId(frame_id);
      }
    }

    reconstruction.AddFrame(glomap_frame);

    for (auto data_id : frame.ImageIds()) {
      image_t image_id = data_id.id;
      if (reconstruction.ExistsImage(image_id)) {
        reconstruction.Image(image_id).SetFramePtr(
            &reconstruction.Frame(frame_id));
      }
    }
  }

  // Create trivial frames for images without frames
  for (auto& [image_id, image] : reconstruction.Images()) {
    EnsureImageHasFrame(reconstruction, image_id, camera_to_rig, max_frame_id);
  }

  // Build view graph from matches
  std::vector<std::pair<colmap::image_pair_t, colmap::FeatureMatches>>
      all_matches = database.ReadAllMatches();

  size_t invalid_count = 0;
  std::unordered_map<image_pair_t, ImagePair>& image_pairs =
      view_graph.image_pairs;

  for (size_t match_idx = 0; match_idx < all_matches.size(); match_idx++) {
    if ((match_idx + 1) % 1000 == 0 || match_idx == all_matches.size() - 1) {
      std::cout << "\r Loading Image Pair " << match_idx + 1 << " / "
                << all_matches.size() << std::flush;
    }

    // Read the image pair from COLMAP database
    colmap::image_pair_t pair_id = all_matches[match_idx].first;
    auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);

    colmap::FeatureMatches& feature_matches = all_matches[match_idx].second;

    // Initialize the image pair
    auto [it, inserted] = image_pairs.insert(
        std::make_pair(colmap::ImagePairToPairId(image_id1, image_id2),
                       ImagePair(image_id1, image_id2)));
    ImagePair& image_pair = it->second;

    colmap::TwoViewGeometry two_view =
        database.ReadTwoViewGeometry(image_id1, image_id2);

    // If the image is marked as invalid or watermark, then skip
    if (two_view.config == colmap::TwoViewGeometry::UNDEFINED ||
        two_view.config == colmap::TwoViewGeometry::DEGENERATE ||
        two_view.config == colmap::TwoViewGeometry::WATERMARK ||
        two_view.config == colmap::TwoViewGeometry::MULTIPLE) {
      image_pair.is_valid = false;
      invalid_count++;
      continue;
    }

    const Image& image1 = reconstruction.Image(image_id1);
    const Image& image2 = reconstruction.Image(image_id2);

    // Collect the fundamental matrices
    if (two_view.config == colmap::TwoViewGeometry::UNCALIBRATED) {
      image_pair.F = two_view.F;
    } else if (two_view.config == colmap::TwoViewGeometry::CALIBRATED) {
      image_pair.F = colmap::FundamentalFromEssentialMatrix(
          reconstruction.Camera(image2.CameraId()).CalibrationMatrix(),
          colmap::EssentialMatrixFromPose(image_pair.cam2_from_cam1),
          reconstruction.Camera(image1.CameraId()).CalibrationMatrix());
    } else if (two_view.config == colmap::TwoViewGeometry::PLANAR ||
               two_view.config == colmap::TwoViewGeometry::PANORAMIC ||
               two_view.config ==
                   colmap::TwoViewGeometry::PLANAR_OR_PANORAMIC) {
      image_pair.H = two_view.H;
      image_pair.F = two_view.F;
    }
    image_pair.config = two_view.config;

    // Collect the matches
    image_pair.matches = Eigen::MatrixXi(feature_matches.size(), 2);

    size_t count = 0;
    for (int i = 0; i < feature_matches.size(); i++) {
      colmap::point2D_t point2D_idx1 = feature_matches[i].point2D_idx1;
      colmap::point2D_t point2D_idx2 = feature_matches[i].point2D_idx2;
      if (point2D_idx1 != colmap::kInvalidPoint2DIdx &&
          point2D_idx2 != colmap::kInvalidPoint2DIdx) {
        if (point2D_idx1 >= image1.NumPoints2D() ||
            point2D_idx2 >= image2.NumPoints2D()) {
          continue;
        }
        image_pair.matches.row(count) << point2D_idx1, point2D_idx2;
        count++;
      }
    }
    image_pair.matches.conservativeResize(count, 2);
  }
  std::cout << '\n';
  LOG(INFO) << "Pairs read done. " << invalid_count << " / "
            << view_graph.image_pairs.size() << " are invalid";
}

colmap::Reconstruction ExtractCluster(
    const colmap::Reconstruction& reconstruction,
    const std::unordered_map<frame_t, int>& cluster_ids,
    int cluster_id) {
  // If no filtering needed, return a copy
  if (cluster_id == -1 || cluster_ids.empty()) {
    return reconstruction;
  }

  // Helper to get cluster id for a frame
  auto get_cluster_id = [&cluster_ids](frame_t frame_id) -> int {
    auto it = cluster_ids.find(frame_id);
    return it != cluster_ids.end() ? it->second : -1;
  };

  // Make a copy of the reconstruction
  colmap::Reconstruction filtered = reconstruction;

  // Collect frames to deregister (those not in this cluster)
  std::vector<frame_t> frames_to_deregister;
  for (const auto& [frame_id, frame] : filtered.Frames()) {
    if (!frame.HasPose() || get_cluster_id(frame_id) != cluster_id) {
      frames_to_deregister.push_back(frame_id);
    }
  }

  // Deregister frames not in this cluster
  // This also removes point observations from those frames' images
  for (frame_t frame_id : frames_to_deregister) {
    if (filtered.Frame(frame_id).HasPose()) {
      filtered.DeRegisterFrame(frame_id);
    }
  }

  filtered.UpdatePoint3DErrors();
  return filtered;
}

}  // namespace glomap

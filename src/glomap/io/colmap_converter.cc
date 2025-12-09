#include "glomap/io/colmap_converter.h"

#include "colmap/geometry/essential_matrix.h"

namespace glomap {

void ConvertGlomapToColmapImage(const Image& image,
                                colmap::Image& image_colmap,
                                bool keep_points) {
  image_colmap.SetImageId(image.image_id);
  image_colmap.SetCameraId(image.camera_id);
  image_colmap.SetName(image.file_name);
  image_colmap.SetFrameId(image.frame_id);

  if (keep_points) {
    image_colmap.SetPoints2D(image.features);
  }
}

void ConvertGlomapToColmap(
    const std::unordered_map<rig_t, Rig>& rigs,
    const std::unordered_map<camera_t, colmap::Camera>& cameras,
    const std::unordered_map<frame_t, Frame>& frames,
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<point3D_t, Point3D>& tracks,
    colmap::Reconstruction& reconstruction,
    int cluster_id,
    bool include_image_points) {
  // Clear the colmap reconstruction
  reconstruction = colmap::Reconstruction();

  // Add cameras
  for (const auto& [camera_id, camera] : cameras) {
    reconstruction.AddCamera(camera);
  }

  // Add rigs
  for (const auto& [rig_id, rig] : rigs) {
    reconstruction.AddRig(rig);
  }

  // Add frames
  for (auto& [frame_id, frame] : frames) {
    Frame frame_curr = frame;  // Copy the frame to avoid dangling pointer
    frame_curr.ResetRigPtr();
    reconstruction.AddFrame(frame_curr);
  }

  // Prepare the 2d-3d correspondences
  size_t min_supports = 2;
  std::unordered_map<image_t, std::vector<point3D_t>> image_to_point3D;
  if (tracks.size() > 0 || include_image_points) {
    // Initialize every point to corresponds to invalid point
    for (auto& [image_id, image] : images) {
      if (!image.IsRegistered() ||
          (cluster_id != -1 && image.ClusterId() != cluster_id))
        continue;
      image_to_point3D[image_id] =
          std::vector<point3D_t>(image.features.size(), -1);
    }

    if (tracks.size() > 0) {
      for (auto& [track_id, track] : tracks) {
        if (track.track.Length() < min_supports) {
          continue;
        }
        for (auto& observation : track.track.Elements()) {
          if (image_to_point3D.find(observation.image_id) !=
              image_to_point3D.end()) {
            image_to_point3D[observation.image_id][observation.point2D_idx] =
                track_id;
          }
        }
      }
    }
  }

  // Add points
  for (const auto& [track_id, track] : tracks) {
    Point3D colmap_point;
    colmap_point.xyz = track.xyz;
    colmap_point.color = track.color;
    colmap_point.error = track.error;

    // Add track element
    for (auto& observation : track.track.Elements()) {
      const Image& image = images.at(observation.image_id);
      if (!image.IsRegistered() ||
          (cluster_id != -1 && image.ClusterId() != cluster_id))
        continue;
      colmap::TrackElement colmap_track_el;
      colmap_track_el.image_id = observation.image_id;
      colmap_track_el.point2D_idx = observation.point2D_idx;

      colmap_point.track.AddElement(colmap_track_el);
    }

    if (colmap_point.track.Length() < min_supports) continue;

    colmap_point.track.Compress();
    reconstruction.AddPoint3D(track_id, std::move(colmap_point));
  }

  // Add images
  for (const auto& [image_id, image] : images) {
    colmap::Image image_colmap;
    bool keep_points =
        image_to_point3D.find(image_id) != image_to_point3D.end();
    ConvertGlomapToColmapImage(image, image_colmap, keep_points);
    if (keep_points) {
      std::vector<point3D_t>& track_ids = image_to_point3D[image_id];
      for (size_t i = 0; i < image.features.size(); i++) {
        if (track_ids[i] != -1 && reconstruction.ExistsPoint3D(track_ids[i])) {
          image_colmap.SetPoint3DForPoint2D(i, track_ids[i]);
        }
      }
    }

    reconstruction.AddImage(std::move(image_colmap));
  }

  // Deregister frames
  for (auto& [frame_id, frame] : frames) {
    if ((cluster_id != 0 && !frame.is_registered) ||
        (frame.cluster_id != cluster_id && cluster_id != -1)) {
      reconstruction.DeRegisterFrame(frame_id);
    }
  }

  reconstruction.UpdatePoint3DErrors();
}

void ConvertColmapToGlomap(
    const colmap::Reconstruction& reconstruction,
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<camera_t, colmap::Camera>& cameras,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& tracks) {
  // Clear the glomap reconstruction
  cameras.clear();
  images.clear();

  // Add cameras
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    cameras[camera_id] = camera;
  }

  // Add rigs
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    rigs[rig_id] = rig;
  }

  // Add frames
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    frames[frame_id] = Frame(frame);
    frames[frame_id].SetRigPtr(rigs.find(frame.RigId()) != rigs.end()
                                   ? &rigs[frame.RigId()]
                                   : nullptr);
    frames[frame_id].is_registered = frame.HasPose();
  }

  for (auto& [image_id, image_colmap] : reconstruction.Images()) {
    auto ite = images.insert(std::make_pair(image_colmap.ImageId(),
                                            Image(image_colmap.ImageId(),
                                                  image_colmap.CameraId(),
                                                  image_colmap.Name())));

    Image& image = ite.first->second;
    image.frame_id = image_colmap.FrameId();
    image.frame_ptr = frames.find(image.frame_id) != frames.end()
                          ? &frames[image.frame_id]
                          : nullptr;
    image.features.clear();
    image.features.reserve(image_colmap.NumPoints2D());

    for (auto& point2D : image_colmap.Points2D()) {
      image.features.push_back(point2D.xy);
    }
  }

  ConvertColmapPoints3DToGlomapTracks(reconstruction, tracks);
}

void ConvertColmapPoints3DToGlomapTracks(
    const colmap::Reconstruction& reconstruction,
    std::unordered_map<point3D_t, Point3D>& tracks) {
  // Read tracks
  tracks.clear();
  tracks.reserve(reconstruction.NumPoints3D());

  auto& points3D = reconstruction.Points3D();
  for (auto& [point3d_id, point3D] : points3D) {
    tracks.insert(std::make_pair(point3d_id, point3D));
  }
}

// For ease of debug, go through the database twice: first extract the
// available pairs, then read matches from pairs.
void ConvertDatabaseToGlomap(
    const colmap::Database& database,
    ViewGraph& view_graph,
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<camera_t, colmap::Camera>& cameras,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images) {
  // Add the images
  std::vector<colmap::Image> images_colmap = database.ReadAllImages();
  image_t counter = 0;
  for (auto& image : images_colmap) {
    std::cout << "\r Loading Images " << counter + 1 << " / "
              << images_colmap.size() << std::flush;
    counter++;

    const image_t image_id = image.ImageId();
    if (image_id == colmap::kInvalidImageId) continue;
    images.insert(std::make_pair(
        image_id, Image(image_id, image.CameraId(), image.Name())));

    // TODO: Implement the logic of reading prior pose from the database
    // const colmap::PosePrior prior = database.ReadPosePrior(image_id);
    // if (prior.HasPosition()) {
    //   const colmap::Rigid3d
    //   world_from_cam_prior(Eigen::Quaterniond::Identity(),
    //                                              prior.position);
    //   ite.first->second.cam_from_world =
    //   Rigid3d(Inverse(world_from_cam_prior));
    // } else {
    //   ite.first->second.cam_from_world = Rigid3d();
    // }
  }
  std::cout << '\n';

  // Read keypoints
  for (auto& [image_id, image] : images) {
    const colmap::FeatureKeypoints keypoints = database.ReadKeypoints(image_id);
    const int num_keypoints = keypoints.size();
    image.features.resize(num_keypoints);
    for (int i = 0; i < num_keypoints; i++) {
      image.features[i] = Eigen::Vector2d(keypoints[i].x, keypoints[i].y);
    }
  }

  LOG(INFO) << "Read " << images.size() << " images";

  // Add the cameras
  std::vector<colmap::Camera> cameras_colmap = database.ReadAllCameras();
  for (auto& camera : cameras_colmap) {
    cameras[camera.camera_id] = camera;
  }

  // Add the rigs
  std::vector<colmap::Rig> rigs_colmap = database.ReadAllRigs();
  for (auto& rig : rigs_colmap) {
    rigs[rig.RigId()] = rig;
  }

  // Add the frames
  std::vector<colmap::Frame> frames_colmap = database.ReadAllFrames();
  for (auto& frame : frames_colmap) {
    frame_t frame_id = frame.FrameId();
    if (frame_id == colmap::kInvalidFrameId) continue;
    frames[frame_id] = Frame(frame);
    frames[frame_id].SetRigId(frame.RigId());
    frames[frame_id].SetRigPtr(rigs.find(frame.RigId()) != rigs.end()
                                   ? &rigs[frame.RigId()]
                                   : nullptr);
    frames[frame_id].SetRigFromWorld(Rigid3d());

    for (auto data_id : frame.ImageIds()) {
      image_t image_id = data_id.id;
      if (images.find(image_id) != images.end()) {
        images[image_id].frame_id = frame_id;
        images[image_id].frame_ptr = &frames[frame_id];
      }
    }
  }

  // cameras that are not used in any rig
  rig_t max_rig_id = 0;
  std::unordered_map<camera_t, rig_t> cameras_id_to_rig_id;
  for (const auto& [rig_id, rig] : rigs) {
    max_rig_id = std::max(max_rig_id, rig_id);

    sensor_t sensor_id = rig.RefSensorId();
    if (sensor_id.type == SensorType::CAMERA) {
      cameras_id_to_rig_id[rig.RefSensorId().id] = rig_id;
    }
    const std::map<sensor_t, std::optional<Rigid3d>>& sensors =
        rig.NonRefSensors();
    for (const auto& [sensor_id, sensor_pose] : sensors) {
      if (sensor_id.type == SensorType::CAMERA) {
        cameras_id_to_rig_id[sensor_id.id] = rig_id;
      }
    }
  }

  // For cameras that are not in any rig, add camera rigs
  for (const auto& [camera_id, camera] : cameras) {
    if (cameras_id_to_rig_id.find(camera_id) == cameras_id_to_rig_id.end()) {
      Rig rig;
      rig.SetRigId(++max_rig_id);
      rig.AddRefSensor(camera.SensorId());
      rigs[rig.RigId()] = rig;
      cameras_id_to_rig_id[camera_id] = rig.RigId();
    }
  }

  frame_t max_frame_id = 0;
  // For frames that are not in any rig, add camera rigs
  for (const auto& [frame_id, frame] : frames) {
    if (frame_id == colmap::kInvalidFrameId) continue;
    max_frame_id = std::max(max_frame_id, frame_id);
  }

  // For images without frames, initialize trivial frames
  for (auto& [image_id, image] : images) {
    if (image.frame_id == colmap::kInvalidFrameId) {
      frame_t frame_id = ++max_frame_id;

      CreateFrameForImage(Rigid3d(),
                          image,
                          rigs,
                          frames,
                          cameras_id_to_rig_id[image.camera_id],
                          frame_id);
    }
  }

  // Add the matches
  std::vector<std::pair<colmap::image_pair_t, colmap::FeatureMatches>>
      all_matches = database.ReadAllMatches();

  // Go through all matches and store the matche with enough observations in
  // the view_graph
  size_t invalid_count = 0;
  std::unordered_map<image_pair_t, ImagePair>& image_pairs =
      view_graph.image_pairs;
  for (size_t match_idx = 0; match_idx < all_matches.size(); match_idx++) {
    if ((match_idx + 1) % 1000 == 0 || match_idx == all_matches.size() - 1)
      std::cout << "\r Loading Image Pair " << match_idx + 1 << " / "
                << all_matches.size() << std::flush;
    // Read the image pair from COLMAP database
    colmap::image_pair_t pair_id = all_matches[match_idx].first;
    std::pair<colmap::image_t, colmap::image_t> image_pair_colmap =
        colmap::PairIdToImagePair(pair_id);
    colmap::image_t image_id1 = image_pair_colmap.first;
    colmap::image_t image_id2 = image_pair_colmap.second;

    colmap::FeatureMatches& feature_matches = all_matches[match_idx].second;

    // Initialize the image pair
    auto ite = image_pairs.insert(
        std::make_pair(colmap::ImagePairToPairId(image_id1, image_id2),
                       ImagePair(image_id1, image_id2)));
    ImagePair& image_pair = ite.first->second;

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

    // Collect the fundemental matrices
    if (two_view.config == colmap::TwoViewGeometry::UNCALIBRATED) {
      image_pair.F = two_view.F;
    } else if (two_view.config == colmap::TwoViewGeometry::CALIBRATED) {
      image_pair.F = colmap::FundamentalFromEssentialMatrix(
          cameras.at(images.at(image_pair.image_id2).camera_id)
              .CalibrationMatrix(),
          colmap::EssentialMatrixFromPose(image_pair.cam2_from_cam1),
          cameras.at(images.at(image_pair.image_id1).camera_id)
              .CalibrationMatrix());
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

    std::vector<Eigen::Vector2d>& keypoints1 =
        images[image_pair.image_id1].features;
    std::vector<Eigen::Vector2d>& keypoints2 =
        images[image_pair.image_id2].features;

    size_t count = 0;
    for (int i = 0; i < feature_matches.size(); i++) {
      colmap::point2D_t point2D_idx1 = feature_matches[i].point2D_idx1;
      colmap::point2D_t point2D_idx2 = feature_matches[i].point2D_idx2;
      if (point2D_idx1 != colmap::kInvalidPoint2DIdx &&
          point2D_idx2 != colmap::kInvalidPoint2DIdx) {
        if (keypoints1.size() <= point2D_idx1 ||
            keypoints2.size() <= point2D_idx2)
          continue;
        image_pair.matches.row(count) << point2D_idx1, point2D_idx2;
        count++;
      }
    }
    image_pair.matches.conservativeResize(count, 2);
  }
  LOG(INFO) << "Pairs read done. " << invalid_count << " / "
            << view_graph.image_pairs.size() << " are invalid";
}

void CreateOneRigPerCamera(
    const std::unordered_map<camera_t, colmap::Camera>& cameras,
    std::unordered_map<rig_t, Rig>& rigs) {
  for (const auto& [camera_id, camera] : cameras) {
    Rig rig;
    rig.SetRigId(camera_id);
    rig.AddRefSensor(camera.SensorId());
  }
}

void CreateFrameForImage(const Rigid3d& cam_from_world,
                         Image& image,
                         std::unordered_map<rig_t, Rig>& rigs,
                         std::unordered_map<frame_t, Frame>& frames,
                         rig_t rig_id,
                         frame_t frame_id) {
  Frame frame;
  if (frame_id == colmap::kInvalidFrameId) {
    frame_id = image.image_id;
  }
  if (rig_id == colmap::kInvalidRigId) {
    rig_id = image.camera_id;
  }
  frame.SetFrameId(frame_id);
  frame.SetRigId(rig_id);
  frame.SetRigPtr(rigs.find(rig_id) != rigs.end() ? &rigs[rig_id] : nullptr);
  frame.AddDataId(image.DataId());
  frame.SetRigFromWorld(cam_from_world);
  frames[frame_id] = frame;

  image.frame_id = frame_id;
  image.frame_ptr = &frames[frame_id];
}

}  // namespace glomap

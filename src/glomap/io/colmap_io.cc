#include "glomap/io/colmap_io.h"

#include "colmap/feature/utils.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

namespace {

void WriteReconstruction(const colmap::Reconstruction& reconstruction,
                         const std::string& path,
                         const std::string& output_format,
                         const std::string& image_path) {
  colmap::Reconstruction recon_copy = reconstruction;
  if (!image_path.empty()) {
    LOG(INFO) << "Extracting colors ...";
    recon_copy.ExtractColorsForAllImages(image_path);
  }
  colmap::CreateDirIfNotExists(path, true);
  if (output_format == "txt") {
    recon_copy.WriteText(path);
  } else if (output_format == "bin") {
    recon_copy.WriteBinary(path);
  } else {
    LOG(ERROR) << "Unsupported output type";
  }
}

}  // namespace

namespace glomap {

void InitializeEmptyReconstructionFromDatabase(
    const colmap::Database& database, colmap::Reconstruction& reconstruction) {
  reconstruction = colmap::Reconstruction();

  // Add all cameras
  for (auto& camera : database.ReadAllCameras()) {
    reconstruction.AddCamera(std::move(camera));
  }

  // Add all rigs from database
  rig_t max_rig_id = 0;
  std::unordered_map<camera_t, rig_t> camera_to_rig;

  for (auto& rig : database.ReadAllRigs()) {
    max_rig_id = std::max(max_rig_id, rig.RigId());
    for (sensor_t sensor_id : rig.SensorIds()) {
      if (sensor_id.type == SensorType::CAMERA) {
        camera_to_rig[sensor_id.id] = rig.RigId();
      }
    }
    reconstruction.AddRig(std::move(rig));
  }

  // Create trivial rigs for cameras not in any rig
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    if (camera_to_rig.find(camera_id) != camera_to_rig.end()) {
      continue;  // Camera already has a rig
    }
    Rig rig;
    rig.SetRigId(++max_rig_id);
    rig.AddRefSensor(camera.SensorId());
    reconstruction.AddRig(rig);
    camera_to_rig[camera_id] = rig.RigId();
  }

  // Read all images from database (but don't add to reconstruction yet)
  std::vector<Image> images = database.ReadAllImages();
  for (auto& image : images) {
    image.SetPoints2D(colmap::FeatureKeypointsToPointsVector(
        database.ReadKeypoints(image.ImageId())));
  }

  // Add all frames from database first (before adding images).
  frame_t max_frame_id = 0;
  for (auto& frame : database.ReadAllFrames()) {
    if (frame.FrameId() == colmap::kInvalidFrameId) continue;
    max_frame_id = std::max(max_frame_id, frame.FrameId());
    reconstruction.AddFrame(std::move(frame));
  }

  // Create trivial frames for images that don't have a frame in the database.
  for (auto& image : images) {
    if (image.HasFrameId() && reconstruction.ExistsFrame(image.FrameId())) {
      continue;  // Image already has a valid frame
    }

    frame_t frame_id = ++max_frame_id;
    rig_t rig_id = camera_to_rig.at(image.CameraId());

    Frame frame;
    frame.SetFrameId(frame_id);
    frame.SetRigId(rig_id);
    frame.AddDataId(image.DataId());
    reconstruction.AddFrame(frame);

    image.SetFrameId(frame_id);
  }

  // Now add all images to reconstruction (frames already exist)
  // Note: AddImage also sets the frame pointer automatically
  for (auto& image : images) {
    reconstruction.AddImage(std::move(image));
  }

  LOG(INFO) << "Read " << reconstruction.NumImages() << " images";
}

void InitializeViewGraphFromDatabase(
    const colmap::Database& database,
    const colmap::Reconstruction& reconstruction,
    ViewGraph& view_graph) {
  view_graph.Clear();

  // Build view graph from matches
  auto all_matches = database.ReadAllMatches();
  size_t invalid_count = 0;

  for (auto& [pair_id, feature_matches] : all_matches) {
    auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);

    THROW_CHECK(!view_graph.HasImagePair(image_id1, image_id2))
        << "Duplicate image pair in database: " << image_id1 << ", "
        << image_id2;

    colmap::TwoViewGeometry two_view =
        database.ReadTwoViewGeometry(image_id1, image_id2);

    // Build the image pair from TwoViewGeometry
    ImagePair image_pair;
    static_cast<colmap::TwoViewGeometry&>(image_pair) = std::move(two_view);

    // If the image is marked as invalid or watermark, then skip
    if (image_pair.config == colmap::TwoViewGeometry::UNDEFINED ||
        image_pair.config == colmap::TwoViewGeometry::DEGENERATE ||
        image_pair.config == colmap::TwoViewGeometry::WATERMARK ||
        image_pair.config == colmap::TwoViewGeometry::MULTIPLE) {
      invalid_count++;
      view_graph.AddImagePair(image_id1, image_id2, std::move(image_pair));
      view_graph.SetInvalidImagePair(
          colmap::ImagePairToPairId(image_id1, image_id2));
      continue;
    }

    const Image& image1 = reconstruction.Image(image_id1);
    const Image& image2 = reconstruction.Image(image_id2);

    // For calibrated pairs, recompute F from the relative pose.
    // TODO: Once this update is moved to the colmap side, we can safely drop
    // the reconstruction argument in this function and move it to ViewGraph
    // implementation.
    if (image_pair.config == colmap::TwoViewGeometry::CALIBRATED) {
      image_pair.F = colmap::FundamentalFromEssentialMatrix(
          reconstruction.Camera(image2.CameraId()).CalibrationMatrix(),
          colmap::EssentialMatrixFromPose(image_pair.cam2_from_cam1),
          reconstruction.Camera(image1.CameraId()).CalibrationMatrix());
    }

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

    view_graph.AddImagePair(image_id1, image_id2, std::move(image_pair));
  }
  LOG(INFO) << "Loaded " << all_matches.size() << " image pairs, "
            << invalid_count << " invalid";
}

colmap::Reconstruction SubReconstructionByClusterId(
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

void WriteReconstructionsByClusters(
    const std::string& reconstruction_path,
    const colmap::Reconstruction& reconstruction,
    const std::unordered_map<frame_t, int>& cluster_ids,
    const std::string& output_format,
    const std::string& image_path) {
  // Find the maximum cluster id to determine if we have multiple clusters
  int max_cluster_id = -1;
  for (const auto& [frame_id, cluster_id] : cluster_ids) {
    if (cluster_id > max_cluster_id) {
      max_cluster_id = cluster_id;
    }
  }

  // If no clusters, output as single reconstruction
  if (max_cluster_id == -1) {
    WriteReconstruction(
        reconstruction, reconstruction_path + "/0", output_format, image_path);
  } else {
    // Export each cluster separately
    for (int comp = 0; comp <= max_cluster_id; comp++) {
      colmap::Reconstruction cluster_recon =
          SubReconstructionByClusterId(reconstruction, cluster_ids, comp);
      WriteReconstruction(cluster_recon,
                          reconstruction_path + "/" + std::to_string(comp),
                          output_format,
                          image_path);
    }
    LOG(INFO) << "Exported " << max_cluster_id + 1 << " reconstructions";
  }
}

}  // namespace glomap

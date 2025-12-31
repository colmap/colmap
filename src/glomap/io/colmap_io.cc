#include "glomap/io/colmap_io.h"

#include "colmap/feature/utils.h"
#include "colmap/util/logging.h"

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

  // Add all frames from database first (before adding images)
  frame_t max_frame_id = 0;
  for (auto& frame : database.ReadAllFrames()) {
    if (frame.FrameId() == colmap::kInvalidFrameId) continue;
    max_frame_id = std::max(max_frame_id, frame.FrameId());
    frame.SetRigFromWorld(Rigid3d());
    reconstruction.AddFrame(std::move(frame));
  }

  // Create trivial frames for images that don't have a frame in the database
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
    frame.SetRigFromWorld(Rigid3d());
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

}  // namespace glomap

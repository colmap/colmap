#include "glomap/sfm/rotation_averager.h"

#include "colmap/scene/reconstruction_io_utils.h"

#include "glomap/estimators/rotation_initializer.h"
#include "glomap/io/colmap_io.h"

namespace glomap {
namespace {

// Returns true if any camera in the reconstruction has unknown cam_from_rig.
bool HasUnknownCamsFromRig(const colmap::Reconstruction& reconstruction) {
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor_id.type != SensorType::CAMERA) continue;
      if (!rig.MaybeSensorFromRig(sensor_id).has_value()) {
        return true;
      }
    }
  }
  return false;
}

// Creates an expanded reconstruction where cameras with unknown cam_from_rig
// are split into separate singleton rigs (each such camera becomes its own
// rig). This allows rotation averaging to estimate their orientations
// independently.
colmap::Reconstruction CreateExpandedReconstruction(
    const colmap::Reconstruction& reconstruction) {
  // Collect cameras with unknown cam_from_rig and find max rig ID.
  std::unordered_set<camera_t> unknown_cams_from_rig;
  rig_t max_rig_id = 0;
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    max_rig_id = std::max(max_rig_id, rig_id);
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor_id.type != SensorType::CAMERA) continue;
      if (!rig.MaybeSensorFromRig(sensor_id).has_value()) {
        unknown_cams_from_rig.insert(sensor_id.id);
      }
    }
  }

  colmap::Reconstruction recon_expanded;

  // Add all cameras first (required before adding rigs).
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    recon_expanded.AddCamera(camera);
  }

  // For cameras with known cam_from_rig, create rigs with only those sensors.
  std::unordered_map<camera_t, rig_t> camera_id_to_rig_id;
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    Rig rig_expanded;
    rig_expanded.SetRigId(rig_id);
    rig_expanded.AddRefSensor(rig.RefSensorId());
    for (const auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
      if (sensor_id.type != SensorType::CAMERA) continue;
      if (rig.MaybeSensorFromRig(sensor_id).has_value()) {
        rig_expanded.AddSensor(sensor_id, sensor_from_rig);
        camera_id_to_rig_id[sensor_id.id] = rig_id;
      }
    }
    camera_id_to_rig_id[rig.RefSensorId().id] = rig_id;
    recon_expanded.AddRig(rig_expanded);
  }

  // For each camera with unknown cam_from_rig, create a separate singleton rig.
  for (const auto& camera_id : unknown_cams_from_rig) {
    Rig rig_expanded;
    rig_expanded.SetRigId(++max_rig_id);
    rig_expanded.AddRefSensor(sensor_t(SensorType::CAMERA, camera_id));
    camera_id_to_rig_id[camera_id] = rig_expanded.RigId();
    recon_expanded.AddRig(rig_expanded);
  }

  frame_t max_frame_id = 0;
  for (const auto& [frame_id, _] : reconstruction.Frames()) {
    THROW_CHECK_NE(frame_id, colmap::kInvalidFrameId);
    max_frame_id = std::max(max_frame_id, frame_id);
  }
  max_frame_id++;

  const Eigen::Quaterniond kUnknownRotation = Eigen::Quaterniond(
      Eigen::Vector4d::Constant(std::numeric_limits<double>::quiet_NaN()));
  const Eigen::Vector3d kUnknownTranslation =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  const Rigid3d kUnknownPose(kUnknownRotation, kUnknownTranslation);

  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    Frame frame_expanded;
    frame_expanded.SetFrameId(frame_id);
    frame_expanded.SetRigId(frame.RigId());
    frame_expanded.SetRigFromWorld(kUnknownPose);
    recon_expanded.AddFrame(frame_expanded);
  }

  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    Frame& frame_expanded = recon_expanded.Frame(frame_id);
    frame_expanded.SetRigPtr(recon_expanded.ExistsRig(frame.RigId())
                                 ? &recon_expanded.Rig(frame.RigId())
                                 : nullptr);

    for (const auto& data_id : frame.ImageIds()) {
      const auto& image = reconstruction.Image(data_id.id);
      if (!image.HasPose()) {
        continue;
      }

      Image image_expanded;
      image_expanded.SetImageId(image.ImageId());
      image_expanded.SetCameraId(image.CameraId());
      image_expanded.SetName(image.Name());

      if (unknown_cams_from_rig.count(image.CameraId()) == 0) {
        // Image belongs to the existing frame.
        frame_expanded.AddDataId(image_expanded.DataId());
        image_expanded.SetFrameId(frame_id);
        recon_expanded.AddImage(std::move(image_expanded));
      } else {
        // If cam_from_rig is unknown, create a new frame with a singleton rig
        // (not camera_id, since we created separate rigs for these).
        const frame_t new_frame_id = ++max_frame_id;
        const rig_t rig_id = camera_id_to_rig_id.at(image.CameraId());
        Frame new_frame;
        new_frame.SetFrameId(new_frame_id);
        new_frame.SetRigId(rig_id);
        new_frame.AddDataId(image_expanded.DataId());
        new_frame.SetRigFromWorld(kUnknownPose);
        recon_expanded.AddFrame(new_frame);

        image_expanded.SetFrameId(new_frame_id);
        recon_expanded.AddImage(std::move(image_expanded));
      }
    }
  }

  return recon_expanded;
}

}  // namespace

bool SolveRotationAveraging(ViewGraph& view_graph,
                            colmap::Reconstruction& reconstruction,
                            std::vector<colmap::PosePrior>& pose_priors,
                            const RotationEstimatorOptions& options) {
  view_graph.KeepLargestConnectedComponents(reconstruction);

  // If there are cameras with unknown cam_from_rig, run expanded rotation
  // averaging first to estimate their orientations.
  if (HasUnknownCamsFromRig(reconstruction)) {
    LOG(INFO) << "Running expanded rotation averaging for rigged cameras";

    colmap::Reconstruction recon_expanded =
        CreateExpandedReconstruction(reconstruction);

    view_graph.KeepLargestConnectedComponents(recon_expanded);

    // Run rotation averaging on the expanded reconstruction.
    RotationEstimator rotation_estimator_expanded(options);
    rotation_estimator_expanded.EstimateRotations(
        view_graph, pose_priors, recon_expanded);

    // Collect the results and initialize the original reconstruction.
    std::unordered_map<image_t, Rigid3d> expanded_cams_from_world;
    for (const auto& [image_id, image] : recon_expanded.Images()) {
      if (!image.HasPose()) continue;
      expanded_cams_from_world[image_id] = image.CamFromWorld();
    }

    LOG(INFO) << "Initializing rig rotations from expanded reconstruction";
    InitializeRigRotationsFromImages(expanded_cams_from_world, reconstruction);

    // Run rotation averaging on the original reconstruction with initialization
    // from the expanded reconstruction.
    RotationEstimatorOptions options_ra = options;
    options_ra.skip_initialization = true;
    options_ra.use_stratified = false;
    RotationEstimator rotation_estimator(options_ra);
    if (!rotation_estimator.EstimateRotations(
            view_graph, pose_priors, reconstruction)) {
      return false;
    }
  } else {
    // No unknown cam_from_rig, run rotation averaging directly.
    RotationEstimator rotation_estimator(options);
    if (!rotation_estimator.EstimateRotations(
            view_graph, pose_priors, reconstruction)) {
      return false;
    }
  }

  view_graph.KeepLargestConnectedComponents(reconstruction);
  return true;
}

}  // namespace glomap

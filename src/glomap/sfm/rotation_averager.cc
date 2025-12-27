#include "glomap/sfm/rotation_averager.h"

#include "colmap/scene/reconstruction_io_utils.h"

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
  colmap::Reconstruction recon_expanded;

  // Add all cameras first (required before adding rigs).
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    recon_expanded.AddCamera(camera);
  }

  // Create expanded rigs with known sensors only.
  // Cameras with unknown cam_from_rig get their own singleton rigs.
  std::unordered_map<camera_t, rig_t> singleton_rig_ids;
  rig_t next_rig_id = 0;

  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    next_rig_id = std::max(next_rig_id, rig_id + 1);

    Rig rig_expanded;
    rig_expanded.SetRigId(rig_id);
    rig_expanded.AddRefSensor(rig.RefSensorId());

    for (const auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
      if (sensor_id.type != SensorType::CAMERA) continue;
      if (rig.MaybeSensorFromRig(sensor_id).has_value()) {
        rig_expanded.AddSensor(sensor_id, sensor_from_rig);
      } else {
        // Create singleton rig for this camera.
        const rig_t singleton_rig_id = next_rig_id++;
        Rig rig_singleton;
        rig_singleton.SetRigId(singleton_rig_id);
        rig_singleton.AddRefSensor(sensor_id);
        recon_expanded.AddRig(rig_singleton);
        singleton_rig_ids[sensor_id.id] = singleton_rig_id;
      }
    }
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
    if (frame.HasPose()) {
      frame_expanded.SetRigFromWorld(frame.RigFromWorld());
    } else {
      frame_expanded.SetRigFromWorld(kUnknownPose);
    }
    recon_expanded.AddFrame(frame_expanded);
  }

  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    Frame& frame_expanded = recon_expanded.Frame(frame_id);
    const Rig& original_rig = reconstruction.Rig(frame.RigId());

    for (const auto& data_id : frame.ImageIds()) {
      const auto& image = reconstruction.Image(data_id.id);
      if (!image.HasPose()) {
        continue;
      }

      Image image_expanded;
      image_expanded.SetImageId(image.ImageId());
      image_expanded.SetCameraId(image.CameraId());
      image_expanded.SetName(image.Name());

      // Check if camera belongs to this frame's rig (ref sensor or known
      // cam_from_rig).
      const sensor_t sensor_id(SensorType::CAMERA, image.CameraId());
      const bool belongs_to_frame_rig =
          original_rig.RefSensorId() == sensor_id ||
          original_rig.MaybeSensorFromRig(sensor_id).has_value();

      if (belongs_to_frame_rig) {
        // Camera belongs to this frame's rig.
        frame_expanded.AddDataId(image_expanded.DataId());
        image_expanded.SetFrameId(frame_id);
        recon_expanded.AddImage(std::move(image_expanded));
      } else {
        // Camera has its own singleton rig, create a new frame for it.
        const frame_t new_frame_id = ++max_frame_id;
        Frame new_frame;
        new_frame.SetFrameId(new_frame_id);
        new_frame.SetRigId(singleton_rig_ids.at(image.CameraId()));
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
    LOG(INFO) << "Detected cameras with unknown cam_from_rig, "
                 "estimating rotations with these cameras as independent";

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

    LOG(INFO)
        << "Initializing cam_from_rig from preliminary rotation estimates";
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

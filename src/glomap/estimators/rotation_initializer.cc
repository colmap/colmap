#include "glomap/estimators/rotation_initializer.h"

#include "colmap/geometry/pose.h"

namespace glomap {

bool ConvertRotationsFromImageToRig(
    const std::unordered_map<image_t, Rigid3d>& cams_from_world,
    colmap::Reconstruction& reconstruction) {
  std::unordered_map<camera_t, rig_t> camera_id_to_rig_id;
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor_id.type != SensorType::CAMERA) {
        continue;
      }
      camera_id_to_rig_id[sensor_id.id] = rig_id;
    }
  }

  std::unordered_map<camera_t, std::vector<Eigen::Quaterniond>>
      cam_from_ref_cam_rotations;

  std::unordered_map<frame_t, image_t> frame_to_ref_image_id;
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    // First, figure out the reference camera in the frame
    image_t ref_image_id = -1;
    for (const auto& data_id : frame.ImageIds()) {
      const image_t image_id = data_id.id;
      const auto& image = reconstruction.Image(image_id);
      if (image.HasPose() && image.IsRefInFrame()) {
        ref_image_id = image_id;
        frame_to_ref_image_id[frame_id] = ref_image_id;
        break;
      }
    }

    // If the reference image is not found, then skip the frame
    if (ref_image_id == -1) {
      continue;
    }

    const auto world_from_ref_cam_it = cams_from_world.find(ref_image_id);
    if (world_from_ref_cam_it == cams_from_world.end()) {
      continue;
    }

    // Then, collect the rotations from the cameras to the reference camera
    for (const auto& data_id : frame.ImageIds()) {
      const image_t image_id = data_id.id;
      const auto& image = reconstruction.Image(image_id);
      if (!image.HasPose() || image.IsRefInFrame()) {
        continue;
      }

      const auto world_from_cam_it = cams_from_world.find(image_id);
      if (world_from_cam_it == cams_from_world.end()) {
        continue;
      }

      cam_from_ref_cam_rotations[image.CameraId()].push_back(
          world_from_cam_it->second.rotation *
          world_from_ref_cam_it->second.rotation.inverse());
    }
  }

  const Eigen::Vector3d kNaNTranslation =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());

  // Use the average of the rotations to set the rotation from the camera.
  std::vector<double> weights;
  for (auto& [camera_id, curr_cam_from_ref_cam_rotations] :
       cam_from_ref_cam_rotations) {
    weights.resize(curr_cam_from_ref_cam_rotations.size(), 1.0);
    const Eigen::Quaterniond curr_cam_from_ref_cam_rotation =
        colmap::AverageQuaternions(curr_cam_from_ref_cam_rotations, weights);
    reconstruction.Rig(camera_id_to_rig_id.at(camera_id))
        .SetSensorFromRig(
            sensor_t(SensorType::CAMERA, camera_id),
            Rigid3d(curr_cam_from_ref_cam_rotation, kNaNTranslation));
  }

  // Then, collect the rotations into frames and rigs
  std::vector<Eigen::Quaterniond> rig_from_world_rotations;
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    // Then, collect the rotations from the cameras to the reference camera
    rig_from_world_rotations.clear();
    for (const auto& data_id : frame.ImageIds()) {
      const image_t image_id = data_id.id;

      if (!reconstruction.ExistsImage(image_id)) {
        continue;
      }

      const auto& image = reconstruction.Image(image_id);
      if (!image.HasPose()) {
        continue;
      }

      // For images that not estimated directly, we need to skip it
      const auto cam_from_world_it = cams_from_world.find(image_id);
      if (cam_from_world_it == cams_from_world.end()) {
        continue;
      }

      if (image_id == frame_to_ref_image_id.at(frame_id)) {
        rig_from_world_rotations.push_back(cam_from_world_it->second.rotation);
      } else {
        const auto& maybe_cam_from_rig =
            reconstruction.Rig(camera_id_to_rig_id.at(image.CameraId()))
                .MaybeSensorFromRig(
                    sensor_t(SensorType::CAMERA, image.CameraId()));
        if (!maybe_cam_from_rig.has_value()) {
          continue;
        }
        rig_from_world_rotations.push_back(
            maybe_cam_from_rig.value().rotation.inverse() *
            cam_from_world_it->second.rotation);
      }

      weights.resize(rig_from_world_rotations.size(), 1);
      const Eigen::Quaterniond rig_from_world_rotation =
          colmap::AverageQuaternions(rig_from_world_rotations, weights);
      reconstruction.Frame(frame_id).SetRigFromWorld(
          Rigid3d(rig_from_world_rotation, kNaNTranslation));
    }
  }

  return true;
}

}  // namespace glomap

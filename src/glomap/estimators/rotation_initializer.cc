#include "glomap/estimators/rotation_initializer.h"

#include "colmap/geometry/pose.h"

namespace glomap {

bool ConvertRotationsFromImageToRig(
    const std::unordered_map<image_t, Rigid3d>& cam_from_worlds,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<frame_t, Frame>& frames) {
  std::unordered_map<camera_t, rig_t> camera_id_to_rig_id;
  for (auto& [rig_id, rig] : rigs) {
    for (auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor_id.type != SensorType::CAMERA) continue;
      camera_id_to_rig_id[sensor_id.id] = rig_id;
    }
  }

  std::unordered_map<camera_t, std::vector<Eigen::Quaterniond>>
      cam_from_ref_cam_rotations;

  std::unordered_map<frame_t, image_t> frame_to_ref_image_id;
  for (auto& [frame_id, frame] : frames) {
    // First, figure out the reference camera in the frame
    image_t ref_img_id = -1;
    for (const auto& data_id : frame.ImageIds()) {
      image_t image_id = data_id.id;
      if (images.find(image_id) == images.end()) continue;
      const auto& image = images.at(image_id);
      if (!image.IsRegistered()) continue;

      if (image.camera_id == frame.RigPtr()->RefSensorId().id) {
        ref_img_id = image_id;
        frame_to_ref_image_id[frame_id] = ref_img_id;
        break;
      }
    }

    // If the reference image is not found, then skip the frame
    if (ref_img_id == -1) {
      continue;
    }

    // Then, collect the rotations from the cameras to the reference camera
    for (const auto& data_id : frame.ImageIds()) {
      image_t image_id = data_id.id;
      if (images.find(image_id) == images.end()) continue;
      const auto& image = images.at(image_id);
      if (!image.IsRegistered()) continue;

      Rig* rig_ptr = frame.RigPtr();

      // If the camera is a reference camera, then skip it
      if (image.camera_id == rig_ptr->RefSensorId().id) continue;

      if (rig_ptr
              ->MaybeSensorFromRig(
                  sensor_t(SensorType::CAMERA, image.camera_id))
              .has_value())
        continue;

      if (cam_from_ref_cam_rotations.find(image.camera_id) ==
          cam_from_ref_cam_rotations.end())
        cam_from_ref_cam_rotations[image.camera_id] =
            std::vector<Eigen::Quaterniond>();

      // Set the rotation from the camera to the world
      cam_from_ref_cam_rotations[image.camera_id].push_back(
          cam_from_worlds.at(image_id).rotation *
          cam_from_worlds.at(ref_img_id).rotation.inverse());
    }
  }

  Eigen::Vector3d nan_translation;
  nan_translation.setConstant(std::numeric_limits<double>::quiet_NaN());

  // Use the average of the rotations to set the rotation from the camera
  for (auto& [camera_id, cam_from_ref_cam_rotations_i] :
       cam_from_ref_cam_rotations) {
    const std::vector<double> weights(cam_from_ref_cam_rotations_i.size(), 1.0);
    Eigen::Quaterniond cam_from_ref_cam_rotation =
        colmap::AverageQuaternions(cam_from_ref_cam_rotations_i, weights);

    rigs[camera_id_to_rig_id[camera_id]].SetSensorFromRig(
        sensor_t(SensorType::CAMERA, camera_id),
        Rigid3d(cam_from_ref_cam_rotation, nan_translation));
  }

  // Then, collect the rotations into frames and rigs
  for (auto& [frame_id, frame] : frames) {
    // Then, collect the rotations from the cameras to the reference camera
    std::vector<Eigen::Quaterniond> rig_from_world_rotations;
    for (const auto& data_id : frame.ImageIds()) {
      image_t image_id = data_id.id;
      if (images.find(image_id) == images.end()) continue;
      const auto& image = images.at(image_id);
      if (!image.IsRegistered()) continue;

      // For images that not estimated directly, we need to skip it
      if (cam_from_worlds.find(image_id) == cam_from_worlds.end()) continue;

      if (image_id == frame_to_ref_image_id[frame_id]) {
        rig_from_world_rotations.push_back(
            cam_from_worlds.at(image_id).rotation);
      } else {
        auto cam_from_rig_opt =
            rigs[camera_id_to_rig_id[image.camera_id]].MaybeSensorFromRig(
                sensor_t(SensorType::CAMERA, image.camera_id));
        if (!cam_from_rig_opt.has_value()) continue;
        rig_from_world_rotations.push_back(
            cam_from_rig_opt.value().rotation.inverse() *
            cam_from_worlds.at(image_id).rotation);
      }

      const std::vector<double> rotation_weights(
          rig_from_world_rotations.size(), 1);
      Eigen::Quaterniond rig_from_world_rotation = colmap::AverageQuaternions(
          rig_from_world_rotations, rotation_weights);
      frame.SetRigFromWorld(Rigid3d(rig_from_world_rotation, nan_translation));
    }
  }

  return true;
}

}  // namespace glomap

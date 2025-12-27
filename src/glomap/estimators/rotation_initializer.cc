#include "glomap/estimators/rotation_initializer.h"

#include "colmap/geometry/pose.h"

namespace glomap {

bool InitializeRigRotationsFromImages(
    const std::unordered_map<image_t, Rigid3d>& cams_from_world,
    colmap::Reconstruction& reconstruction) {
  // Step 1: Estimate sensor_from_rig for cameras with unknown calibration.
  // Collect samples across frames, then average.
  std::unordered_map<camera_t,
                     std::pair<rig_t, std::vector<Eigen::Quaterniond>>>
      sensor_from_rig_samples;

  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    // Find the rotation of the reference image.
    const Eigen::Quaterniond* ref_rotation = nullptr;
    for (const auto& data_id : frame.ImageIds()) {
      const auto& image = reconstruction.Image(data_id.id);
      if (image.HasPose() && image.IsRefInFrame()) {
        const auto it = cams_from_world.find(data_id.id);
        if (it != cams_from_world.end()) {
          ref_rotation = &it->second.rotation;
        }
        break;
      }
    }
    if (ref_rotation == nullptr) {
      continue;
    }

    // Collect sensor_from_rig samples for non-reference cameras.
    for (const auto& data_id : frame.ImageIds()) {
      const auto& image = reconstruction.Image(data_id.id);
      if (!image.HasPose() || image.IsRefInFrame()) {
        continue;
      }

      const auto it = cams_from_world.find(data_id.id);
      if (it == cams_from_world.end()) {
        continue;
      }

      auto& [rig_id, rotations] = sensor_from_rig_samples[image.CameraId()];
      rig_id = frame.RigId();
      rotations.push_back(it->second.rotation * ref_rotation->inverse());
    }
  }

  const Eigen::Vector3d kNaNTranslation =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());

  std::vector<double> weights;
  for (auto& [camera_id, rig_id_and_samples] : sensor_from_rig_samples) {
    auto& [rig_id, samples] = rig_id_and_samples;
    weights.resize(samples.size(), 1.0);
    const Eigen::Quaterniond sensor_from_rig_rotation =
        colmap::AverageQuaternions(samples, weights);
    reconstruction.Rig(rig_id).SetSensorFromRig(
        sensor_t(SensorType::CAMERA, camera_id),
        Rigid3d(sensor_from_rig_rotation, kNaNTranslation));
  }

  // Step 2: Compute rig_from_world for each frame by averaging across images.
  std::vector<Eigen::Quaterniond> rig_from_world_samples;
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    rig_from_world_samples.clear();

    for (const auto& data_id : frame.ImageIds()) {
      if (!reconstruction.ExistsImage(data_id.id)) {
        continue;
      }

      const auto& image = reconstruction.Image(data_id.id);
      if (!image.HasPose()) {
        continue;
      }

      const auto it = cams_from_world.find(data_id.id);
      if (it == cams_from_world.end()) {
        continue;
      }

      if (image.IsRefInFrame()) {
        rig_from_world_samples.push_back(it->second.rotation);
      } else {
        const auto& maybe_sensor_from_rig =
            reconstruction.Rig(frame.RigId())
                .MaybeSensorFromRig(
                    sensor_t(SensorType::CAMERA, image.CameraId()));
        if (!maybe_sensor_from_rig.has_value()) {
          continue;
        }
        rig_from_world_samples.push_back(
            maybe_sensor_from_rig.value().rotation.inverse() *
            it->second.rotation);
      }
    }

    if (!rig_from_world_samples.empty()) {
      weights.resize(rig_from_world_samples.size(), 1.0);
      const Eigen::Quaterniond rig_from_world =
          colmap::AverageQuaternions(rig_from_world_samples, weights);
      reconstruction.Frame(frame_id).SetRigFromWorld(
          Rigid3d(rig_from_world, kNaNTranslation));
    }
  }

  return true;
}

}  // namespace glomap

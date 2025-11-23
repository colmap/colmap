#pragma once

#include "glomap/math/gravity.h"
#include "glomap/scene/frame.h"
#include "glomap/scene/types.h"
#include "glomap/types.h"

namespace glomap {

struct Image {
  Image() : image_id(-1), file_name("") {}
  Image(image_t img_id, camera_t cam_id, std::string file_name)
      : image_id(img_id), file_name(file_name), camera_id(cam_id) {}

  // Basic information
  // image_id, file_name need to be specified at construction time
  const image_t image_id;
  const std::string file_name;

  // The id of the camera
  camera_t camera_id;

  // Frame info
  // By default, set it to be invalid index
  frame_t frame_id = -1;
  struct Frame* frame_ptr = nullptr;

  // Distorted feature points in pixels.
  std::vector<Eigen::Vector2d> features;
  // Normalized feature rays, can be obtained by calling UndistortImages.
  std::vector<Eigen::Vector3d> features_undist;

  // Methods
  inline Eigen::Vector3d Center() const;

  // Methods to access the camera pose
  inline Rigid3d CamFromWorld() const;

  // Check whether the frame is registered
  inline bool IsRegistered() const;

  inline int ClusterId() const;

  // Check if cam_from_world needs to be composed with sensor_from_rig pose.
  inline bool HasTrivialFrame() const;

  // Easy way to check if the image has gravity information
  inline bool HasGravity() const;

  inline Eigen::Matrix3d GetRAlign() const;

  inline data_t DataId() const;
};

Eigen::Vector3d Image::Center() const {
  return CamFromWorld().rotation.inverse() * -CamFromWorld().translation;
}

// Concrete implementation of the methods
Rigid3d Image::CamFromWorld() const {
  return THROW_CHECK_NOTNULL(frame_ptr)->SensorFromWorld(
      sensor_t(SensorType::CAMERA, camera_id));
}

bool Image::IsRegistered() const {
  return frame_ptr != nullptr && frame_ptr->is_registered;
}

int Image::ClusterId() const {
  return frame_ptr != nullptr ? frame_ptr->cluster_id : -1;
}

bool Image::HasTrivialFrame() const {
  return THROW_CHECK_NOTNULL(frame_ptr)->RigPtr()->IsRefSensor(
      sensor_t(SensorType::CAMERA, camera_id));
}

bool Image::HasGravity() const {
  return frame_ptr->HasGravity() &&
         (HasTrivialFrame() ||
          frame_ptr->RigPtr()
              ->MaybeSensorFromRig(sensor_t(SensorType::CAMERA, camera_id))
              .has_value());
}

Eigen::Matrix3d Image::GetRAlign() const {
  if (HasGravity()) {
    if (HasTrivialFrame()) {
      return frame_ptr->gravity_info.GetRAlign();
    } else {
      return frame_ptr->RigPtr()
                 ->SensorFromRig(sensor_t(SensorType::CAMERA, camera_id))
                 .rotation.toRotationMatrix() *
             frame_ptr->gravity_info.GetRAlign();
    }
  } else {
    return Eigen::Matrix3d::Identity();
  }
}

data_t Image::DataId() const {
  return data_t(sensor_t(SensorType::CAMERA, camera_id), image_id);
}

}  // namespace glomap

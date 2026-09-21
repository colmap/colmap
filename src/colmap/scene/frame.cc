// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/frame.h"

namespace colmap {

Frame::Frame(const Frame& other)
    : frame_id_(other.frame_id_),
      rig_id_(other.rig_id_),
      data_ids_(other.data_ids_),
      has_final_data_ids_(false),
      rig_from_world_(other.rig_from_world_),
      rig_ptr_(other.rig_ptr_) {}

Frame& Frame::operator=(const Frame& other) {
  if (this != &other) {
    frame_id_ = other.frame_id_;
    rig_id_ = other.rig_id_;
    data_ids_ = other.data_ids_;
    has_final_data_ids_ = false;
    rig_from_world_ = other.rig_from_world_;
    rig_ptr_ = other.rig_ptr_;
  }
  return *this;
}

void Frame::ClearDataIds() {
  THROW_CHECK(!has_final_data_ids_)
      << "Cannot clear data ids of a finalized frame.";
  data_ids_.clear();
}

void Frame::SetRigPtr(class Rig* rig) {
  THROW_CHECK_NOTNULL(rig);
  THROW_CHECK_NE(rig->RigId(), kInvalidRigId);
  for (const auto& data_id : data_ids_) {
    switch (data_id.sensor_id.type) {
      case SensorType::CAMERA:
        THROW_CHECK(rig->HasSensor(data_id.sensor_id));
        break;
      case SensorType::IMU:
        // Note that we do not (yet) support IMU measurement data.
        break;
      case SensorType::INVALID:
        LOG(FATAL_THROW) << "Invalid sensor type: " << data_id.sensor_id.type;
        break;
    }
  }
  if (HasRigPtr()) {
    rig_id_ = rig->RigId();
    rig_ptr_ = rig;
  } else {
    THROW_CHECK_EQ(rig->RigId(), rig_id_);
    rig_ptr_ = rig;
  }
}

void Frame::SetCamFromWorld(camera_t camera_id, const Rigid3d& cam_from_world) {
  THROW_CHECK_NOTNULL(rig_ptr_);
  const sensor_t sensor_id(SensorType::CAMERA, camera_id);
  if (rig_ptr_->IsRefSensor(sensor_id)) {
    SetRigFromWorld(cam_from_world);
  } else {
    const Rigid3d& cam_from_rig = rig_ptr_->SensorFromRig(sensor_id);
    SetRigFromWorld(Inverse(cam_from_rig) * cam_from_world);
  }
}

std::ostream& operator<<(std::ostream& stream, const Frame& frame) {
  stream << "Frame(frame_id=" << frame.FrameId() << ", rig_id=";
  if (frame.HasRigId()) {
    if (frame.RigId() == kInvalidRigId) {
      stream << "Invalid";
    } else {
      stream << frame.RigId();
    }
  } else {
    stream << "Unknown";
  }
  stream << ", has_pose=" << frame.HasPose() << ", data_ids=[";
  for (auto it = frame.DataIds().begin(); it != frame.DataIds().end();) {
    stream << "(" << it->sensor_id.type << ", " << it->sensor_id.id << ", "
           << it->id << ")";
    if (++it != frame.DataIds().end()) {
      stream << ", ";
    }
  }
  stream << "])";
  return stream;
}

}  // namespace colmap

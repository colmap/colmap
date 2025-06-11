// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/sensor/rig.h"
#include "colmap/util/types.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>

namespace colmap {

// Frames represent (posed) instantiations of rigs with associated measurements
// for the different sensors. The captured sensor measurements are defined by
// the list of data ids.
class Frame {
 public:
  // Access the unique identifier of the frame.
  inline frame_t FrameId() const;
  inline void SetFrameId(frame_t frame_id);

  // Access the frame's associated data.
  inline std::set<data_t>& DataIds();
  inline const std::set<data_t>& DataIds() const;
  inline void AddDataId(data_t data_id);

  // Check whether the data is associated with the frame.
  inline bool HasDataId(data_t data_id) const;

  // Access the unique identifier of the rig. Note that multiple frames
  // might share the same rig.
  inline rig_t RigId() const;
  inline void SetRigId(rig_t rig_id);
  inline bool HasRigId() const;

  // Access to the underlying, shared rig object.
  // This is typically only set when the frame was added to a reconstruction.
  inline class Rig* RigPtr() const;
  inline void SetRigPtr(class Rig* rig);
  inline void ResetRigPtr();
  // Check if the frame has a non-trivial rig.
  inline bool HasRigPtr() const;

  // Access the rig from world transformation.
  inline const Rigid3d& RigFromWorld() const;
  inline Rigid3d& RigFromWorld();
  inline const std::optional<Rigid3d>& MaybeRigFromWorld() const;
  inline std::optional<Rigid3d>& MaybeRigFromWorld();
  inline void SetRigFromWorld(const Rigid3d& rig_from_world);
  inline void SetRigFromWorld(const std::optional<Rigid3d>& rig_from_world);
  inline bool HasPose() const;
  inline void ResetPose();

  // Get the sensor from world transformation.
  inline Rigid3d SensorFromWorld(sensor_t sensor_id) const;

  // Set the world to frame from the given camera to world transformation.
  void SetCamFromWorld(camera_t camera_id, const Rigid3d& cam_from_world);

  // Convenience method with view into all image data identifiers.
  inline auto ImageIds() const {
    return filter_view(
        [](const data_t& data_id) {
          return data_id.sensor_id.type == SensorType::CAMERA;
        },
        data_ids_.begin(),
        data_ids_.end());
  }

  inline bool operator==(const Frame& other) const;
  inline bool operator!=(const Frame& other) const;

 private:
  frame_t frame_id_ = kInvalidFrameId;
  std::set<data_t> data_ids_;

  // Store the rig_from_world transformation and an optional rig calibration.
  // If the rig calibration is a nullptr, the frame becomes a single sensor
  // case, where rig modeling is no longer needed.
  std::optional<Rigid3d> rig_from_world_;

  // Rig calibration.
  rig_t rig_id_ = kInvalidRigId;
  class Rig* rig_ptr_ = nullptr;
};

std::ostream& operator<<(std::ostream& stream, const Frame& frame);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

frame_t Frame::FrameId() const { return frame_id_; }

void Frame::SetFrameId(frame_t frame_id) { frame_id_ = frame_id; }

std::set<data_t>& Frame::DataIds() { return data_ids_; }

const std::set<data_t>& Frame::DataIds() const { return data_ids_; }

void Frame::AddDataId(data_t data_id) { data_ids_.insert(data_id); }

bool Frame::HasDataId(data_t data_id) const {
  return data_ids_.find(data_id) != data_ids_.end();
}

rig_t Frame::RigId() const { return rig_id_; }

void Frame::SetRigId(rig_t rig_id) { rig_id_ = rig_id; }

bool Frame::HasRigId() const { return rig_id_ != kInvalidRigId; }

Rig* Frame::RigPtr() const { return THROW_CHECK_NOTNULL(rig_ptr_); }

void Frame::SetRigPtr(class Rig* rig) { rig_ptr_ = rig; }

void Frame::ResetRigPtr() { rig_ptr_ = nullptr; }

bool Frame::HasRigPtr() const { return rig_ptr_ != nullptr; }

const Rigid3d& Frame::RigFromWorld() const {
  THROW_CHECK(rig_from_world_) << "Frame does not have a valid pose.";
  return *rig_from_world_;
}

Rigid3d& Frame::RigFromWorld() {
  THROW_CHECK(rig_from_world_) << "Frame does not have a valid pose.";
  return *rig_from_world_;
}

const std::optional<Rigid3d>& Frame::MaybeRigFromWorld() const {
  return rig_from_world_;
}

std::optional<Rigid3d>& Frame::MaybeRigFromWorld() { return rig_from_world_; }

void Frame::SetRigFromWorld(const Rigid3d& rig_from_world) {
  rig_from_world_ = rig_from_world;
}

void Frame::SetRigFromWorld(const std::optional<Rigid3d>& rig_from_world) {
  rig_from_world_ = rig_from_world;
}

bool Frame::HasPose() const { return rig_from_world_.has_value(); }

void Frame::ResetPose() { rig_from_world_.reset(); }

Rigid3d Frame::SensorFromWorld(sensor_t sensor_id) const {
  THROW_CHECK_NOTNULL(rig_ptr_);
  if (rig_ptr_->IsRefSensor(sensor_id)) {
    return RigFromWorld();
  } else {
    return rig_ptr_->SensorFromRig(sensor_id) * RigFromWorld();
  }
}

bool Frame::operator==(const Frame& other) const {
  return frame_id_ == other.frame_id_ && rig_id_ == other.rig_id_ &&
         data_ids_ == other.data_ids_ &&
         rig_from_world_ == other.rig_from_world_;
}

bool Frame::operator!=(const Frame& other) const { return !(*this == other); }

}  // namespace colmap

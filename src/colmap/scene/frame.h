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
  inline const class Rig* RigPtr() const;
  inline void SetRigPtr(const class Rig* rig);
  inline void ResetRigPtr();
  // Check if the frame has a non-trivial rig.
  inline bool HasRigPtr() const;

  // Access the frame from world transformation.
  inline const Rigid3d& FrameFromWorld() const;
  inline Rigid3d& FrameFromWorld();
  inline const std::optional<Rigid3d>& MaybeFrameFromWorld() const;
  inline std::optional<Rigid3d>& MaybeFrameFromWorld();
  inline void SetFrameFromWorld(const Rigid3d& frame_from_world);
  inline void SetFrameFromWorld(const std::optional<Rigid3d>& frame_from_world);
  inline bool HasPose() const;
  inline void ResetPose();

  // Get the sensor from world transformation.
  inline Rigid3d SensorFromWorld(sensor_t sensor_id) const;

  // Set the world to frame from the given camera to world transformation.
  void ApplyCamFromWorld(camera_t camera_id, const Rigid3d& cam_from_world);

  inline bool operator==(const Frame& other) const;
  inline bool operator!=(const Frame& other) const;

 private:
  frame_t frame_id_ = kInvalidFrameId;
  std::set<data_t> data_ids_;

  // Store the frame_from_world transformation and an optional rig calibration.
  // If the rig calibration is a nullptr, the frame becomes a single sensor
  // case, where rig modeling is no longer needed.
  std::optional<Rigid3d> frame_from_world_;

  // Rig calibration.
  rig_t rig_id_ = kInvalidRigId;
  const class Rig* rig_ptr_ = nullptr;
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

const Rig* Frame::RigPtr() const { return THROW_CHECK_NOTNULL(rig_ptr_); }

void Frame::SetRigPtr(const class Rig* rig) { rig_ptr_ = rig; }

void Frame::ResetRigPtr() { rig_ptr_ = nullptr; }

bool Frame::HasRigPtr() const { return rig_ptr_ != nullptr; }

const Rigid3d& Frame::FrameFromWorld() const {
  THROW_CHECK(frame_from_world_) << "Frame does not have a valid pose.";
  return *frame_from_world_;
}

Rigid3d& Frame::FrameFromWorld() {
  THROW_CHECK(frame_from_world_) << "Frame does not have a valid pose.";
  return *frame_from_world_;
}

const std::optional<Rigid3d>& Frame::MaybeFrameFromWorld() const {
  return frame_from_world_;
}

std::optional<Rigid3d>& Frame::MaybeFrameFromWorld() {
  return frame_from_world_;
}

void Frame::SetFrameFromWorld(const Rigid3d& frame_from_world) {
  frame_from_world_ = frame_from_world;
}

void Frame::SetFrameFromWorld(const std::optional<Rigid3d>& frame_from_world) {
  frame_from_world_ = frame_from_world;
}

bool Frame::HasPose() const { return frame_from_world_.has_value(); }

void Frame::ResetPose() { frame_from_world_.reset(); }

Rigid3d Frame::SensorFromWorld(sensor_t sensor_id) const {
  if (!HasRigPtr() || rig_ptr_->IsRefSensor(sensor_id)) {
    return FrameFromWorld();
  }
  THROW_CHECK(rig_ptr_->HasSensor(sensor_id));
  return rig_ptr_->SensorFromRig(sensor_id) * FrameFromWorld();
}

bool Frame::operator==(const Frame& other) const {
  return frame_id_ == other.frame_id_ && rig_id_ == other.rig_id_ &&
         data_ids_ == other.data_ids_ &&
         frame_from_world_ == other.frame_from_world_;
}

bool Frame::operator!=(const Frame& other) const { return !(*this == other); }

}  // namespace colmap

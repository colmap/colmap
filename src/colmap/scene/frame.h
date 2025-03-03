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
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

namespace colmap {

class Frame {
 public:
  Frame() = default;

  // Access the unique identifier of the frame.
  inline frame_t FrameId() const;
  inline void SetFrameId(frame_t frame_id);

  // Access data ids
  inline std::set<data_t>& DataIds();
  inline const std::set<data_t>& DataIds() const;
  inline void AddDataId(data_t data_id);

  // Check whether the data id is existent in the frame.
  inline bool HasDataId(data_t data_id) const;

  // Access the rig calibration.
  inline const class Rig* Rig() const;
  inline void SetRig(const class Rig* rig);
  // Check if the frame has a non-trivial rig calibration.
  inline bool HasRig() const;

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

  inline bool operator==(const Frame& other) const;
  inline bool operator!=(const Frame& other) const;

 private:
  frame_t frame_id_ = kInvalidFrameId;
  std::set<data_t> data_ids_;

  // Store the frame_from_world transformation and an optional rig calibration.
  // If the rig calibration is a nullptr, the frame becomes a single sensor
  // case, where rig modeling is no longer needed.
  std::optional<Rigid3d> frame_from_world_;

  // [Optional] Rig calibration.
  const class Rig* rig_ = nullptr;
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

const Rig* Frame::Rig() const { return rig_; }

void Frame::SetRig(const class Rig* rig) { rig_ = rig; }

bool Frame::HasRig() const {
  if (rig_ == nullptr)
    return false;
  else
    return rig_->NumSensors() > 1;
}

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
  if (!HasRig() || rig_->IsRefSensor(sensor_id)) {
    return FrameFromWorld();
  }
  THROW_CHECK(rig_->HasSensor(sensor_id));
  return rig_->SensorFromRig(sensor_id) * FrameFromWorld();
}

bool Frame::operator==(const Frame& other) const {
  return frame_id_ == other.frame_id_ && data_ids_ == other.data_ids_ &&
         frame_from_world_ == other.frame_from_world_;
}

bool Frame::operator!=(const Frame& other) const { return !(*this == other); }

}  // namespace colmap

namespace std {

// [Reference]
// https://stackoverflow.com/questions/26705751/why-is-the-magic-number-in-boosthash-combine-specified-in-hex
template <>
struct hash<colmap::data_t> {
  std::size_t operator()(const colmap::data_t& d) const noexcept {
    size_t h1 =
        std::hash<uint64_t>{}(std::hash<colmap::sensor_t>{}(d.sensor_id));
    size_t h2 = std::hash<uint64_t>{}(d.id);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

}  // namespace std

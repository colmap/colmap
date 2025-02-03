// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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
#include "colmap/util/enum_utils.h"
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

// Sensor type
MAKE_ENUM_CLASS_OVERLOAD_STREAM(SensorType,
                                0,
                                CAMERA,
                                IMU,
                                LOCATION  // include GNSS, radios, compass, etc.
);

struct sensor_t {
  SensorType type;
  uint32_t id;
  sensor_t(const SensorType& type, uint32_t id) : type(type), id(id) {}

  bool operator<(const sensor_t& other) const {
    return std::tie(type, id) < std::tie(other.type, other.id);
  }
  bool operator==(const sensor_t& other) const {
    return type == other.type && id == other.id;
  }
};

struct data_t {
  sensor_t sensor_id;
  uint64_t id;
  data_t(const sensor_t& sensor_id, uint64_t id)
      : sensor_id(sensor_id), id(id) {}

  bool operator<(const data_t& other) const {
    return std::tie(sensor_id, id) < std::tie(other.sensor_id, other.id);
  }
  bool operator==(const data_t& other) const {
    return sensor_id == other.sensor_id && id == other.id;
  }
};

// Rig calibration storing the sensor from rig transformation.
// The reference sensor shares identity poses with the device.
// This design is mainly for two purposes:
// 1) In the visual-inertial optimization one of the IMUs is generally used as
// the reference frame since it is metric.
// 2) Not having a reference frame brings a 6 DoF Gauge for each rig, which is
// not ideal particularly when it comes to covariance estimation.
class RigCalibration {
 public:
  // Access the unique identifier of the rig
  inline rig_t RigId() const;
  inline void SetRigId(rig_t rig_id);

  // Add sensor into the rig
  // ``AddReferenceSensor`` needs to called first before all the ``AddSensor``
  // operation
  inline void AddReferenceSensor(sensor_t ref_sensor_id);
  inline void AddSensor(sensor_t sensor_id, const Rigid3d& sensor_from_rig);

  // Check whether the sensor exists in the rig
  inline bool HasSensor(sensor_t sensor_id) const;

  // Count the number of sensors available in the rig
  inline size_t NumSensors() const;

  // Access the reference sensor id (default to be the first added sensor)
  inline sensor_t RefSensorId() const;

  // Check if the sensor is the reference sensor of the rig
  inline bool IsReference(sensor_t sensor_id) const;

  // Access sensor from rig transformations
  inline Rigid3d& SensorFromRig(sensor_t sensor_id);
  inline const Rigid3d& SensorFromRig(sensor_t sensor_id) const;

 private:
  // Unique identifier of the device.
  rig_t rig_id_;

  // Reference sensor id which has the identity transformation to the rig.
  sensor_t ref_sensor_id_;

  // list of sensors
  std::set<sensor_t> sensor_ids_;

  // sensor_from_rig transformation.
  std::map<sensor_t, Rigid3d> sensors_from_rig_;
};

class Frame {
 public:
  // Access the unique identifier of the frame
  inline frame_t FrameId() const;
  inline void SetFrameId(frame_t frame_id);

  // Access data ids
  inline std::set<data_t>& DataIds();
  inline const std::set<data_t>& DataIds() const;
  inline void AddData(data_t data_id);

  // Check whether the data id is existent in the frame
  inline bool HasData(data_t data_id) const;

  // Access the unique identifier of the rig
  inline rig_t RigId() const;
  inline void SetRigId(rig_t rig_id);

  // Check if the frame has a non-trivial rig calibration
  inline const std::shared_ptr<class RigCalibration>& RigCalibration() const;
  inline void SetRigCalibration(
      std::shared_ptr<class RigCalibration> rig_calibration);
  inline bool HasRigCalibration() const;

  // Access the frame from world transformation
  inline const Rigid3d& FrameFromWorld() const;
  inline Rigid3d& FrameFromWorld();
  inline const std::optional<Rigid3d>& MaybeFrameFromWorld() const;
  inline std::optional<Rigid3d>& MaybeFrameFromWorld();
  inline void SetFrameFromWorld(const Rigid3d& frame_from_world);
  inline void SetFrameFromWorld(const std::optional<Rigid3d>& frame_from_world);
  inline bool HasPose() const;
  inline void ResetPose();

  // Get the sensor from world transformation
  inline Rigid3d SensorFromWorld(sensor_t sensor_id) const;

 private:
  frame_t frame_id_;
  std::set<data_t> data_ids_;

  // Store the frame_from_world transformation and an optional rig calibration.
  // If the rig calibration is a nullptr, the frame becomes a single sensor
  // case, where rig modeling is no longer needed.
  std::optional<Rigid3d> frame_from_world_;
  rig_t rig_id_ = kInvalidRigId;
  std::shared_ptr<class RigCalibration> rig_calibration_ = nullptr;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

rig_t RigCalibration::RigId() const { return rig_id_; }

void RigCalibration::SetRigId(rig_t rig_id) { rig_id_ = rig_id; }

bool RigCalibration::HasSensor(sensor_t sensor_id) const {
  return sensor_ids_.find(sensor_id) != sensor_ids_.end();
}

size_t RigCalibration::NumSensors() const { return sensor_ids_.size(); }

void RigCalibration::AddReferenceSensor(sensor_t ref_sensor_id) {
  THROW_CHECK(sensor_ids_.empty());  // The reference sensor must be added first
  ref_sensor_id_ = ref_sensor_id;
  sensor_ids_.insert(ref_sensor_id);
}

void RigCalibration::AddSensor(sensor_t sensor_id,
                               const Rigid3d& sensor_from_rig) {
  if (NumSensors() == 0)
    LOG(FATAL_THROW) << "The reference sensor needs to added first before any "
                        "sensor being added.";
  else {
    THROW_CHECK(!HasSensor(sensor_id))
        << StringPrintf("Sensor id (%d, %d) is inserted twice into the rig",
                        sensor_id.type,
                        sensor_id.id);
    sensors_from_rig_.emplace(sensor_id, sensor_from_rig);
    sensor_ids_.insert(sensor_id);
  }
}

sensor_t RigCalibration::RefSensorId() const { return ref_sensor_id_; }

bool RigCalibration::IsReference(sensor_t sensor_id) const {
  return sensor_id == ref_sensor_id_;
}

Rigid3d& RigCalibration::SensorFromRig(sensor_t sensor_id) {
  THROW_CHECK(!IsReference(sensor_id));
  if (sensors_from_rig_.find(sensor_id) == sensors_from_rig_.end())
    LOG(FATAL_THROW) << StringPrintf(
        "Sensor id (%d, %d) not found in the rig calibration",
        sensor_id.type,
        sensor_id.id);
  return sensors_from_rig_.at(sensor_id);
}

const Rigid3d& RigCalibration::SensorFromRig(sensor_t sensor_id) const {
  THROW_CHECK(!IsReference(sensor_id));
  auto it = sensors_from_rig_.find(sensor_id);
  if (it == sensors_from_rig_.end())
    LOG(FATAL_THROW) << StringPrintf(
        "Sensor id (%d, %d) not found in the rig calibration",
        sensor_id.type,
        sensor_id.id);
  return sensors_from_rig_.at(sensor_id);
}

frame_t Frame::FrameId() const { return frame_id_; }

void Frame::SetFrameId(frame_t frame_id) { frame_id_ = frame_id; }

std::set<data_t>& Frame::DataIds() { return data_ids_; }

const std::set<data_t>& Frame::DataIds() const { return data_ids_; }

void Frame::AddData(data_t data_id) { data_ids_.insert(data_id); }

bool Frame::HasData(data_t data_id) const {
  return data_ids_.find(data_id) != data_ids_.end();
}

rig_t Frame::RigId() const { return rig_id_; }

void Frame::SetRigId(rig_t rig_id) { rig_id_ = rig_id; }

const std::shared_ptr<class RigCalibration>& Frame::RigCalibration() const {
  return rig_calibration_;
}

void Frame::SetRigCalibration(
    std::shared_ptr<class RigCalibration> rig_calibration) {
  rig_calibration_ = std::move(rig_calibration);
}

bool Frame::HasRigCalibration() const {
  if (!rig_calibration_)
    return false;
  else
    return rig_calibration_->NumSensors() > 1;
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
  if (!HasRigCalibration() || rig_calibration_->IsReference(sensor_id)) {
    return FrameFromWorld();
  }
  THROW_CHECK(rig_calibration_->HasSensor(sensor_id));
  return rig_calibration_->SensorFromRig(sensor_id) * FrameFromWorld();
}

}  // namespace colmap

namespace std {
template <>
struct hash<colmap::sensor_t> {
  std::size_t operator()(const colmap::sensor_t& s) const noexcept {
    return std::hash<std::pair<uint32_t, uint32_t>>{}(
        std::make_pair(static_cast<uint32_t>(s.type), s.id));
  }
};

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

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
#include "colmap/util/types.h"

#include <map>
#include <set>
#include <vector>

namespace colmap {

// Sensor type
enum class SensorType {
  Camera = 0,
  IMU = 1,
  Location = 2,  // include GNSS, radios, compass, etc.
};

// Unique identifier of the sensor
typedef std::pair<SensorType, uint32_t> sensor_t;

// Unique identifier of the data point from a sensor
typedef std::pair<sensor_t, uint64_t> data_t;

// Rig calibration storing the sensor from rig transformation
class RigCalibration {
 public:
  // Access the unique identifier of the rig
  inline rig_t RigId() const;
  inline void SetRigId(rig_t rig_id);

  // Add sensor into the rig
  // ``AddReferenceSensor`` needs to called first before all the ``AddSensor``
  // operation
  inline void AddReferenceSensor(sensor_t ref_sensor_id);
  inline void AddSensor(sensor_t sensor_id,
                        Rigid3d sensor_from_rig = Rigid3d(),
                        bool is_fixed = false);

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
  std::map<sensor_t, Rigid3d> map_sensor_from_rig_;
  std::map<sensor_t, bool> is_fixed_sensor_from_rig_;  // for optimization
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
  inline const std::shared_ptr<class RigCalibration> RigCalibration() const;
  inline void SetRigCalibration(
      std::shared_ptr<class RigCalibration> rig_calibration);
  inline bool HasRigCalibration() const;

  // Access the frame from world transformation
  inline const Rigid3d& FrameFromWorld() const;
  inline Rigid3d& FrameFromWorld();
  inline const Rigid3d& SensorFromWorld() const;
  inline Rigid3d& SensorFromWorld();

  // Get the sensor from world transformation
  inline Rigid3d SensorFromWorld(sensor_t sensor_id) const;

 private:
  frame_t frame_id_;
  std::set<data_t> data_ids_;

  // Store the frame_from_world transformation and an optional rig calibration.
  // If the rig calibration is a nullptr, the frame becomes a single sensor
  // case, where rig modeling is no longer needed.
  Rigid3d frame_from_world_;
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
                               Rigid3d sensor_from_rig,
                               bool is_fixed) {
  if (NumSensors() == 0)
    LOG(FATAL_THROW) << "The reference sensor needs to added first before any "
                        "sensor being added.";
  else {
    if (HasSensor(sensor_id))
      LOG(FATAL_THROW) << StringPrintf(
          "Sensor id (%d, %d) is inserted twice into the rig",
          sensor_id.first,
          sensor_id.second);
    map_sensor_from_rig_.emplace(sensor_id, sensor_from_rig);
    is_fixed_sensor_from_rig_.emplace(sensor_id, is_fixed);
    sensor_ids_.insert(sensor_id);
  }
}

sensor_t RigCalibration::RefSensorId() const { return ref_sensor_id_; }

bool RigCalibration::IsReference(sensor_t sensor_id) const {
  return sensor_id == ref_sensor_id_;
}

Rigid3d& RigCalibration::SensorFromRig(sensor_t sensor_id) {
  THROW_CHECK(!IsReference(sensor_id));
  if (map_sensor_from_rig_.find(sensor_id) == map_sensor_from_rig_.end())
    LOG(FATAL_THROW) << StringPrintf(
        "Sensor id (%d, %d) not found in the rig calibration",
        sensor_id.first,
        sensor_id.second);
  return map_sensor_from_rig_.at(sensor_id);
}

const Rigid3d& RigCalibration::SensorFromRig(sensor_t sensor_id) const {
  THROW_CHECK(!IsReference(sensor_id));
  auto it = map_sensor_from_rig_.find(sensor_id);
  if (it == map_sensor_from_rig_.end())
    LOG(FATAL_THROW) << StringPrintf(
        "Sensor id (%d, %d) not found in the rig calibration",
        sensor_id.first,
        sensor_id.second);
  return map_sensor_from_rig_.at(sensor_id);
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

const std::shared_ptr<class RigCalibration> Frame::RigCalibration() const {
  return rig_calibration_;
}

void Frame::SetRigCalibration(
    std::shared_ptr<class RigCalibration> rig_calibration) {
  rig_calibration_ = rig_calibration;
}

bool Frame::HasRigCalibration() const {
  if (!rig_calibration_)
    return false;
  else
    return rig_calibration_->NumSensors() > 1;
}

const Rigid3d& Frame::FrameFromWorld() const { return frame_from_world_; }
Rigid3d& Frame::FrameFromWorld() { return frame_from_world_; }

const Rigid3d& Frame::SensorFromWorld() const {
  THROW_CHECK(!HasRigCalibration());
  return FrameFromWorld();
}
Rigid3d& Frame::SensorFromWorld() {
  THROW_CHECK(!HasRigCalibration());
  return FrameFromWorld();
}

Rigid3d Frame::SensorFromWorld(sensor_t sensor_id) const {
  if (!HasRigCalibration() || rig_calibration_->IsReference(sensor_id)) {
    return SensorFromWorld();
  }
  THROW_CHECK(rig_calibration_->HasSensor(sensor_id));
  return rig_calibration_->SensorFromRig(sensor_id) * frame_from_world_;
}

}  // namespace colmap

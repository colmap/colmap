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
#include "colmap/util/enum_utils.h"
#include "colmap/util/types.h"

#include <map>
#include <optional>

namespace colmap {

// Rigs represent a collection of rigidly mounted sensors and the associated
// sensor from rig transformations. The reference sensor is defined to have
// identity pose in the rig frame. This design is mainly for two purposes: 1) In
// visual-inertial optimization, one of the IMUs is generally used as the
// reference frame since it is metric. 2) Not having a reference frame brings a
// 6 DoF Gauge for each rig, which is not ideal particularly when it comes to
// covariance estimation.
class Rig {
 public:
  // Access the unique identifier of the rig.
  inline rig_t RigId() const;
  inline void SetRigId(rig_t rig_id);

  // Add sensor into the rig. ``AddRefSensor`` needs to called first before all
  // the ``AddSensor`` operations.
  void AddRefSensor(sensor_t ref_sensor_id);
  void AddSensor(sensor_t sensor_id,
                 const std::optional<Rigid3d>& sensor_from_rig = std::nullopt);

  // Check whether the sensor exists in the rig.
  inline bool HasSensor(sensor_t sensor_id) const;

  // Count the number of sensors available in the rig.
  inline size_t NumSensors() const;

  // Access the reference sensor id (default to be the first added sensor).
  inline sensor_t RefSensorId() const;

  // Check if the sensor is the reference sensor of the rig.
  inline bool IsRefSensor(sensor_t sensor_id) const;
  inline bool HasSensorFromRig(sensor_t sensor_id) const;

  // Access all sensors in the rig except for the reference sensor.
  inline const std::map<sensor_t, std::optional<Rigid3d>>& Sensors() const;
  inline std::map<sensor_t, std::optional<Rigid3d>>& Sensors();

  // Access sensor from rig transformations.
  inline Rigid3d& SensorFromRig(sensor_t sensor_id);
  inline const Rigid3d& SensorFromRig(sensor_t sensor_id) const;
  inline std::optional<Rigid3d>& MaybeSensorFromRig(sensor_t sensor_id);
  inline const std::optional<Rigid3d>& MaybeSensorFromRig(
      sensor_t sensor_id) const;
  inline void SetSensorFromRig(sensor_t sensor_id,
                               const Rigid3d& sensor_from_rig);
  inline void SetSensorFromRig(sensor_t sensor_id,
                               const std::optional<Rigid3d>& sensor_from_rig);
  inline void ResetSensorFromRig(sensor_t sensor_id);

  inline bool operator==(const Rig& other) const;
  inline bool operator!=(const Rig& other) const;

 private:
  inline std::optional<Rigid3d>& FindSensorFromRigOrThrow(sensor_t sensor_id);
  inline const std::optional<Rigid3d>& FindSensorFromRigOrThrow(
      sensor_t sensor_id) const;

  // Unique identifier of the rig.
  rig_t rig_id_ = kInvalidRigId;

  // Reference sensor id which has the identity transformation to the rig.
  sensor_t ref_sensor_id_ = kInvalidSensorId;

  // sensor_from_rig transformations.
  std::map<sensor_t, std::optional<Rigid3d>> sensors_from_rig_;
};

std::ostream& operator<<(std::ostream& stream, const Rig& rig);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

rig_t Rig::RigId() const { return rig_id_; }

void Rig::SetRigId(rig_t rig_id) { rig_id_ = rig_id; }

bool Rig::HasSensor(sensor_t sensor_id) const {
  return sensor_id == ref_sensor_id_ ||
         sensors_from_rig_.find(sensor_id) != sensors_from_rig_.end();
}

size_t Rig::NumSensors() const {
  size_t num_sensors = sensors_from_rig_.size();
  if (ref_sensor_id_ != kInvalidSensorId) num_sensors += 1;
  return num_sensors;
}

sensor_t Rig::RefSensorId() const { return ref_sensor_id_; }

bool Rig::IsRefSensor(sensor_t sensor_id) const {
  return sensor_id == ref_sensor_id_;
}

const std::map<sensor_t, std::optional<Rigid3d>>& Rig::Sensors() const {
  return sensors_from_rig_;
}

std::map<sensor_t, std::optional<Rigid3d>>& Rig::Sensors() {
  return sensors_from_rig_;
}

Rigid3d& Rig::SensorFromRig(sensor_t sensor_id) {
  return FindSensorFromRigOrThrow(sensor_id).value();
}

const Rigid3d& Rig::SensorFromRig(sensor_t sensor_id) const {
  return FindSensorFromRigOrThrow(sensor_id).value();
}

std::optional<Rigid3d>& Rig::MaybeSensorFromRig(sensor_t sensor_id) {
  return FindSensorFromRigOrThrow(sensor_id);
}

const std::optional<Rigid3d>& Rig::MaybeSensorFromRig(
    sensor_t sensor_id) const {
  return FindSensorFromRigOrThrow(sensor_id);
}

void Rig::SetSensorFromRig(sensor_t sensor_id, const Rigid3d& sensor_from_rig) {
  FindSensorFromRigOrThrow(sensor_id) = sensor_from_rig;
}

void Rig::SetSensorFromRig(sensor_t sensor_id,
                           const std::optional<Rigid3d>& sensor_from_rig) {
  FindSensorFromRigOrThrow(sensor_id) = sensor_from_rig;
}

void Rig::ResetSensorFromRig(sensor_t sensor_id) {
  FindSensorFromRigOrThrow(sensor_id).reset();
}

bool Rig::operator==(const Rig& other) const {
  return rig_id_ == other.rig_id_ && ref_sensor_id_ == other.ref_sensor_id_ &&
         sensors_from_rig_ == other.sensors_from_rig_;
}

bool Rig::operator!=(const Rig& other) const { return !(*this == other); }

inline std::optional<Rigid3d>& Rig::FindSensorFromRigOrThrow(
    sensor_t sensor_id) {
  THROW_CHECK(sensor_id != ref_sensor_id_)
      << "The reference sensor does not have a SensorFromRig transformation, "
         "which is fixed to identity";
  auto it = sensors_from_rig_.find(sensor_id);
  THROW_CHECK(it != sensors_from_rig_.end())
      << "Sensor (" << sensor_id.type << ", " << sensor_id.id
      << ") not found in the rig";
  return it->second;
}

inline const std::optional<Rigid3d>& Rig::FindSensorFromRigOrThrow(
    sensor_t sensor_id) const {
  THROW_CHECK(sensor_id != ref_sensor_id_)
      << "The reference sensor does not have a SensorFromRig transformation, "
         "which is fixed to identity";
  auto it = sensors_from_rig_.find(sensor_id);
  THROW_CHECK(it != sensors_from_rig_.end())
      << "Sensor (" << sensor_id.type << ", " << sensor_id.id
      << ") not found in the rig";
  return it->second;
}

}  // namespace colmap

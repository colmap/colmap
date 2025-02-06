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
MAKE_ENUM_CLASS_OVERLOAD_STREAM(SensorType, -1, INVALID, CAMERA, IMU);

struct sensor_t {
  // Type of the sensor (INVALID / CAMERA / IMU)
  SensorType type;
  // Unique identifier of the sensor.
  // This can be camera_t / imu_t (not supported yet)
  uint32_t id;
  constexpr sensor_t(const SensorType& type, uint32_t id)
      : type(type), id(id) {}

  inline bool operator<(const sensor_t& other) const {
    return std::tie(type, id) < std::tie(other.type, other.id);
  }
  inline bool operator==(const sensor_t& other) const {
    return type == other.type && id == other.id;
  }
  inline bool operator!=(const sensor_t& other) const {
    return !(*this == other);
  }
};
constexpr sensor_t kInvalidSensorId =
    sensor_t(SensorType::INVALID, std::numeric_limits<uint32_t>::max());

// Rig calibration storing the sensor from rig transformation.
// The reference sensor shares identity poses with the device.
// This design is mainly for two purposes:
// 1) In the visual-inertial optimization one of the IMUs is generally used as
// the reference frame since it is metric.
// 2) Not having a reference frame brings a 6 DoF Gauge for each rig, which is
// not ideal particularly when it comes to covariance estimation.
class RigCalib {
 public:
  RigCalib() = default;

  // Access the unique identifier of the rig
  inline rig_t RigId() const;
  inline void SetRigId(rig_t rig_id);

  // Add sensor into the rig. ``AddRefSensor`` needs to called first before all
  // the ``AddSensor`` operations
  void AddRefSensor(sensor_t ref_sensor_id);
  void AddSensor(sensor_t sensor_id,
                 const std::optional<Rigid3d>& sensor_from_rig = std::nullopt);

  // Check whether the sensor exists in the rig
  inline bool HasSensor(sensor_t sensor_id) const;

  // Count the number of sensors available in the rig
  inline size_t NumSensors() const;

  // Access the reference sensor id (default to be the first added sensor)
  inline sensor_t RefSensorId() const;

  // Check if the sensor is the reference sensor of the rig
  inline bool IsRefSensor(sensor_t sensor_id) const;

  // Access sensor from rig transformations
  inline Rigid3d& SensorFromRig(sensor_t sensor_id);
  inline const Rigid3d& SensorFromRig(sensor_t sensor_id) const;
  inline std::optional<Rigid3d>& MaybeSensorFromRig(sensor_t sensor_id);
  inline const std::optional<Rigid3d>& MaybeSensorFromRig(
      sensor_t sensor_id) const;
  inline void SetSensorFromRig(sensor_t sensor_id,
                               const Rigid3d& sensor_from_rig);
  inline void SetSensorFromRig(sensor_t sensor_id,
                               const std::optional<Rigid3d>& sensor_from_rig);
  inline bool HasSensorFromRig(sensor_t sensor_id) const;
  inline void ResetSensorFromRig(sensor_t sensor_id);

 private:
  // Unique identifier of the device.
  rig_t rig_id_ = kInvalidRigId;

  // Reference sensor id which has the identity transformation to the rig.
  sensor_t ref_sensor_id_ = kInvalidSensorId;

  // sensor_from_rig transformation.
  std::map<sensor_t, std::optional<Rigid3d>> sensors_from_rig_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

rig_t RigCalib::RigId() const { return rig_id_; }

void RigCalib::SetRigId(rig_t rig_id) { rig_id_ = rig_id; }

bool RigCalib::HasSensor(sensor_t sensor_id) const {
  return sensor_id == ref_sensor_id_ ||
         sensors_from_rig_.find(sensor_id) != sensors_from_rig_.end();
}

size_t RigCalib::NumSensors() const {
  size_t n_sensors = sensors_from_rig_.size();
  if (ref_sensor_id_ != kInvalidSensorId) n_sensors += 1;
  return n_sensors;
}

sensor_t RigCalib::RefSensorId() const { return ref_sensor_id_; }

bool RigCalib::IsRefSensor(sensor_t sensor_id) const {
  return sensor_id == ref_sensor_id_;
}

Rigid3d& RigCalib::SensorFromRig(sensor_t sensor_id) {
  THROW_CHECK(!IsRefSensor(sensor_id))
      << "No reference is available for the SensorFromRig transformation of "
         "the reference sensor, which is identity";
  if (sensors_from_rig_.find(sensor_id) == sensors_from_rig_.end())
    LOG(FATAL_THROW) << StringPrintf(
        "Sensor id (%d, %d) not found in the rig calibration",
        sensor_id.type,
        sensor_id.id);
  THROW_CHECK(sensors_from_rig_.at(sensor_id))
      << "The corresponding sensor does not have a valid transformation.";
  return *sensors_from_rig_.at(sensor_id);
}

const Rigid3d& RigCalib::SensorFromRig(sensor_t sensor_id) const {
  THROW_CHECK(!IsRefSensor(sensor_id))
      << "No reference is available for the SensorFromRig transformation of "
         "the reference sensor, which is identity";
  if (sensors_from_rig_.find(sensor_id) == sensors_from_rig_.end())
    LOG(FATAL_THROW) << StringPrintf(
        "Sensor id (%d, %d) not found in the rig calibration",
        sensor_id.type,
        sensor_id.id);
  THROW_CHECK(sensors_from_rig_.at(sensor_id))
      << "The corresponding sensor does not have a valid transformation.";
  return *sensors_from_rig_.at(sensor_id);
}

std::optional<Rigid3d>& RigCalib::MaybeSensorFromRig(sensor_t sensor_id) {
  THROW_CHECK(!IsRefSensor(sensor_id))
      << "No reference is available for the SensorFromRig transformation of "
         "the reference sensor, which is identity";
  if (sensors_from_rig_.find(sensor_id) == sensors_from_rig_.end())
    LOG(FATAL_THROW) << StringPrintf(
        "Sensor id (%d, %d) not found in the rig calibration",
        sensor_id.type,
        sensor_id.id);
  return sensors_from_rig_.at(sensor_id);
}

const std::optional<Rigid3d>& RigCalib::MaybeSensorFromRig(
    sensor_t sensor_id) const {
  THROW_CHECK(!IsRefSensor(sensor_id))
      << "No reference is available for the SensorFromRig transformation of "
         "the reference sensor, which is identity";
  if (sensors_from_rig_.find(sensor_id) == sensors_from_rig_.end())
    LOG(FATAL_THROW) << StringPrintf(
        "Sensor id (%d, %d) not found in the rig calibration",
        sensor_id.type,
        sensor_id.id);
  return sensors_from_rig_.at(sensor_id);
}

void RigCalib::SetSensorFromRig(sensor_t sensor_id,
                                const Rigid3d& sensor_from_rig) {
  THROW_CHECK(!IsRefSensor(sensor_id))
      << "Cannot set the SensorFromRig transformation of the reference sensor, "
         "which is fixed to identity";
  if (sensors_from_rig_.find(sensor_id) == sensors_from_rig_.end())
    LOG(FATAL_THROW) << StringPrintf(
        "Sensor id (%d, %d) not found in the rig calibration",
        sensor_id.type,
        sensor_id.id);
  sensors_from_rig_.at(sensor_id) = sensor_from_rig;
}

void RigCalib::SetSensorFromRig(sensor_t sensor_id,
                                const std::optional<Rigid3d>& sensor_from_rig) {
  THROW_CHECK(!IsRefSensor(sensor_id))
      << "Cannot set the SensorFromRig transformation of the reference sensor, "
         "which is fixed to identity";
  if (sensors_from_rig_.find(sensor_id) == sensors_from_rig_.end())
    LOG(FATAL_THROW) << StringPrintf(
        "Sensor id (%d, %d) not found in the rig calibration",
        sensor_id.type,
        sensor_id.id);
  sensors_from_rig_.at(sensor_id) = sensor_from_rig;
}

bool RigCalib::HasSensorFromRig(sensor_t sensor_id) const {
  if (IsRefSensor(sensor_id))
    return true;  // SensorFromRig for the reference sensor is always identity
  if (sensors_from_rig_.find(sensor_id) == sensors_from_rig_.end())
    LOG(FATAL_THROW) << StringPrintf(
        "Sensor id (%d, %d) not found in the rig calibration",
        sensor_id.type,
        sensor_id.id);
  return sensors_from_rig_.at(sensor_id).has_value();
}

void RigCalib::ResetSensorFromRig(sensor_t sensor_id) {
  THROW_CHECK(!IsRefSensor(sensor_id))
      << "Cannot reset the SensorFromRig transformation of the reference "
         "sensor, "
         "which is fixed to identity";
  if (sensors_from_rig_.find(sensor_id) == sensors_from_rig_.end())
    LOG(FATAL_THROW) << StringPrintf(
        "Sensor id (%d, %d) not found in the rig calibration",
        sensor_id.type,
        sensor_id.id);
  sensors_from_rig_.at(sensor_id).reset();
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

}  // namespace std

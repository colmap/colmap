// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/sensor/rig.h"

namespace colmap {

void Rig::AddRefSensor(sensor_t ref_sensor_id) {
  THROW_CHECK(ref_sensor_id_ == kInvalidSensorId)
      << "Reference sensor already set";
  ref_sensor_id_ = ref_sensor_id;
}

void Rig::AddSensor(sensor_t sensor_id,
                    const std::optional<Rigid3d>& sensor_from_rig) {
  THROW_CHECK_GE(NumSensors(), 1)
      << "The reference sensor needs to be added first before other sensors.";
  THROW_CHECK(!HasSensor(sensor_id))
      << "Sensor (" << sensor_id.type << ", " << sensor_id.id
      << ") is inserted twice into the rig";
  sensors_from_rig_.emplace(sensor_id, sensor_from_rig);
}

std::ostream& operator<<(std::ostream& stream, const Rig& rig) {
  const std::string rig_id_str =
      rig.RigId() != kInvalidRigId ? std::to_string(rig.RigId()) : "Invalid";
  stream << "Rig(rig_id=" << rig_id_str << ", ref_sensor_id=("
         << rig.RefSensorId().type << ", " << rig.RefSensorId().id
         << "), sensors=[";
  for (auto it = rig.NonRefSensors().begin();
       it != rig.NonRefSensors().end();) {
    stream << "(" << it->first.type << ", " << it->first.id << ")";
    if (++it != rig.NonRefSensors().end()) {
      stream << ", ";
    }
  }
  stream << "])";
  return stream;
}

}  // namespace colmap

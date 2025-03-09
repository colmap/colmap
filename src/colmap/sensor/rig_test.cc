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

#include "colmap/sensor/rig.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

Rigid3d TestRigid3d() {
  return Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
}

TEST(Rig, Default) {
  Rig rig;
  EXPECT_EQ(rig.RigId(), kInvalidRigId);
  EXPECT_EQ(rig.RefSensorId(), kInvalidSensorId);
  EXPECT_EQ(rig.NumSensors(), 0);
  EXPECT_EQ(rig.Sensors().size(), 0);
}

TEST(Rig, SetUp) {
  Rig rig;
  const sensor_t sensor_id0(SensorType::IMU, 0);
  rig.AddRefSensor(sensor_id0);
  const sensor_t sensor_id1(SensorType::IMU, 1);
  const Rigid3d sensor1_from_rig = TestRigid3d();
  rig.AddSensor(sensor_id1, sensor1_from_rig);
  const sensor_t sensor_id2(SensorType::CAMERA, 0);
  const Rigid3d sensor2_from_rig = TestRigid3d();
  rig.AddSensor(sensor_id2, sensor2_from_rig);
  const sensor_t sensor_id3(SensorType::CAMERA, 1);
  rig.AddSensor(sensor_id3);  // no input sensor_from_rig

  EXPECT_EQ(rig.NumSensors(), 4);
  EXPECT_EQ(rig.Sensors().size(), 3);

  EXPECT_EQ(rig.RefSensorId().type, SensorType::IMU);
  EXPECT_EQ(rig.RefSensorId().id, 0);

  EXPECT_TRUE(rig.IsRefSensor(sensor_id0));
  EXPECT_FALSE(rig.IsRefSensor(sensor_id1));
  EXPECT_FALSE(rig.IsRefSensor(sensor_id2));
  EXPECT_FALSE(rig.IsRefSensor(sensor_id3));

  EXPECT_EQ(rig.SensorFromRig(sensor_id1), sensor1_from_rig);
  EXPECT_EQ(rig.MaybeSensorFromRig(sensor_id1).value(), sensor1_from_rig);

  EXPECT_EQ(rig.SensorFromRig(sensor_id2), sensor2_from_rig);
  EXPECT_EQ(rig.MaybeSensorFromRig(sensor_id2).value(), sensor2_from_rig);

  EXPECT_TRUE(rig.HasSensor(sensor_id3));
  EXPECT_ANY_THROW(rig.SensorFromRig(sensor_id3));
  EXPECT_EQ(rig.MaybeSensorFromRig(sensor_id3), std::nullopt);
  const Rigid3d sensor3_from_rig = TestRigid3d();
  rig.SetSensorFromRig(sensor_id3, sensor3_from_rig);
  EXPECT_EQ(rig.SensorFromRig(sensor_id3), sensor3_from_rig);
  EXPECT_EQ(rig.MaybeSensorFromRig(sensor_id3).value(), sensor3_from_rig);
}

TEST(Rig, Print) {
  Rig rig;
  rig.SetRigId(0);
  rig.AddRefSensor(sensor_t(SensorType::IMU, 0));
  rig.AddSensor(sensor_t(SensorType::CAMERA, 1), Rigid3d());
  rig.AddSensor(sensor_t(SensorType::CAMERA, 2), Rigid3d());
  std::ostringstream stream;
  stream << rig;
  EXPECT_EQ(stream.str(),
            "Rig(rig_id=0, ref_sensor_id=(IMU, 0), sensors=[(CAMERA, 1), "
            "(CAMERA, 2)])");
}

}  // namespace
}  // namespace colmap

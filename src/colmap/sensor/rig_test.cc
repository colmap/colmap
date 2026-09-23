// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/sensor/rig.h"

#include "colmap/math/random_eigen.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

Rigid3d TestRigid3d() {
  return Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>());
}

TEST(Rig, Default) {
  Rig rig;
  EXPECT_EQ(rig.RigId(), kInvalidRigId);
  EXPECT_EQ(rig.RefSensorId(), kInvalidSensorId);
  EXPECT_EQ(rig.NumSensors(), 0);
  EXPECT_EQ(rig.NonRefSensors().size(), 0);
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
  EXPECT_EQ(rig.NonRefSensors().size(), 3);
  EXPECT_THAT(rig.SensorIds(),
              testing::UnorderedElementsAre(
                  sensor_id0, sensor_id1, sensor_id2, sensor_id3));

  EXPECT_EQ(rig.RefSensorId().type, SensorType::IMU);
  EXPECT_EQ(rig.RefSensorId().id, 0);

  EXPECT_TRUE(rig.IsRefSensor(sensor_id0));
  EXPECT_FALSE(rig.IsRefSensor(sensor_id1));
  EXPECT_FALSE(rig.IsRefSensor(sensor_id2));
  EXPECT_FALSE(rig.IsRefSensor(sensor_id3));

  EXPECT_FALSE(rig.HasSensorFromRig(sensor_id0));
  EXPECT_TRUE(rig.HasSensorFromRig(sensor_id1));
  EXPECT_TRUE(rig.HasSensorFromRig(sensor_id2));
  EXPECT_FALSE(rig.HasSensorFromRig(sensor_id3));  // no sensor_from_rig

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

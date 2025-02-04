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

#include "colmap/scene/frame.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

Rigid3d TestRigid3d() {
  return Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
}

TEST(RigCalibration, Default) {
  RigCalibration calib;
  EXPECT_EQ(calib.RigId(), kInvalidRigId);
  EXPECT_EQ(calib.RefSensorId(), kInvalidSensorId);
  EXPECT_EQ(calib.NumSensors(), 0);
}

TEST(RigCalibration, SetUp) {
  RigCalibration calib;
  calib.AddRefSensor(sensor_t(SensorType::IMU, 0));
  calib.AddSensor(sensor_t(SensorType::IMU, 1), TestRigid3d());
  calib.AddSensor(sensor_t(SensorType::CAMERA, 0), TestRigid3d());
  calib.AddSensor(sensor_t(SensorType::CAMERA, 1));  // no input sensor_from_rig

  EXPECT_EQ(calib.NumSensors(), 4);
  EXPECT_EQ(calib.RefSensorId().type, SensorType::IMU);
  EXPECT_EQ(calib.RefSensorId().id, 0);
  EXPECT_TRUE(calib.IsRefSensor(sensor_t(SensorType::IMU, 0)));
  EXPECT_FALSE(calib.IsRefSensor(sensor_t(SensorType::IMU, 1)));
  EXPECT_TRUE(calib.HasSensorFromRig(sensor_t(SensorType::IMU, 0)));
  EXPECT_TRUE(calib.HasSensorFromRig(sensor_t(SensorType::IMU, 1)));
  EXPECT_TRUE(calib.HasSensorFromRig(sensor_t(SensorType::CAMERA, 0)));
  EXPECT_FALSE(calib.HasSensorFromRig(sensor_t(SensorType::CAMERA, 1)));
  EXPECT_TRUE(calib.HasSensor(sensor_t(SensorType::CAMERA, 1)));
}

TEST(Frame, Default) {
  Frame frame;
  EXPECT_EQ(frame.FrameId(), kInvalidFrameId);
  EXPECT_FALSE(frame.HasPose());
  EXPECT_FALSE(frame.HasRigCalibration());
}

TEST(Frame, SetUp) {
  sensor_t s1(SensorType::IMU, 0);
  sensor_t s2(SensorType::CAMERA, 0);
  std::shared_ptr<RigCalibration> calib = std::make_shared<RigCalibration>();
  calib->AddRefSensor(s1);
  calib->AddSensor(s2, TestRigid3d());

  Frame frame;
  frame.AddData(data_t(s1, 2));
  frame.AddData(data_t(s2, 5));
  frame.SetRigCalibration(calib);
  EXPECT_TRUE(frame.HasRigCalibration());
  EXPECT_FALSE(frame.HasPose());
}

TEST(Frame, SetResetPose) {
  Frame frame;
  EXPECT_FALSE(frame.HasPose());
  EXPECT_ANY_THROW(frame.FrameFromWorld());
  frame.SetFrameFromWorld(Rigid3d());
  EXPECT_TRUE(frame.HasPose());
  EXPECT_EQ(frame.FrameFromWorld().rotation.coeffs(),
            Eigen::Quaterniond::Identity().coeffs());
  EXPECT_EQ(frame.FrameFromWorld().translation, Eigen::Vector3d::Zero());
  frame.ResetPose();
  EXPECT_FALSE(frame.HasPose());
  EXPECT_ANY_THROW(frame.FrameFromWorld());
}

}  // namespace
}  // namespace colmap

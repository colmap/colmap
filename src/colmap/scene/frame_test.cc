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

#include "colmap/util/eigen_matchers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

Rigid3d TestRigid3d() {
  return Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
}

TEST(Frame, Default) {
  Frame frame;
  EXPECT_EQ(frame.FrameId(), kInvalidFrameId);
  EXPECT_FALSE(frame.HasPose());
  EXPECT_EQ(frame.RigId(), kInvalidRigId);
  EXPECT_FALSE(frame.HasRigId());
  EXPECT_FALSE(frame.HasRigPtr());
}

TEST(Frame, SetUp) {
  Frame frame;
  Rig rig;

  EXPECT_FALSE(frame.HasRigId());
  EXPECT_FALSE(frame.HasRigPtr());

  frame.SetRigId(1);
  EXPECT_TRUE(frame.HasRigId());
  EXPECT_FALSE(frame.HasRigPtr());

  const sensor_t sensor_id1(SensorType::IMU, 0);
  rig.AddRefSensor(sensor_id1);
  const sensor_t sensor_id2(SensorType::CAMERA, 0);
  rig.AddSensor(sensor_id2, TestRigid3d());
  frame.SetRigPtr(&rig);
  EXPECT_TRUE(frame.HasRigPtr());

  EXPECT_THAT(frame.DataIds(), testing::IsEmpty());
  const data_t data_id1(sensor_id1, 2);
  EXPECT_FALSE(frame.HasDataId(data_id1));
  frame.AddDataId(data_id1);
  EXPECT_TRUE(frame.HasDataId(data_id1));
  EXPECT_THAT(frame.DataIds(), testing::UnorderedElementsAre(data_id1));
  const data_t data_id2(sensor_id2, 5);
  EXPECT_FALSE(frame.HasDataId(data_id2));
  frame.AddDataId(data_id2);
  EXPECT_TRUE(frame.HasDataId(data_id2));
  EXPECT_THAT(frame.DataIds(),
              testing::UnorderedElementsAre(data_id1, data_id2));
  EXPECT_FALSE(frame.HasPose());
}

TEST(Frame, ImageIds) {
  Frame frame;
  const data_t data_id1(sensor_t(SensorType::IMU, 0), 2);
  frame.AddDataId(data_id1);
  const data_t data_id2(sensor_t(SensorType::CAMERA, 0), 2);
  frame.AddDataId(data_id2);
  const data_t data_id3(sensor_t(SensorType::CAMERA, 1), 1);
  frame.AddDataId(data_id3);
  EXPECT_THAT(
      std::vector<data_t>(frame.ImageIds().begin(), frame.ImageIds().end()),
      testing::UnorderedElementsAre(data_id2, data_id3));
}

TEST(Frame, SetResetPose) {
  Frame frame;
  EXPECT_FALSE(frame.HasPose());
  EXPECT_ANY_THROW(frame.RigFromWorld());
  EXPECT_EQ(frame.MaybeRigFromWorld(), std::nullopt);
  frame.SetRigFromWorld(Rigid3d());
  EXPECT_TRUE(frame.HasPose());
  EXPECT_EQ(frame.RigFromWorld(), Rigid3d());
  EXPECT_EQ(frame.MaybeRigFromWorld().value(), Rigid3d());
  frame.ResetPose();
  EXPECT_FALSE(frame.HasPose());
  EXPECT_ANY_THROW(frame.RigFromWorld());
  EXPECT_EQ(frame.MaybeRigFromWorld(), std::nullopt);
}

TEST(Frame, SetCamFromWorld) {
  Frame frame;
  Rig rig;
  frame.SetRigId(1);
  const sensor_t sensor_id1(SensorType::CAMERA, 0);
  rig.AddRefSensor(sensor_id1);
  const sensor_t sensor_id2(SensorType::CAMERA, 1);
  rig.AddSensor(sensor_id2, TestRigid3d());
  frame.SetRigPtr(&rig);

  const Rigid3d cam1_from_world = TestRigid3d();
  frame.SetCamFromWorld(sensor_id1.id, cam1_from_world);
  EXPECT_EQ(frame.RigFromWorld(), cam1_from_world);
  EXPECT_EQ(frame.SensorFromWorld(sensor_id1), cam1_from_world);

  const Rigid3d cam2_from_world = TestRigid3d();
  frame.SetCamFromWorld(sensor_id2.id, cam2_from_world);
  const Rigid3d sensor2_from_world = frame.SensorFromWorld(sensor_id2);
  EXPECT_THAT(cam2_from_world.translation,
              EigenMatrixNear(sensor2_from_world.translation, 1e-6));
  EXPECT_THAT(cam2_from_world.rotation.coeffs(),
              EigenMatrixNear(sensor2_from_world.rotation.coeffs(), 1e-6));
}

TEST(Image, Equals) {
  Frame frame;
  Frame other = frame;
  EXPECT_EQ(frame, other);
  frame.SetFrameId(2);
  EXPECT_NE(frame, other);
  other.SetFrameId(2);
  EXPECT_EQ(frame, other);
}

TEST(Frame, Print) {
  Frame frame;
  frame.SetFrameId(1);
  frame.SetRigId(2);
  frame.AddDataId(data_t(sensor_t(SensorType::IMU, 0), 2));
  frame.AddDataId(data_t(sensor_t(SensorType::CAMERA, 1), 3));
  std::ostringstream stream;
  stream << frame;
  EXPECT_EQ(stream.str(),
            "Frame(frame_id=1, rig_id=2, has_pose=0, "
            "data_ids=[(CAMERA, 1, 3), (IMU, 0, 2)])");
}

}  // namespace
}  // namespace colmap

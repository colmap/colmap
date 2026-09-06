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

#include "colmap/scene/imu.h"

#include <sstream>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(Imu, Default) {
  const Imu imu;
  EXPECT_EQ(imu.imu_id, kInvalidCameraId);
  EXPECT_EQ(imu.camera_id, kInvalidCameraId);
}

TEST(Imu, Print) {
  Imu imu;
  imu.imu_id = 1;
  imu.camera_id = 2;
  std::ostringstream stream;
  stream << imu;
  EXPECT_EQ(stream.str(), "Imu(imu_id=1, camera_id=2)");
}

TEST(ImuState, Default) {
  const ImuState state;
  EXPECT_EQ(state.params, (Eigen::Matrix<double, 9, 1>::Zero()));
  EXPECT_EQ(state.velocity(), Eigen::Vector3d::Zero());
  EXPECT_EQ(state.bias_gyro(), Eigen::Vector3d::Zero());
  EXPECT_EQ(state.bias_accel(), Eigen::Vector3d::Zero());
}

TEST(ImuState, Accessors) {
  ImuState state(Eigen::Vector3d(1, 2, 3),
                 Eigen::Vector3d(4, 5, 6),
                 Eigen::Vector3d(7, 8, 9));
  EXPECT_EQ(state.velocity(), Eigen::Vector3d(1, 2, 3));
  EXPECT_EQ(state.bias_gyro(), Eigen::Vector3d(4, 5, 6));
  EXPECT_EQ(state.bias_accel(), Eigen::Vector3d(7, 8, 9));
  // Params layout: [velocity(3), bias_gyro(3), bias_accel(3)].
  EXPECT_EQ(state.params.head<3>(), Eigen::Vector3d(1, 2, 3));
  EXPECT_EQ(state.params.segment<3>(3), Eigen::Vector3d(4, 5, 6));
  EXPECT_EQ(state.params.tail<3>(), Eigen::Vector3d(7, 8, 9));
  // Non-const accessors write back into params.
  state.velocity() = Eigen::Vector3d(10, 11, 12);
  EXPECT_EQ(state.params.head<3>(), Eigen::Vector3d(10, 11, 12));
}

TEST(ImuState, Print) {
  const ImuState state(Eigen::Vector3d(1, 2, 3),
                       Eigen::Vector3d(4, 5, 6),
                       Eigen::Vector3d(7, 8, 9));
  std::ostringstream stream;
  stream << state;
  EXPECT_EQ(stream.str(),
            "ImuState(vel=[1 2 3], bias_gyro=[4 5 6], bias_accel=[7 8 9])");
}

}  // namespace
}  // namespace colmap

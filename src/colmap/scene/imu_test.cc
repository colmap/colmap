// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/imu.h"

#include <sstream>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(Imu, Default) {
  const Imu imu;
  EXPECT_EQ(imu.imu_id, kInvalidImuId);
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

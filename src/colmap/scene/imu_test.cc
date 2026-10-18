// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/imu.h"

#include <sstream>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(Imu, Default) {
  const Imu imu;
  EXPECT_EQ(imu.imu_id, kInvalidImuId);
  EXPECT_EQ(imu.SensorId(), sensor_t(SensorType::IMU, kInvalidImuId));
}

TEST(Imu, Equals) {
  Imu imu1;
  imu1.imu_id = 1;
  Imu imu2 = imu1;
  EXPECT_EQ(imu1, imu2);
  imu2.imu_id = 2;
  EXPECT_NE(imu1, imu2);
}

TEST(Imu, Print) {
  Imu imu;
  imu.imu_id = 1;
  std::ostringstream stream;
  stream << imu;
  EXPECT_EQ(stream.str(), "Imu(imu_id=1)");
}

TEST(ImuState, Default) {
  const ImuState state;
  EXPECT_EQ(state.params, (Eigen::Matrix<double, 6, 1>::Zero()));
  EXPECT_EQ(state.bias_gyro(), Eigen::Vector3d::Zero());
  EXPECT_EQ(state.bias_accel(), Eigen::Vector3d::Zero());
}

TEST(ImuState, Accessors) {
  ImuState state(Eigen::Vector3d(1, 2, 3), Eigen::Vector3d(4, 5, 6));
  EXPECT_EQ(state.bias_gyro(), Eigen::Vector3d(1, 2, 3));
  EXPECT_EQ(state.bias_accel(), Eigen::Vector3d(4, 5, 6));
  // Params layout: [bias_gyro(3), bias_accel(3)].
  EXPECT_EQ(state.params.head<3>(), Eigen::Vector3d(1, 2, 3));
  EXPECT_EQ(state.params.tail<3>(), Eigen::Vector3d(4, 5, 6));
  // Non-const accessors write back into params.
  state.bias_gyro() = Eigen::Vector3d(7, 8, 9);
  EXPECT_EQ(state.params.head<3>(), Eigen::Vector3d(7, 8, 9));
}

TEST(ImuState, Equals) {
  ImuState state1(Eigen::Vector3d(1, 2, 3), Eigen::Vector3d(4, 5, 6));
  ImuState state2 = state1;
  EXPECT_EQ(state1, state2);
  state2.bias_accel() = Eigen::Vector3d(7, 8, 9);
  EXPECT_NE(state1, state2);
}

TEST(ImuState, Print) {
  const ImuState state(Eigen::Vector3d(1, 2, 3), Eigen::Vector3d(4, 5, 6));
  std::ostringstream stream;
  stream << state;
  EXPECT_EQ(stream.str(), "ImuState(bias_gyro=[1 2 3], bias_accel=[4 5 6])");
}

TEST(ImuCalibration, Equals) {
  ImuCalibration calib1;
  ImuCalibration calib2 = calib1;
  EXPECT_EQ(calib1, calib2);
  calib2.gravity_magnitude = 10.0;
  EXPECT_NE(calib1, calib2);
}

}  // namespace
}  // namespace colmap

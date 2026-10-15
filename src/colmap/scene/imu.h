// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/sensor/imu.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"
#include "colmap/util/types.h"

#include <ostream>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace colmap {

// An Imu class storing the sensor calibration and identifier.
struct Imu {
  ImuCalibration calib;
  imu_t imu_id = kInvalidImuId;

  inline sensor_t SensorId() const { return sensor_t(SensorType::IMU, imu_id); }

  bool operator==(const Imu& other) const {
    return calib == other.calib && imu_id == other.imu_id;
  }
  bool operator!=(const Imu& other) const { return !(*this == other); }
};

// IMU state for discrete-time optimization: biases.
// Parameters stored as [bias_gyro(3), bias_accel(3)].
// Design mirrors Rigid3d: public contiguous params, Eigen::Map accessors.
struct ImuState {
  Eigen::Matrix<double, 6, 1> params = Eigen::Matrix<double, 6, 1>::Zero();

  ImuState() = default;

  ImuState(const Eigen::Vector3d& bias_gyro,
           const Eigen::Vector3d& bias_accel) {
    params.head<3>() = bias_gyro;
    params.tail<3>() = bias_accel;
  }

  inline Eigen::Map<Eigen::Vector3d> bias_gyro() {
    return Eigen::Map<Eigen::Vector3d>(params.data());
  }
  inline Eigen::Map<const Eigen::Vector3d> bias_gyro() const {
    return Eigen::Map<const Eigen::Vector3d>(params.data());
  }

  inline Eigen::Map<Eigen::Vector3d> bias_accel() {
    return Eigen::Map<Eigen::Vector3d>(params.data() + 3);
  }
  inline Eigen::Map<const Eigen::Vector3d> bias_accel() const {
    return Eigen::Map<const Eigen::Vector3d>(params.data() + 3);
  }

  bool operator==(const ImuState& other) const {
    return params == other.params;
  }
  bool operator!=(const ImuState& other) const { return !(*this == other); }
};

std::ostream& operator<<(std::ostream& stream, const Imu& imu);

std::ostream& operator<<(std::ostream& stream, const ImuState& state);

}  // namespace colmap

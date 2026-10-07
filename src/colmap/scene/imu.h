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

// An Imu class storing the sensor information and a linked visual camera.
// TODO: Integrate with the Rig abstraction (sensor/rig.h) by making IMU a
// proper sensor in the rig and using sensor_from_rig transforms.
struct Imu {
  ImuCalibration calib;
  imu_t imu_id = kInvalidImuId;

  // Information for the associated visual camera. TODO: Use Rig instead.
  camera_t camera_id = kInvalidCameraId;  // The camera linked to IMU.
  Rigid3d imu_from_cam;
};

// IMU state for discrete-time optimization: velocity and biases.
// Parameters stored as [velocity(3), bias_gyro(3), bias_accel(3)].
// Design mirrors Rigid3d: public contiguous params, Eigen::Map accessors.
struct ImuState {
  Eigen::Matrix<double, 9, 1> params = Eigen::Matrix<double, 9, 1>::Zero();

  ImuState() = default;

  ImuState(const Eigen::Vector3d& velocity,
           const Eigen::Vector3d& bias_gyro,
           const Eigen::Vector3d& bias_accel) {
    params.head<3>() = velocity;
    params.segment<3>(3) = bias_gyro;
    params.tail<3>() = bias_accel;
  }

  inline Eigen::Map<Eigen::Vector3d> velocity() {
    return Eigen::Map<Eigen::Vector3d>(params.data());
  }
  inline Eigen::Map<const Eigen::Vector3d> velocity() const {
    return Eigen::Map<const Eigen::Vector3d>(params.data());
  }

  inline Eigen::Map<Eigen::Vector3d> bias_gyro() {
    return Eigen::Map<Eigen::Vector3d>(params.data() + 3);
  }
  inline Eigen::Map<const Eigen::Vector3d> bias_gyro() const {
    return Eigen::Map<const Eigen::Vector3d>(params.data() + 3);
  }

  inline Eigen::Map<Eigen::Vector3d> bias_accel() {
    return Eigen::Map<Eigen::Vector3d>(params.data() + 6);
  }
  inline Eigen::Map<const Eigen::Vector3d> bias_accel() const {
    return Eigen::Map<const Eigen::Vector3d>(params.data() + 6);
  }
};

std::ostream& operator<<(std::ostream& stream, const Imu& imu);

std::ostream& operator<<(std::ostream& stream, const ImuState& state);

}  // namespace colmap

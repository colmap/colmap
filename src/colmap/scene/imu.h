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
class Imu {
 public:
  Imu() = default;
  ~Imu() = default;
  ImuCalibration calib;
  camera_t imu_id = kInvalidCameraId;

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

  camera_t imu_id = kInvalidCameraId;
  image_t image_id = kInvalidImageId;
};

inline std::ostream& operator<<(std::ostream& stream, const Imu& imu) {
  stream << "Imu("
         << "imu_id=" << imu.imu_id << ", "
         << "camera_id=" << imu.camera_id << ")";
  return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const ImuState& state) {
  stream << "ImuState("
         << "vel=[" << state.velocity().transpose() << "], "
         << "bias_gyro=[" << state.bias_gyro().transpose() << "], "
         << "bias_accel=[" << state.bias_accel().transpose() << "])";
  return stream;
}

}  // namespace colmap

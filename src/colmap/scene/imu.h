// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace colmap {

// An Imu class storing the sensor information and a linked visual camera
// TODO: support rigs + unify sensors
class Imu {
 public:
  Imu() = default;
  ~Imu() = default;
  ImuCalibration calib;
  camera_t imu_id = kInvalidCameraId;

  // information for the associated visual camera. TODO: change to rigs
  camera_t camera_id = kInvalidCameraId;  // the camera linked to IMU.
  Rigid3d imu_from_cam;
};

// A state class storing speed and biases for discrete-time optimization
class ImuState {
 public:
  ImuState() = default;
  ~ImuState() = default;
  inline const Eigen::Matrix<double, 9, 1>& Data() const;
  inline Eigen::Matrix<double, 9, 1>& Data();

  inline void SetVelocity(const Eigen::Vector3d& vec);
  inline const Eigen::Vector3d Velocity() const;
  inline const double* VelocityPtr();

  inline void SetAccBias(const Eigen::Vector3d& vec);
  inline const Eigen::Vector3d AccBias() const;
  inline const double* AccBiasPtr();

  inline void SetGyroBias(const Eigen::Vector3d& vec);
  inline const Eigen::Vector3d GyroBias() const;
  inline const double* GyroBiasPtr();

  inline const camera_t& ImuId() const;
  inline camera_t ImuId();
  inline const image_t& ImageId() const;
  inline image_t ImageId();

 private:
  Eigen::Matrix<double, 9, 1> data_ =
      Eigen::Matrix<double, 9, 1>::Zero();  // 3-DoF speed + 6-DoF biases (acc +
                                            // gyro)
  camera_t imu_id_;   // the identifier of the associated IMU
  image_t image_id_;  // the corresponding image from visual input
};

const Eigen::Matrix<double, 9, 1>& ImuState::Data() const { return data_; }

Eigen::Matrix<double, 9, 1>& ImuState::Data() { return data_; }

void ImuState::SetVelocity(const Eigen::Vector3d& vec) {
  data_.head<3>() = vec;
}

const Eigen::Vector3d ImuState::Velocity() const {
  return Eigen::Vector3d(data_.data());
}

const double* ImuState::VelocityPtr() { return data_.data(); }

void ImuState::SetAccBias(const Eigen::Vector3d& vec) {
  data_.segment<3>(3) = vec;
}

const Eigen::Vector3d ImuState::AccBias() const {
  return Eigen::Vector3d(data_.data() + 3);
}

const double* ImuState::AccBiasPtr() { return data_.data() + 3; }

void ImuState::SetGyroBias(const Eigen::Vector3d& vec) {
  data_.tail<3>() = vec;
}

const Eigen::Vector3d ImuState::GyroBias() const {
  return Eigen::Vector3d(data_.data() + 6);
}

const double* ImuState::GyroBiasPtr() { return data_.data() + 6; }

const camera_t& ImuState::ImuId() const { return imu_id_; }

camera_t ImuState::ImuId() { return imu_id_; }

const image_t& ImuState::ImageId() const { return image_id_; }

image_t ImuState::ImageId() { return image_id_; }

}  // namespace colmap

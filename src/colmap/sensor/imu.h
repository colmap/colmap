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

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <algorithm>
#include <ostream>
#include <vector>

namespace colmap {

// References:
// [1] https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model-and-Intrinsics
// [2]
// https://github.com/uzh-rpg/rpg_svo_pro_open/blob/master/svo_common/include/svo/common/imu_calibration.h
// Default parameters are for ADIS16448 IMU.
struct ImuCalibration {
  /// Accelerometer noise density (sigma). [m/s^2*1/sqrt(Hz)]
  double acc_noise_density = 0.01883649;

  /// Gyro noise density (sigma). [rad/s*1/sqrt(Hz)]
  double gyro_noise_density = 0.00073088444;

  /// Accelerometer bias random walk (sigma). [m/s^3*1/sqrt(Hz)]
  double bias_accel_random_walk_sigma = 0.012589254;

  /// Gyro bias random walk (sigma). [rad/s^2*1/sqrt(Hz)]
  double bias_gyro_random_walk_sigma = 0.00038765;

  /// Accelerometer saturation. [m/s^2]
  double acc_saturation_max = 150;

  /// Gyroscope saturation. [rad/s]
  double gyro_saturation_max = 7.8;

  /// Norm of the Gravitational acceleration. [m/s^2]
  double gravity_magnitude = 9.81007;

  /// Expected IMU Rate [1/s]
  double imu_rate = 20.0;

  /// Rectification matrices correcting axis misalignment and scale.
  /// m_true = M^{-1}(m - b). Identity if data is already rectified.
  /// TODO: Support online calibration by making these optimizable
  /// as parameter blocks in the IMU cost function.
  Eigen::Matrix3d acc_rectification = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d gyro_rectification = Eigen::Matrix3d::Identity();
};

struct ImuMeasurement {
  timestamp_t timestamp = kInvalidTimestamp;  // [nanoseconds]
  Eigen::Vector3d accel = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();

  ImuMeasurement() {}
  ImuMeasurement(timestamp_t t,
                 const Eigen::Vector3d& accel,
                 const Eigen::Vector3d& gyro)
      : timestamp(t), accel(accel), gyro(gyro) {}
};

std::ostream& operator<<(std::ostream& stream,
                         const ImuCalibration& calibration);

std::ostream& operator<<(std::ostream& stream,
                         const ImuMeasurement& measurement);

// IMU measurements stored as a plain vector. Callers must ensure
// measurements are sorted by timestamp (chronological order).
using ImuMeasurements = std::vector<ImuMeasurement>;

// Extract measurements spanning the edge [t1, t2] from a sorted vector.
// Returns measurements from the sample just before t1 through the sample at
// or just after t2.
ImuMeasurements GetMeasurementsContainEdge(const ImuMeasurements& measurements,
                                           timestamp_t t1,
                                           timestamp_t t2);

}  // namespace colmap

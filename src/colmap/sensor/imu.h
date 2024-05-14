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

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <algorithm>

namespace colmap {

// References:
// [1] https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model-and-Intrinsics
// [2]
// https://github.com/uzh-rpg/rpg_svo_pro_open/blob/master/svo_common/include/svo/common/imu_calibration.h
// Default parameters are for ADIS16448 IMU.
class ImuCalibration {
 public:
  /// Accelerometer noise density (sigma). [m/s^2*1/sqrt(Hz)]
  double acc_noise_density = 0.01883649;

  /// Gyro noise density (sigma). [rad/s*1/sqrt(Hz)]
  double gyro_noise_density = 0.00073088444;

  /// Accelerometer bias random walk (sigma). [m/s^3*1/sqrt(Hz)]
  double acc_bias_random_walk_sigma = 0.012589254;

  /// Gyro bias random walk (sigma). [rad/s^2*1/sqrt(Hz)]
  double gyro_bias_random_walk_sigma = 0.00038765;

  /// Accelerometer saturation. [m/s^2]
  double acc_saturation_max = 150;

  /// Gyroscope saturation. [rad/s]
  double gyro_saturation_max = 7.8;

  /// Norm of the Gravitational acceleration. [m/s^2]
  double gravity_magnitude = 9.81007;

  /// Expected IMU Rate [1/s]
  double imu_rate = 20.0;

  ImuCalibration() = default;
  ~ImuCalibration() = default;
};

struct ImuMeasurement {
  double timestamp;
  Eigen::Vector3d linear_acceleration;
  Eigen::Vector3d angular_velocity;

  ImuMeasurement(const double t,
                 const Eigen::Vector3d& lin_acc,
                 const Eigen::Vector3d& ang_vel)
      : timestamp(t), linear_acceleration(lin_acc), angular_velocity(ang_vel) {}
  ~ImuMeasurement() = default;
};

// priority list for Imu measurements based on timestamps
class ImuMeasurements {
 public:
  ImuMeasurements() = default;
  ~ImuMeasurements() = default;
  ImuMeasurements(const std::vector<ImuMeasurement>& ms) { insert(ms); }
  ImuMeasurements(const ImuMeasurements& ms) { insert(ms); }
  void insert(const ImuMeasurement& m) {
    auto it = std::lower_bound(
        measurements_.begin(),
        measurements_.end(),
        m,
        [](const ImuMeasurement& m1, const ImuMeasurement& m2) {
          return m1.timestamp < m2.timestamp;
        });
    measurements_.insert(it, m);
  }
  void insert(const std::vector<ImuMeasurement>& ms) {
    for (auto it = ms.begin(); it != ms.end(); ++it) insert(*it);
  }
  void insert(const ImuMeasurements& ms) {
    for (auto it = ms.begin(); it != ms.end(); ++it) insert(*it);
  }
  void remove(const ImuMeasurement& m) {
    auto it = std::lower_bound(
        measurements_.begin(),
        measurements_.end(),
        m,
        [](const ImuMeasurement& m1, const ImuMeasurement& m2) {
          return m1.timestamp < m2.timestamp;
        });
    if (it != measurements_.end() && it->timestamp == m.timestamp)
      measurements_.erase(it);
    else
      throw std::invalid_argument("Element not found in the list");
  }
  void clear() { measurements_.clear(); }
  bool empty() const { return measurements_.empty(); }
  size_t size() const { return measurements_.size(); }
  const ImuMeasurement& operator[](size_t index) const {
    return measurements_[index];
  }
  typename std::vector<ImuMeasurement>::const_iterator begin() const {
    return measurements_.begin();
  }
  const ImuMeasurement& front() const { return measurements_.front(); }
  typename std::vector<ImuMeasurement>::const_iterator end() const {
    return measurements_.end();
  }
  const ImuMeasurement& back() const { return measurements_.back(); }
  const std::vector<ImuMeasurement>& Data() const { return measurements_; }
  ImuMeasurements GetMeasurementsContainEdge(const double t1, const double t2);

 private:
  std::vector<ImuMeasurement> measurements_;
};

}  // namespace colmap

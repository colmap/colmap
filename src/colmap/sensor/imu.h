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
  /// Gyro noise density (sigma). [rad/s*1/sqrt(Hz)]
  double gyro_noise_density = 0.00073088444;

  /// Accelerometer noise density (sigma). [m/s^2*1/sqrt(Hz)]
  double accel_noise_density = 0.01883649;

  /// Gyro bias random walk (sigma). [rad/s^2*1/sqrt(Hz)]
  double bias_gyro_random_walk_sigma = 0.00038765;

  /// Accelerometer bias random walk (sigma). [m/s^3*1/sqrt(Hz)]
  double bias_accel_random_walk_sigma = 0.012589254;

  /// Gyroscope saturation. [rad/s]
  double gyro_saturation_max = 7.8;

  /// Accelerometer saturation. [m/s^2]
  double accel_saturation_max = 150;

  /// Norm of the Gravitational acceleration. [m/s^2]
  double gravity_magnitude = 9.81007;

  /// Expected IMU Rate [1/s]
  double imu_rate = 20.0;

  /// Rectification matrices correcting axis misalignment and scale.
  /// m_true = M^{-1}(m - b). Identity if data is already rectified.
  /// TODO: Support online calibration by making these optimizable
  /// as parameter blocks in the IMU cost function.
  Eigen::Matrix3d gyro_rectification = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d accel_rectification = Eigen::Matrix3d::Identity();
};

struct ImuMeasurement {
  timestamp_t timestamp = kInvalidTimestamp;  // [nanoseconds]
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel = Eigen::Vector3d::Zero();

  ImuMeasurement() {}
  ImuMeasurement(timestamp_t t,
                 const Eigen::Vector3d& gyro,
                 const Eigen::Vector3d& accel)
      : timestamp(t), gyro(gyro), accel(accel) {}
};

std::ostream& operator<<(std::ostream& stream,
                         const ImuCalibration& calibration);

std::ostream& operator<<(std::ostream& stream,
                         const ImuMeasurement& measurement);

// Sorted list of IMU measurements ordered by timestamp.
class ImuMeasurements {
 public:
  ImuMeasurements() = default;
  explicit ImuMeasurements(const std::vector<ImuMeasurement>& ms) {
    Insert(ms);
  }

  void Insert(const ImuMeasurement& m) {
    // Fast path: append if empty or new measurement comes after all existing.
    if (Empty() || m.timestamp > measurements_.back().timestamp) {
      measurements_.push_back(m);
      return;
    }
    if (m.timestamp == measurements_.back().timestamp) {
      throw std::invalid_argument("Duplicate timestamp in ImuMeasurements: " +
                                  std::to_string(m.timestamp));
    }
    auto cmp = [](const ImuMeasurement& m1, const ImuMeasurement& m2) {
      return m1.timestamp < m2.timestamp;
    };
    auto it =
        std::lower_bound(measurements_.begin(), measurements_.end(), m, cmp);
    if (it != measurements_.end() && it->timestamp == m.timestamp) {
      throw std::invalid_argument("Duplicate timestamp in ImuMeasurements: " +
                                  std::to_string(m.timestamp));
    }
    measurements_.insert(it, m);
  }

  void Insert(const std::vector<ImuMeasurement>& ms) {
    if (Empty()) {
      measurements_ = ms;
      std::sort(measurements_.begin(),
                measurements_.end(),
                [](const ImuMeasurement& m1, const ImuMeasurement& m2) {
                  return m1.timestamp < m2.timestamp;
                });
      ThrowIfHasDuplicates(measurements_);
    } else {
      for (auto it = ms.begin(); it != ms.end(); ++it) Insert(*it);
    }
  }

  void Insert(const ImuMeasurements& ms) {
    if (Empty()) {
      measurements_ = ms.Data();
    } else {
      InsertSorted(ms.Data());
    }
  }

  // Insert measurements that are already sorted by timestamp.
  // If all new measurements come after the existing ones, this is O(m) append.
  // Otherwise falls back to O(n+m) merge. Throws on duplicate timestamps.
  void InsertSorted(const std::vector<ImuMeasurement>& sorted_ms) {
    if (sorted_ms.empty()) return;
    ThrowIfHasDuplicates(sorted_ms);
    if (Empty()) {
      measurements_ = sorted_ms;
      return;
    }
    if (sorted_ms.front().timestamp > measurements_.back().timestamp) {
      measurements_.insert(
          measurements_.end(), sorted_ms.begin(), sorted_ms.end());
      return;
    }
    if (sorted_ms.front().timestamp == measurements_.back().timestamp) {
      throw std::invalid_argument("Duplicate timestamp in ImuMeasurements: " +
                                  std::to_string(sorted_ms.front().timestamp));
    }
    std::vector<ImuMeasurement> merged;
    merged.reserve(measurements_.size() + sorted_ms.size());
    std::merge(measurements_.begin(),
               measurements_.end(),
               sorted_ms.begin(),
               sorted_ms.end(),
               std::back_inserter(merged),
               [](const ImuMeasurement& m1, const ImuMeasurement& m2) {
                 return m1.timestamp < m2.timestamp;
               });
    // Check for cross-range duplicates after merge.
    ThrowIfHasDuplicates(merged);
    measurements_ = std::move(merged);
  }

  void Remove(const ImuMeasurement& m) {
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

  void Clear() { measurements_.clear(); }
  bool Empty() const { return measurements_.empty(); }
  size_t Size() const { return measurements_.size(); }

  const ImuMeasurement& operator[](size_t index) const {
    return measurements_[index];
  }
  typename std::vector<ImuMeasurement>::const_iterator begin() const {
    return measurements_.begin();
  }
  typename std::vector<ImuMeasurement>::const_iterator end() const {
    return measurements_.end();
  }
  const ImuMeasurement& front() const { return measurements_.front(); }
  const ImuMeasurement& back() const { return measurements_.back(); }
  const std::vector<ImuMeasurement>& Data() const { return measurements_; }

  // Extract measurements that fully contain the edge [t1, t2]: from the
  // sample at or just before t1 to the sample at or just after t2. This
  // ensures the returned range brackets both endpoints, which is required
  // for correct IMU preintegration when t1/t2 fall between samples.
  ImuMeasurements ExtractMeasurementsContainEdge(timestamp_t t1,
                                                 timestamp_t t2) const;

 private:
  static void ThrowIfHasDuplicates(const std::vector<ImuMeasurement>& ms) {
    for (size_t i = 1; i < ms.size(); ++i) {
      if (ms[i].timestamp == ms[i - 1].timestamp) {
        throw std::invalid_argument("Duplicate timestamp in ImuMeasurements: " +
                                    std::to_string(ms[i].timestamp));
      }
    }
  }

  std::vector<ImuMeasurement> measurements_;
};

}  // namespace colmap

// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <ostream>
#include <vector>

namespace colmap {

// References:
// [1] https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model-and-Intrinsics
// [2]
// https://github.com/uzh-rpg/rpg_svo_pro_open/blob/master/svo_common/include/svo/common/imu_calibration.h
// Default parameters are for ADIS16448 IMU.
//
// The only modeled sensor error is an additive bias per axis, which is
// estimated during optimization. Scale and axis misalignment must be
// corrected before integration, and the noise parameters below then refer
// to the corrected signal.
struct ImuCalibration {
  // Gyro noise density (sigma). [rad/s*1/sqrt(Hz)]
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

  /// Expected IMU rate. [1/s]
  double imu_rate = 20.0;

  bool operator==(const ImuCalibration& other) const {
    return gyro_noise_density == other.gyro_noise_density &&
           accel_noise_density == other.accel_noise_density &&
           bias_gyro_random_walk_sigma == other.bias_gyro_random_walk_sigma &&
           bias_accel_random_walk_sigma == other.bias_accel_random_walk_sigma &&
           gyro_saturation_max == other.gyro_saturation_max &&
           accel_saturation_max == other.accel_saturation_max &&
           gravity_magnitude == other.gravity_magnitude &&
           imu_rate == other.imu_rate;
  }
  bool operator!=(const ImuCalibration& other) const {
    return !(*this == other);
  }
};

struct ImuMeasurement {
  timestamp_t timestamp = kInvalidTimestamp;  // [nanoseconds]
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel = Eigen::Vector3d::Zero();

  ImuMeasurement() {}
  ImuMeasurement(timestamp_t timestamp,
                 const Eigen::Vector3d& gyro,
                 const Eigen::Vector3d& accel)
      : timestamp(timestamp), gyro(gyro), accel(accel) {}
};

std::ostream& operator<<(std::ostream& stream,
                         const ImuCalibration& calibration);

std::ostream& operator<<(std::ostream& stream,
                         const ImuMeasurement& measurement);

// Sorted list of IMU measurements ordered by timestamp.
class ImuMeasurements {
 public:
  ImuMeasurements() = default;
  explicit ImuMeasurements(std::vector<ImuMeasurement> ms) {
    Insert(std::move(ms));
  }

  // Insert a single measurement, keeping the list sorted by timestamp.
  // Throws on a duplicate timestamp.
  // Note: Repeated out-of-order single insertions are O(N^2) due to vector
  // shifts; prefer batch-inserting multiple measurements via Insert(ms) or
  // InsertSorted(sorted_ms).
  void Insert(const ImuMeasurement& m);

  // Insert (unsorted) measurements, keeping the list sorted by timestamp.
  // Throws on a duplicate timestamp.
  void Insert(std::vector<ImuMeasurement> ms);

  // Merge in another sorted list. Throws on a duplicate timestamp.
  void Insert(ImuMeasurements ms);

  // Insert measurements that are already sorted by timestamp.
  // If all new measurements come after the existing ones, this is O(m) append.
  // Otherwise falls back to O(n+m) merge. Throws on unsorted or duplicate
  // timestamps.
  void InsertSorted(std::vector<ImuMeasurement> sorted_ms);

  // Remove the measurement with a matching timestamp. Throws if not found.
  void Remove(const ImuMeasurement& m);

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
  void ExtractMeasurementsInRange(timestamp_t t1,
                                  timestamp_t t2,
                                  ImuMeasurements* measurements) const;

 private:
  std::vector<ImuMeasurement> measurements_;
};

}  // namespace colmap

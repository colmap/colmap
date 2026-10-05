// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/sensor/imu.h"

#include "colmap/util/logging.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace colmap {
namespace {

void ThrowIfDuplicatesOrNotSorted(const std::vector<ImuMeasurement>& ms) {
  for (size_t i = 1; i < ms.size(); ++i) {
    if (ms[i].timestamp < ms[i - 1].timestamp) {
      throw std::invalid_argument(
          "ImuMeasurements are not sorted by timestamp: " +
          std::to_string(ms[i - 1].timestamp) + " > " +
          std::to_string(ms[i].timestamp));
    }
    if (ms[i].timestamp == ms[i - 1].timestamp) {
      throw std::invalid_argument("Duplicate timestamp in ImuMeasurements: " +
                                  std::to_string(ms[i].timestamp));
    }
  }
}

}  // namespace

void ImuMeasurements::Insert(const ImuMeasurement& m) {
  // Fast path: append if empty or new measurement comes after all existing.
  if (Empty() || m.timestamp > measurements_.back().timestamp) {
    measurements_.push_back(m);
    return;
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

void ImuMeasurements::Insert(std::vector<ImuMeasurement> ms) {
  std::sort(ms.begin(),
            ms.end(),
            [](const ImuMeasurement& m1, const ImuMeasurement& m2) {
              return m1.timestamp < m2.timestamp;
            });
  InsertSorted(std::move(ms));
}

void ImuMeasurements::Insert(ImuMeasurements ms) {
  if (Empty()) {
    measurements_ = std::move(ms.measurements_);
  } else {
    InsertSorted(std::move(ms.measurements_));
  }
}

void ImuMeasurements::InsertSorted(std::vector<ImuMeasurement> sorted_ms) {
  if (sorted_ms.empty()) return;
  ThrowIfDuplicatesOrNotSorted(sorted_ms);
  if (Empty()) {
    measurements_ = std::move(sorted_ms);
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
  ThrowIfDuplicatesOrNotSorted(merged);
  measurements_ = std::move(merged);
}

void ImuMeasurements::Remove(const ImuMeasurement& m) {
  auto it =
      std::lower_bound(measurements_.begin(),
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

void ImuMeasurements::ExtractMeasurementsInRange(
    timestamp_t t1, timestamp_t t2, ImuMeasurements* measurements) const {
  THROW_CHECK_NOTNULL(measurements);
  THROW_CHECK(!Empty()) << "Cannot query measurements from empty container.";
  THROW_CHECK_LT(t1, t2) << "t1 must be less than t2.";
  measurements->Clear();
  // The edge cannot be bracketed if it extends beyond the available samples
  // (e.g. the first/last image edge or across an IMU gap). Leave output empty
  // so callers can skip this edge rather than treating it as a fatal error.
  if (t1 < front().timestamp || t2 > back().timestamp) {
    return;
  }
  auto cmp = [](const ImuMeasurement& m1, const ImuMeasurement& m2) {
    return m1.timestamp < m2.timestamp;
  };
  ImuMeasurement range;
  range.timestamp = t1;
  auto it1 = std::upper_bound(begin(), end(), range, cmp);
  range.timestamp = t2;
  auto it2 = std::lower_bound(begin(), end(), range, cmp);
  // Range: sample at/before t1 through sample at/after t2.
  measurements->measurements_.assign(it1 - 1, it2 + 1);
}

std::ostream& operator<<(std::ostream& stream,
                         const ImuCalibration& calibration) {
  stream << "ImuCalibration("
         << "gyro_noise_density=" << calibration.gyro_noise_density << ", "
         << "accel_noise_density=" << calibration.accel_noise_density << ", "
         << "gravity_magnitude=" << calibration.gravity_magnitude << ")";
  return stream;
}

std::ostream& operator<<(std::ostream& stream,
                         const ImuMeasurement& measurement) {
  stream << "ImuMeasurement("
         << "t=" << measurement.timestamp << ", "
         << "gyro=[" << measurement.gyro.transpose() << "], "
         << "accel=[" << measurement.accel.transpose() << "])";
  return stream;
}

}  // namespace colmap

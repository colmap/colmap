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

#include "colmap/sensor/imu.h"

#include "colmap/util/logging.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace colmap {
namespace {

void ThrowIfHasDuplicates(const std::vector<ImuMeasurement>& ms) {
  for (size_t i = 1; i < ms.size(); ++i) {
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

void ImuMeasurements::Insert(const std::vector<ImuMeasurement>& ms) {
  std::vector<ImuMeasurement> sorted = ms;
  std::sort(sorted.begin(),
            sorted.end(),
            [](const ImuMeasurement& m1, const ImuMeasurement& m2) {
              return m1.timestamp < m2.timestamp;
            });
  InsertSorted(sorted);
}

void ImuMeasurements::Insert(const ImuMeasurements& ms) {
  if (Empty()) {
    measurements_ = ms.Data();
  } else {
    InsertSorted(ms.Data());
  }
}

void ImuMeasurements::InsertSorted(
    const std::vector<ImuMeasurement>& sorted_ms) {
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

ImuMeasurements ImuMeasurements::ExtractMeasurementsInRange(
    timestamp_t t1, timestamp_t t2) const {
  THROW_CHECK(!Empty()) << "Cannot query measurements from empty container.";
  THROW_CHECK_LT(t1, t2) << "t1 must be less than t2.";
  // The edge cannot be bracketed if it extends beyond the available samples
  // (e.g. the first/last image edge or across an IMU gap). Return empty so
  // callers can skip this edge rather than treating it as a fatal error.
  if (t1 < front().timestamp || t2 > back().timestamp) {
    return ImuMeasurements();
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
  ImuMeasurements result;
  result.InsertSorted(std::vector<ImuMeasurement>(it1 - 1, it2 + 1));
  return result;
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

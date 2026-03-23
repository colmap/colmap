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

namespace colmap {

ImuMeasurements GetMeasurementsContainEdge(const ImuMeasurements& measurements,
                                           timestamp_t t1,
                                           timestamp_t t2) {
  THROW_CHECK(!measurements.empty())
      << "Cannot query measurements from empty container.";
  THROW_CHECK_LT(t1, t2) << "t1 must be less than t2.";
  THROW_CHECK_GE(t1, measurements.front().timestamp)
      << "t1 is before the first measurement.";
  THROW_CHECK_LE(t2, measurements.back().timestamp)
      << "t2 is after the last measurement.";
  auto cmp = [](const ImuMeasurement& m1, const ImuMeasurement& m2) {
    return m1.timestamp < m2.timestamp;
  };
  ImuMeasurement dummy;
  dummy.timestamp = t1;
  auto it1 =
      std::upper_bound(measurements.begin(), measurements.end(), dummy, cmp);
  dummy.timestamp = t2;
  auto it2 =
      std::lower_bound(measurements.begin(), measurements.end(), dummy, cmp);
  // Include the sample just before t1 through the sample at/after t2.
  return ImuMeasurements(it1 - 1, it2 + 1);
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
         << "accel=[" << measurement.accel.transpose() << "], "
         << "gyro=[" << measurement.gyro.transpose() << "])";
  return stream;
}

}  // namespace colmap

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

#include "colmap/sensor/imu.h"

namespace colmap {

ImuMeasurements ImuMeasurements::GetMeasurementsContainEdge(double t1,
                                                            double t2) {
  ImuMeasurements res;
  if (t1 >= t2 || t1 < front().timestamp || t2 > back().timestamp) return res;
  auto it1 = std::upper_bound(
      measurements_.begin(),
      measurements_.end(),
      ImuMeasurement(t1, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()),
      [](const ImuMeasurement& m1, const ImuMeasurement& m2) {
        return m1.timestamp < m2.timestamp;
      });
  auto it2 = std::lower_bound(
      measurements_.begin(),
      measurements_.end(),
      ImuMeasurement(t2, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()),
      [](const ImuMeasurement& m1, const ImuMeasurement& m2) {
        return m1.timestamp < m2.timestamp;
      });
  for (auto it = it1 - 1; it != it2; ++it) {
    res.insert(*it);
  }
  res.insert(*it2);
  return res;
}

}  // namespace colmap

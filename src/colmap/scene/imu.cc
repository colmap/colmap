// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/imu.h"

namespace colmap {

std::ostream& operator<<(std::ostream& stream, const Imu& imu) {
  stream << "Imu(imu_id=" << imu.imu_id << ")";
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const ImuState& state) {
  stream << "ImuState("
         << "bias_gyro=[" << state.bias_gyro().transpose() << "], "
         << "bias_accel=[" << state.bias_accel().transpose() << "])";
  return stream;
}

}  // namespace colmap

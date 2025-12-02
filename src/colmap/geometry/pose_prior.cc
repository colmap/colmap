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

#include "colmap/geometry/pose_prior.h"

#include "colmap/util/logging.h"

namespace colmap {
namespace {

// Handle NaNs explicitly and consider them equal, whereas the default C++
// comparison operator returns false for a NaN == NaN comparison.
template <typename T>
bool IsNaNEqual(const T& left, const T& right) {
  THROW_CHECK_EQ(left.rows(), right.rows());
  THROW_CHECK_EQ(left.cols(), right.cols());
  for (int i = 0; i < left.rows(); ++i) {
    for (int j = 0; j < left.cols(); ++j) {
      if ((std::isnan(left(i, j)) != std::isnan(right(i, j)) ||
           (!std::isnan(left(i, j)) && left(i, j) != right(i, j)))) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

bool PosePrior::operator==(const PosePrior& other) const {
  return pose_prior_id == other.pose_prior_id &&
         corr_data_id == other.corr_data_id &&
         coordinate_system == other.coordinate_system &&
         IsNaNEqual(position, other.position) &&
         IsNaNEqual(position_covariance, other.position_covariance) &&
         IsNaNEqual(gravity, other.gravity) &&
         IsNaNEqual(gravity_covariance, other.gravity_covariance);
}

bool PosePrior::operator!=(const PosePrior& other) const {
  return !(*this == other);
}

std::ostream& operator<<(std::ostream& stream, const PosePrior& prior) {
  const static Eigen::IOFormat kVecFmt(
      Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ");
  stream << "PosePrior(pose_prior_id=" << prior.pose_prior_id
         << ", corr_data_id=(" << prior.corr_data_id.sensor_id.type << ", "
         << prior.corr_data_id.sensor_id.id << ", " << prior.corr_data_id.id
         << "), position=[" << prior.position.format(kVecFmt)
         << "], position_covariance=["
         << prior.position_covariance.format(kVecFmt) << "], coordinate_system="
         << PosePrior::CoordinateSystemToString(prior.coordinate_system)
         << ", gravity=[" << prior.gravity.format(kVecFmt)
         << "], gravity_covariance=["
         << prior.gravity_covariance.format(kVecFmt) << "])";
  return stream;
}

}  // namespace colmap

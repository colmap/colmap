// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/geometry/pose_prior.h"

#include "colmap/math/math.h"
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
         IsNaNEqual(gravity, other.gravity);
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
         << ", gravity=[" << prior.gravity.format(kVecFmt) << "])";
  return stream;
}

std::optional<Eigen::Vector3d> GravityFromExifOrientation(int orientation) {
  switch (orientation) {
    case 1:  // Normal
      return Eigen::Vector3d(0, 1, 0);
    case 3:  // Rotate 180
      return Eigen::Vector3d(0, -1, 0);
    case 6:  // Rotate 90 CW
      return Eigen::Vector3d(1, 0, 0);
    case 8:  // Rotate 270 CW
      return Eigen::Vector3d(-1, 0, 0);
    case 2:
    case 4:
    case 5:
    case 7:
      LOG(WARNING) << "Unsupported EXIF orientation: " << orientation;
      return std::nullopt;
    default:
      LOG(ERROR) << "Unknown EXIF orientation: " << orientation;
      return std::nullopt;
  }
}

int ComputeRot90FromGravity(const Eigen::Vector3d& gravity) {
  // Calculate the angle of gravity in image space, then find number of 90 deg
  // CCW rotations needed to make the image upright (where gravity is at pi/2).
  const double angle = std::atan2(gravity.y(), gravity.x());
  constexpr double kHalfPi = M_PI / 2.0;
  int rot90_ccw = static_cast<int>(std::round((angle - kHalfPi) / kHalfPi)) % 4;
  if (rot90_ccw < 0) {
    rot90_ccw += 4;
  }
  return rot90_ccw;
}

}  // namespace colmap

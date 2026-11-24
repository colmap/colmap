// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cmath>

#include <Eigen/Geometry>

namespace colmap {

inline constexpr double kDefaultHeadingStddev = 0.05235987755982989;  // 3 deg
inline constexpr double kMinHeadingProjection = 0.01;

// Heading is clockwise from north (+Y) in East-North-Up coordinates.
// The residual is a scaled chord between the observed and predicted azimuths,
// so there is no branch cut at +/-pi. Smoothly downweight an almost vertical
// forward axis, for which azimuth is undefined.
class HeadingErrorCostFunctor {
 public:
  HeadingErrorCostFunctor(
      double heading_rad,
      double stddev_rad,
      const Eigen::Vector3d& forward_in_rig = Eigen::Vector3d::UnitZ())
      : east_(std::sin(heading_rad)),
        north_(std::cos(heading_rad)),
        inv_stddev_(1.0 / stddev_rad),
        forward_in_rig_(forward_in_rig) {}

  template <typename T>
  bool operator()(const T* const rig_from_world_rotation_xyzw,
                  T* residuals) const {
    using std::sqrt;
    const Eigen::Map<const Eigen::Quaternion<T>> rig_from_world(
        rig_from_world_rotation_xyzw);
    const Eigen::Matrix<T, 3, 1> forward_in_world =
        rig_from_world.conjugate() * forward_in_rig_.cast<T>();
    const T x = forward_in_world.x();
    const T y = forward_in_world.y();
    const T r2 = x * x + y * y;
    const T r = sqrt(r2 + T(1e-16));  // Finite derivative at the pole.
    const T weight =
        r / (r2 + T(kMinHeadingProjection * kMinHeadingProjection));
    residuals[0] = T(inv_stddev_) * weight * (x - r * T(east_));
    residuals[1] = T(inv_stddev_) * weight * (y - r * T(north_));
    return true;
  }

 private:
  const double east_;
  const double north_;
  const double inv_stddev_;
  const Eigen::Vector3d forward_in_rig_;
};

}  // namespace colmap

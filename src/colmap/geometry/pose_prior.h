// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/enum_utils.h"
#include "colmap/util/types.h"

#include <optional>
#include <ostream>

#include <Eigen/Core>

namespace colmap {

struct PosePrior {
 public:
  MAKE_ENUM_CLASS(CoordinateSystem,
                  -1,
                  UNDEFINED,  // = -1
                  WGS84,      // = 0
                  CARTESIAN   // = 1
  );

  constexpr static double kNaN = std::numeric_limits<double>::quiet_NaN();

  // The unique identifier of the pose prior.
  pose_prior_t pose_prior_id = kInvalidPosePriorId;

  // The identifier of the associated sensor for which this prior defines the
  // pose. For example, this can refer to a camera or an IMU sensor.
  data_t corr_data_id = kInvalidDataId;

  // The position of the associated sensor in the world coordinate system.
  Eigen::Vector3d position = Eigen::Vector3d::Constant(kNaN);
  // The position covariance in the Cartesian world coordinate system.
  Eigen::Matrix3d position_covariance = Eigen::Matrix3d::Constant(kNaN);
  // The coordinate system of the position in the world.
  CoordinateSystem coordinate_system = CoordinateSystem::UNDEFINED;

  // The gravity (down) in the sensor coordinate system.
  Eigen::Vector3d gravity = Eigen::Vector3d::Constant(kNaN);

  inline bool HasPosition() const { return position.allFinite(); }
  inline bool HasPositionCov() const { return position_covariance.allFinite(); }
  inline bool HasGravity() const { return gravity.allFinite(); }

  bool operator==(const PosePrior& other) const;
  bool operator!=(const PosePrior& other) const;
};

std::ostream& operator<<(std::ostream& stream, const PosePrior& prior);

// Extract gravity vector from EXIF orientation. Returns std::nullopt if not an
// upright orientation (e.g. mirrored).
std::optional<Eigen::Vector3d> GravityFromExifOrientation(int orientation);

// Returns the number of 90 deg counter-clockwise rotations needed to make the
// sensor upright.
int ComputeRot90FromGravity(const Eigen::Vector3d& gravity);

}  // namespace colmap

// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/scene/track.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <Eigen/Core>

namespace colmap {

// 3D point class that holds information about triangulated 2D points.
struct Point3D {
  // The 3D position of the point.
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();

  // The mean reprojection error in pixels.
  double error = -1.;

  // The track of the point as a list of image observations.
  Track track;

  // The color of the point in the range [0, 255].
  Eigen::Vector3ub color = Eigen::Vector3ub::Zero();

  inline bool HasError() const;

  inline bool operator==(const Point3D& other) const;
  inline bool operator!=(const Point3D& other) const;
};

std::ostream& operator<<(std::ostream& stream, const Point3D& point3D);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

bool Point3D::HasError() const { return error != -1; }

bool Point3D::operator==(const Point3D& other) const {
  return xyz == other.xyz && color == other.color && error == other.error &&
         track == other.track;
}

bool Point3D::operator!=(const Point3D& other) const {
  return !(*this == other);
}

}  // namespace colmap

// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_BASE_POINT3D_H_
#define COLMAP_SRC_BASE_POINT3D_H_

#include <vector>

#include <Eigen/Core>

#include "base/track.h"
#include "util/logging.h"
#include "util/types.h"

namespace colmap {

// 3D point class that holds information about triangulated 2D points.
class Point3D {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Point3D();

  // The point coordinate in world space.
  inline const Eigen::Vector3d& XYZ() const;
  inline Eigen::Vector3d& XYZ();
  inline double XYZ(const size_t idx) const;
  inline double& XYZ(const size_t idx);
  inline double X() const;
  inline double Y() const;
  inline double Z() const;
  inline void SetXYZ(const Eigen::Vector3d& xyz);

  // The RGB color of the point.
  inline const Eigen::Vector3ub& Color() const;
  inline Eigen::Vector3ub& Color();
  inline uint8_t Color(const size_t idx) const;
  inline uint8_t& Color(const size_t idx);
  inline void SetColor(const Eigen::Vector3ub& color);

  // The mean reprojection error in image space.
  inline double Error() const;
  inline bool HasError() const;
  inline void SetError(const double error);

  inline const class Track& Track() const;
  inline class Track& Track();
  inline void SetTrack(const class Track& track);

 private:
  // The 3D position of the point.
  Eigen::Vector3d xyz_;

  // The color of the point in the range [0, 255].
  Eigen::Vector3ub color_;

  // The mean reprojection error in pixels.
  double error_;

  // The track of the point as a list of image observations.
  class Track track_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

const Eigen::Vector3d& Point3D::XYZ() const { return xyz_; }

Eigen::Vector3d& Point3D::XYZ() { return xyz_; }

double Point3D::XYZ(const size_t idx) const { return xyz_(idx); }

double& Point3D::XYZ(const size_t idx) { return xyz_(idx); }

double Point3D::X() const { return xyz_.x(); }

double Point3D::Y() const { return xyz_.y(); }

double Point3D::Z() const { return xyz_.z(); }

void Point3D::SetXYZ(const Eigen::Vector3d& xyz) { xyz_ = xyz; }

const Eigen::Vector3ub& Point3D::Color() const { return color_; }

Eigen::Vector3ub& Point3D::Color() { return color_; }

uint8_t Point3D::Color(const size_t idx) const { return color_(idx); }

uint8_t& Point3D::Color(const size_t idx) { return color_(idx); }

void Point3D::SetColor(const Eigen::Vector3ub& color) { color_ = color; }

double Point3D::Error() const { return error_; }

bool Point3D::HasError() const { return error_ != -1.0; }

void Point3D::SetError(const double error) { error_ = error; }

const class Track& Point3D::Track() const { return track_; }

class Track& Point3D::Track() {
  return track_;
}

void Point3D::SetTrack(const class Track& track) { track_ = track; }

}  // namespace colmap

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(colmap::Point3D)

#endif  // COLMAP_SRC_BASE_POINT3D_H_

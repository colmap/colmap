// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

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

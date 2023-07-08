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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#pragma once

#include "colmap/util/types.h"

#include <Eigen/Core>

namespace colmap {

// 2D point class corresponds to a feature in an image. It may or may not have a
// corresponding 3D point if it is part of a triangulated track.
class Point2D {
 public:
  Point2D();

  // The coordinate in image space in pixels.
  inline const Eigen::Vector2d& XY() const;
  inline Eigen::Vector2d& XY();
  inline double X() const;
  inline double Y() const;
  inline void SetXY(const Eigen::Vector2d& xy);

  // The identifier of the observed 3D point. If the image point does not
  // observe a 3D point, the identifier is `kInvalidPoint3Did`.
  inline point3D_t Point3DId() const;
  inline bool HasPoint3D() const;
  inline void SetPoint3DId(point3D_t point3D_id);

 private:
  // The image coordinates in pixels, starting at upper left corner with 0.
  Eigen::Vector2d xy_;

  // The identifier of the 3D point. If the 2D point is not part of a 3D point
  // track the identifier is `kInvalidPoint3DId` and `HasPoint3D() = false`.
  point3D_t point3D_id_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

const Eigen::Vector2d& Point2D::XY() const { return xy_; }

Eigen::Vector2d& Point2D::XY() { return xy_; }

double Point2D::X() const { return xy_.x(); }

double Point2D::Y() const { return xy_.y(); }

void Point2D::SetXY(const Eigen::Vector2d& xy) { xy_ = xy; }

point3D_t Point2D::Point3DId() const { return point3D_id_; }

bool Point2D::HasPoint3D() const { return point3D_id_ != kInvalidPoint3DId; }

void Point2D::SetPoint3DId(const point3D_t point3D_id) {
  point3D_id_ = point3D_id;
}

}  // namespace colmap

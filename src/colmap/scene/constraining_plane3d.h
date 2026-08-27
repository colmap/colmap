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

#pragma once

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <ostream>
#include <vector>

#include <Eigen/Core>

namespace colmap {

// Plane that a labeled set of 3D points is constrained to lie on during bundle
// adjustment. Stored in Hesse normal form, `normal.dot(xyz) + offset == 0` with
// a unit normal, so that `SignedDistance` is a Euclidean distance. The three
// degrees of freedom are split into a normal on the unit sphere and a scalar
// offset, which is exactly the dimension of a plane and avoids the scale
// degeneracy of a homogeneous 4-vector.
struct ConstrainingPlane3D {
  Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
  double offset = 0.0;

  // Hold the plane constant during bundle adjustment. Use for planes that are
  // already known in the reconstruction frame, e.g. derived from CCT markers.
  bool is_fixed = false;

  // Direction the normal is pulled towards. Keeps the plane identifiable when
  // its labeled points are nearly collinear, which would otherwise leave one
  // rotational degree of freedom of the normal unconstrained. Disabled when the
  // sigma is non-positive.
  Eigen::Vector3d prior_normal = Eigen::Vector3d::Zero();
  double prior_normal_sigma_deg = 0.0;

  ConstrainingPlane3D() = default;
  ConstrainingPlane3D(const Eigen::Vector3d& normal, double offset);

  inline bool HasNormalPrior() const;

  // Signed Euclidean distance of a point from the plane, positive on the side
  // the normal points to. Only exact while the normal has unit length.
  inline double SignedDistance(const Eigen::Vector3d& xyz) const;

  // Rescale into unit normal form. No-op for a degenerate normal.
  void Normalize();

  inline bool operator==(const ConstrainingPlane3D& other) const;
  inline bool operator!=(const ConstrainingPlane3D& other) const;
};

// Conditioning of a plane fitted to a point set. The eigenvalues are those of
// the point scatter matrix in ascending order, so `eigenvalues(0)` measures the
// off-plane spread and `eigenvalues(1)` the narrower in-plane spread.
struct ConstrainingPlaneFit {
  bool success = false;
  Eigen::Vector3d eigenvalues = Eigen::Vector3d::Zero();
  double rms_distance = 0.0;

  // Ratio of the two in-plane extents. Approaches zero for a collinear point
  // set, for which the fitted normal is not identifiable.
  double InPlaneExtentRatio() const;
};

// Least-squares plane through the given points, oriented so that the normal
// points towards `view_direction_hint` when that hint is non-zero. Fails for
// fewer than three points or a rank-deficient scatter matrix.
ConstrainingPlaneFit FitConstrainingPlane3D(
    const std::vector<Eigen::Vector3d>& points,
    ConstrainingPlane3D* plane,
    const Eigen::Vector3d& view_direction_hint = Eigen::Vector3d::Zero());

std::ostream& operator<<(std::ostream& stream,
                         const ConstrainingPlane3D& plane);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

bool ConstrainingPlane3D::HasNormalPrior() const {
  return prior_normal_sigma_deg > 0.0 && prior_normal.squaredNorm() > 0.0;
}

double ConstrainingPlane3D::SignedDistance(const Eigen::Vector3d& xyz) const {
  return normal.dot(xyz) + offset;
}

bool ConstrainingPlane3D::operator==(const ConstrainingPlane3D& other) const {
  return normal == other.normal && offset == other.offset &&
         is_fixed == other.is_fixed && prior_normal == other.prior_normal &&
         prior_normal_sigma_deg == other.prior_normal_sigma_deg;
}

bool ConstrainingPlane3D::operator!=(const ConstrainingPlane3D& other) const {
  return !(*this == other);
}

}  // namespace colmap

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

#include "colmap/scene/constraining_plane3d.h"

#include <cmath>

#include <Eigen/Eigenvalues>

namespace colmap {

ConstrainingPlane3D::ConstrainingPlane3D(const Eigen::Vector3d& normal,
                                        const double offset)
    : normal(normal), offset(offset) {
  Normalize();
}

void ConstrainingPlane3D::Normalize() {
  const double norm = normal.norm();
  if (norm <= 0.0 || !std::isfinite(norm)) {
    return;
  }
  normal /= norm;
  offset /= norm;
}

double ConstrainingPlaneFit::InPlaneExtentRatio() const {
  if (eigenvalues(2) <= 0.0) {
    return 0.0;
  }
  return std::sqrt(eigenvalues(1) / eigenvalues(2));
}

ConstrainingPlaneFit FitConstrainingPlane3D(
    const std::vector<Eigen::Vector3d>& points,
    ConstrainingPlane3D* plane,
    const Eigen::Vector3d& view_direction_hint) {
  ConstrainingPlaneFit fit;
  if (points.size() < 3 || plane == nullptr) {
    return fit;
  }

  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const Eigen::Vector3d& point : points) {
    centroid += point;
  }
  centroid /= static_cast<double>(points.size());

  Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
  for (const Eigen::Vector3d& point : points) {
    const Eigen::Vector3d centered = point - centroid;
    scatter += centered * centered.transpose();
  }
  scatter /= static_cast<double>(points.size());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(scatter);
  if (solver.info() != Eigen::Success) {
    return fit;
  }

  // Eigenvalues are returned in ascending order, so the first eigenvector is
  // the direction of least spread, i.e. the plane normal.
  fit.eigenvalues = solver.eigenvalues();
  Eigen::Vector3d normal = solver.eigenvectors().col(0);
  const double normal_norm = normal.norm();
  if (normal_norm <= 0.0 || !std::isfinite(normal_norm) ||
      fit.eigenvalues(2) <= 0.0) {
    return fit;
  }
  normal /= normal_norm;

  if (view_direction_hint.squaredNorm() > 0.0 &&
      normal.dot(view_direction_hint) < 0.0) {
    normal = -normal;
  }

  plane->normal = normal;
  plane->offset = -normal.dot(centroid);

  double sum_squared = 0.0;
  for (const Eigen::Vector3d& point : points) {
    const double distance = plane->SignedDistance(point);
    sum_squared += distance * distance;
  }
  fit.rms_distance = std::sqrt(sum_squared / static_cast<double>(points.size()));
  fit.success = true;
  return fit;
}

std::ostream& operator<<(std::ostream& stream,
                         const ConstrainingPlane3D& plane) {
  stream << "ConstrainingPlane3D(normal=[" << plane.normal(0) << ", "
         << plane.normal(1) << ", " << plane.normal(2) << "], offset="
         << plane.offset << ", is_fixed=" << (plane.is_fixed ? "true" : "false")
         << ")";
  return stream;
}

}  // namespace colmap

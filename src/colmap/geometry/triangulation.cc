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

#include "colmap/geometry/triangulation.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <Eigen/Dense>

namespace colmap {

bool TriangulatePoint(const Eigen::Matrix3x4d& cam1_from_world,
                      const Eigen::Matrix3x4d& cam2_from_world,
                      const Eigen::Vector2d& cam_point1,
                      const Eigen::Vector2d& cam_point2,
                      Eigen::Vector3d* xyz) {
  THROW_CHECK_NOTNULL(xyz);

  Eigen::Matrix4d A;
  A.row(0) = cam_point1(0) * cam1_from_world.row(2) - cam1_from_world.row(0);
  A.row(1) = cam_point1(1) * cam1_from_world.row(2) - cam1_from_world.row(1);
  A.row(2) = cam_point2(0) * cam2_from_world.row(2) - cam2_from_world.row(0);
  A.row(3) = cam_point2(1) * cam2_from_world.row(2) - cam2_from_world.row(1);

  const Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
  if (svd.info() != Eigen::Success) {
    return false;
  }
#endif

  if (svd.matrixV()(3, 3) == 0) {
    return false;
  }

  *xyz = svd.matrixV().col(3).hnormalized();
  return true;
}

bool TriangulateMidPoint(const Rigid3d& cam2_from_cam1,
                         const Eigen::Vector3d& cam_ray1,
                         const Eigen::Vector3d& cam_ray2,
                         Eigen::Vector3d* point3D_in_cam1) {
  const Eigen::Quaterniond cam1_from_cam2_rotation =
      cam2_from_cam1.rotation.inverse();
  const Eigen::Vector3d cam_ray2_in_cam1 = cam1_from_cam2_rotation * cam_ray2;
  const Eigen::Vector3d cam2_in_cam1 =
      cam1_from_cam2_rotation * -cam2_from_cam1.translation;

  Eigen::Matrix3d A;
  A << cam_ray1(0), -cam_ray2_in_cam1(0), -cam2_in_cam1(0), cam_ray1(1),
      -cam_ray2_in_cam1(1), -cam2_in_cam1(1), cam_ray1(2), -cam_ray2_in_cam1(2),
      -cam2_in_cam1(2);

  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullV);
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
  if (svd.info() != Eigen::Success) {
    return false;
  }
#endif

  if (svd.matrixV()(2, 2) == 0) {
    return false;
  }

  const Eigen::Vector2d lambda = svd.matrixV().col(2).hnormalized();

  // Check if point is behind cameras.
  if (lambda(0) <= std::numeric_limits<double>::epsilon() ||
      lambda(1) <= std::numeric_limits<double>::epsilon()) {
    return false;
  }

  *point3D_in_cam1 = 0.5 * (lambda(0) * cam_ray1 + cam2_in_cam1 +
                            lambda(1) * cam_ray2_in_cam1);

  return true;
}

bool TriangulatePointBearing(const Rigid3d& cam_from_world1,
                             const Rigid3d& cam_from_world2,
                             const Eigen::Vector3d& bearing1,
                             const Eigen::Vector3d& bearing2,
                             Eigen::Vector3d* point3D) {
  THROW_CHECK_NOTNULL(point3D);

  // Transform bearing vectors to world frame
  const Rigid3d world_from_cam1 = Inverse(cam_from_world1);
  const Rigid3d world_from_cam2 = Inverse(cam_from_world2);

  const Eigen::Vector3d ray1_world = world_from_cam1.rotation * bearing1;
  const Eigen::Vector3d ray2_world = world_from_cam2.rotation * bearing2;

  // Camera centers in world frame
  const Eigen::Vector3d center1 = world_from_cam1.translation;
  const Eigen::Vector3d center2 = world_from_cam2.translation;

  // Find closest points on the two rays using the parametric form:
  // P1 = center1 + t1 * ray1_world
  // P2 = center2 + t2 * ray2_world
  //
  // We solve for t1, t2 that minimize ||P1 - P2||^2
  // This gives us: (center2 - center1) = t1 * ray1_world - t2 * ray2_world
  //
  // Using dot products with ray1 and ray2:
  // d = center2 - center1
  // a = ray1 . ray1 = 1 (unit vectors)
  // b = ray1 . ray2
  // c = ray2 . ray2 = 1 (unit vectors)
  // d1 = d . ray1
  // d2 = d . ray2
  //
  // t1 = (d1 - b*d2) / (1 - b^2)
  // t2 = (b*d1 - d2) / (1 - b^2)

  const Eigen::Vector3d d = center2 - center1;
  const double b = ray1_world.dot(ray2_world);
  const double d1 = d.dot(ray1_world);
  const double d2 = d.dot(ray2_world);

  const double denom = 1.0 - b * b;

  // Check if rays are nearly parallel (degenerate case)
  constexpr double kEpsilon = 1e-10;
  if (std::abs(denom) < kEpsilon) {
    return false;
  }

  const double t1 = (d1 - b * d2) / denom;
  const double t2 = (b * d1 - d2) / denom;

  // For spherical cameras, both positive and negative t values are valid
  // (the bearing vector points toward the scene, not necessarily away from
  // the camera center). However, we require consistent signs - both rays
  // should agree on which side of the baseline the point is.
  // Allow small negative values due to numerical precision.
  constexpr double kMinLambda = -1e-3;
  if (t1 < kMinLambda || t2 < kMinLambda) {
    // Point is behind both cameras - still valid for spherical cameras
    // if both t values are consistently negative
    if (t1 > -kMinLambda || t2 > -kMinLambda) {
      // Inconsistent signs - rays don't truly intersect
      return false;
    }
  }

  // Compute midpoint of the two closest points
  const Eigen::Vector3d P1 = center1 + t1 * ray1_world;
  const Eigen::Vector3d P2 = center2 + t2 * ray2_world;
  *point3D = 0.5 * (P1 + P2);

  return true;
}

bool TriangulateMultiViewPoint(
    const span<const Eigen::Matrix3x4d>& cams_from_world,
    const span<const Eigen::Vector2d>& cam_points,
    Eigen::Vector3d* xyz) {
  THROW_CHECK_EQ(cams_from_world.size(), cam_points.size());
  THROW_CHECK_NOTNULL(xyz);

  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
  for (size_t i = 0; i < cam_points.size(); i++) {
    const Eigen::Vector3d point = cam_points[i].homogeneous().normalized();
    const Eigen::Matrix3x4d term =
        cams_from_world[i] - point * point.transpose() * cams_from_world[i];
    A += term.transpose() * term;
  }

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(A);
  if (eigen_solver.info() != Eigen::Success ||
      eigen_solver.eigenvectors()(3, 0) == 0) {
    return false;
  }

  *xyz = eigen_solver.eigenvectors().col(0).hnormalized();
  return true;
}

bool TriangulateOptimalPoint(const Eigen::Matrix3x4d& cam1_from_world_mat,
                             const Eigen::Matrix3x4d& cam2_from_world_mat,
                             const Eigen::Vector2d& cam_point1,
                             const Eigen::Vector2d& cam_point2,
                             Eigen::Vector3d* xyz) {
  const Rigid3d cam1_from_world(
      Eigen::Quaterniond(cam1_from_world_mat.leftCols<3>()),
      cam1_from_world_mat.col(3));
  const Rigid3d cam2_from_world(
      Eigen::Quaterniond(cam2_from_world_mat.leftCols<3>()),
      cam2_from_world_mat.col(3));
  const Rigid3d cam2_from_cam1 = cam2_from_world * Inverse(cam1_from_world);
  const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);

  Eigen::Vector2d optimal_point1;
  Eigen::Vector2d optimal_point2;
  FindOptimalImageObservations(
      E, cam_point1, cam_point2, &optimal_point1, &optimal_point2);

  return TriangulatePoint(cam1_from_world_mat,
                          cam2_from_world_mat,
                          optimal_point1,
                          optimal_point2,
                          xyz);
}

double CalculateTriangulationAngle(const Eigen::Vector3d& proj_center1,
                                   const Eigen::Vector3d& proj_center2,
                                   const Eigen::Vector3d& point3D) {
  const double angle = CalculateAngleBetweenVectors(point3D - proj_center1,
                                                    point3D - proj_center2);
  // Triangulation is unstable for acute angles (far away points) and
  // obtuse angles (close points), so always compute the minimum angle
  // between the two intersecting rays.
  return std::min(angle, static_cast<double>(EIGEN_PI) - angle);
}

std::vector<double> CalculateTriangulationAngles(
    const Eigen::Vector3d& proj_center1,
    const Eigen::Vector3d& proj_center2,
    const std::vector<Eigen::Vector3d>& points3D) {
  std::vector<double> angles(points3D.size());
  for (size_t i = 0; i < points3D.size(); ++i) {
    angles[i] =
        CalculateTriangulationAngle(proj_center1, proj_center2, points3D[i]);
  }
  return angles;
}

double CalculateAngleBetweenVectors(const Eigen::Vector3d& v1,
                                    const Eigen::Vector3d& v2) {
  const double squared_norm1 = v1.squaredNorm();
  const double squared_norm2 = v2.squaredNorm();
  if (squared_norm1 == 0.0 || squared_norm2 == 0.0) {
    return 0.0;
  }
  return std::acos(std::clamp(
      v1.dot(v2) / std::sqrt(squared_norm1 * squared_norm2), -1.0, 1.0));
}

}  // namespace colmap

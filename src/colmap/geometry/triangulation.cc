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

bool TriangulatePoint(const Eigen::Matrix3x4d& cam1_from_world,
                      const Eigen::Matrix3x4d& cam2_from_world,
                      const Eigen::Vector3d& cam_ray1,
                      const Eigen::Vector3d& cam_ray2,
                      Eigen::Vector3d* xyz) {
  THROW_CHECK_NOTNULL(xyz);

  Eigen::Matrix<double, 6, 4> A;
  A.topRows<3>() =
      cam1_from_world - cam_ray1 * (cam_ray1.transpose() * cam1_from_world);
  A.bottomRows<3>() =
      cam2_from_world - cam_ray2 * (cam_ray2.transpose() * cam2_from_world);

  const Eigen::JacobiSVD<Eigen::Matrix<double, 6, 4>> svd(A,
                                                          Eigen::ComputeFullV);
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
      cam2_from_cam1.rotation().inverse();
  const Eigen::Vector3d cam_ray2_in_cam1 = cam1_from_cam2_rotation * cam_ray2;
  const Eigen::Vector3d cam2_in_cam1 =
      cam1_from_cam2_rotation * -cam2_from_cam1.translation();

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

namespace {

// Contribution of a single bearing observation to the projector-based DLT
// normal-equation matrix. With unit bearing b and projection matrix
// P = cam_from_world, the residual operator is term = P - b b^T P, and the
// system accumulates term^T term.
inline Eigen::Matrix4d TriangulationDltTerm(
    const Eigen::Matrix3x4d& cam_from_world, const Eigen::Vector3d& cam_ray) {
  const Eigen::Matrix3x4d term =
      cam_from_world - cam_ray * cam_ray.transpose() * cam_from_world;
  return term.transpose() * term;
}

// Solve the DLT system for the homogeneous point (smallest eigenvector of A).
inline bool SolveTriangulationDlt(const Eigen::Matrix4d& A,
                                  Eigen::Vector3d* xyz) {
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(A);
  if (eigen_solver.info() != Eigen::Success ||
      eigen_solver.eigenvectors()(3, 0) == 0) {
    return false;
  }
  *xyz = eigen_solver.eigenvectors().col(0).hnormalized();
  return true;
}

}  // namespace

bool TriangulateMultiViewPoint(
    const span<const Eigen::Matrix3x4d>& cams_from_world,
    const span<const Eigen::Vector2d>& cam_points,
    Eigen::Vector3d* xyz) {
  THROW_CHECK_EQ(cams_from_world.size(), cam_points.size());
  THROW_CHECK_NOTNULL(xyz);
  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
  for (size_t i = 0; i < cam_points.size(); ++i) {
    A += TriangulationDltTerm(cams_from_world[i],
                              cam_points[i].homogeneous().normalized());
  }
  return SolveTriangulationDlt(A, xyz);
}

bool TriangulateMultiViewPoint(
    const span<const Eigen::Matrix3x4d>& cams_from_world,
    const span<const Eigen::Vector3d>& cam_rays,
    Eigen::Vector3d* xyz) {
  THROW_CHECK_EQ(cams_from_world.size(), cam_rays.size());
  THROW_CHECK_NOTNULL(xyz);
  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
  for (size_t i = 0; i < cam_rays.size(); ++i) {
    A += TriangulationDltTerm(cams_from_world[i], cam_rays[i]);
  }
  return SolveTriangulationDlt(A, xyz);
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

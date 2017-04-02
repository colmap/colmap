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

#include "base/triangulation.h"

#include "base/essential_matrix.h"
#include "base/pose.h"

namespace colmap {

Eigen::Vector3d TriangulatePoint(const Eigen::Matrix3x4d& proj_matrix1,
                                 const Eigen::Matrix3x4d& proj_matrix2,
                                 const Eigen::Vector2d& point1,
                                 const Eigen::Vector2d& point2) {
  Eigen::Matrix4d A;

  A.row(0) = point1(0) * proj_matrix1.row(2) - proj_matrix1.row(0);
  A.row(1) = point1(1) * proj_matrix1.row(2) - proj_matrix1.row(1);
  A.row(2) = point2(0) * proj_matrix2.row(2) - proj_matrix2.row(0);
  A.row(3) = point2(1) * proj_matrix2.row(2) - proj_matrix2.row(1);

  Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);

  return svd.matrixV().col(3).hnormalized();
}

std::vector<Eigen::Vector3d> TriangulatePoints(
    const Eigen::Matrix3x4d& proj_matrix1,
    const Eigen::Matrix3x4d& proj_matrix2,
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2) {
  CHECK_EQ(points1.size(), points2.size());

  std::vector<Eigen::Vector3d> points3D(points1.size());

  for (size_t i = 0; i < points3D.size(); ++i) {
    points3D[i] =
        TriangulatePoint(proj_matrix1, proj_matrix2, points1[i], points2[i]);
  }

  return points3D;
}

Eigen::Vector3d TriangulateMultiViewPoint(
    const std::vector<Eigen::Matrix3x4d>& proj_matrices,
    const std::vector<Eigen::Vector2d>& points) {
  CHECK_EQ(proj_matrices.size(), points.size());

  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();

  for (size_t i = 0; i < points.size(); i++) {
    const Eigen::Vector3d point = points[i].homogeneous().normalized();
    const Eigen::Matrix3x4d term =
        proj_matrices[i] - point * point.transpose() * proj_matrices[i];
    A += term.transpose() * term;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(A);

  return eigen_solver.eigenvectors().col(0).hnormalized();
}

Eigen::Vector3d TriangulateOptimalPoint(const Eigen::Matrix3x4d& proj_matrix1,
                                        const Eigen::Matrix3x4d& proj_matrix2,
                                        const Eigen::Vector2d& point1,
                                        const Eigen::Vector2d& point2) {
  const Eigen::Matrix3d E =
      EssentialMatrixFromAbsolutePoses(proj_matrix1, proj_matrix2);

  Eigen::Vector2d optimal_point1;
  Eigen::Vector2d optimal_point2;
  FindOptimalImageObservations(E, point1, point2, &optimal_point1,
                               &optimal_point2);

  return TriangulatePoint(proj_matrix1, proj_matrix2, optimal_point1,
                          optimal_point2);
}

std::vector<Eigen::Vector3d> TriangulateOptimalPoints(
    const Eigen::Matrix3x4d& proj_matrix1,
    const Eigen::Matrix3x4d& proj_matrix2,
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2) {
  std::vector<Eigen::Vector3d> points3D(points1.size());

  for (size_t i = 0; i < points3D.size(); ++i) {
    points3D[i] =
        TriangulatePoint(proj_matrix1, proj_matrix2, points1[i], points2[i]);
  }

  return points3D;
}

double CalculateTriangulationAngle(const Eigen::Vector3d& proj_center1,
                                   const Eigen::Vector3d& proj_center2,
                                   const Eigen::Vector3d& point3D) {
  const double baseline2 = (proj_center1 - proj_center2).squaredNorm();

  const double ray1 = (point3D - proj_center1).norm();
  const double ray2 = (point3D - proj_center2).norm();

  // Angle between rays at point within the enclosing triangle,
  // see "law of cosines".
  const double angle = std::abs(
      std::acos((ray1 * ray1 + ray2 * ray2 - baseline2) / (2 * ray1 * ray2)));

  if (IsNaN(angle)) {
    return 0;
  } else {
    // Triangulation is unstable for acute angles (far away points) and
    // obtuse angles (close points), so always compute the minimum angle
    // between the two intersecting rays.
    return std::min(angle, M_PI - angle);
  }
}

std::vector<double> CalculateTriangulationAngles(
    const Eigen::Matrix3x4d& proj_matrix1,
    const Eigen::Matrix3x4d& proj_matrix2,
    const std::vector<Eigen::Vector3d>& points3D) {
  const Eigen::Vector3d& proj_center1 =
      ProjectionCenterFromMatrix(proj_matrix1);
  const Eigen::Vector3d& proj_center2 =
      ProjectionCenterFromMatrix(proj_matrix2);

  // Baseline length between cameras.
  const double baseline2 = (proj_center1 - proj_center2).squaredNorm();

  std::vector<double> angles(points3D.size());

  for (size_t i = 0; i < points3D.size(); ++i) {
    const Eigen::Vector3d& point3D = points3D[i];

    // Ray lengths from cameras to point.
    const double ray1 = (point3D - proj_center1).norm();
    const double ray2 = (point3D - proj_center2).norm();

    // Angle between rays at point within the enclosing triangle,
    // see "law of cosines".
    const double angle = std::abs(
        std::acos((ray1 * ray1 + ray2 * ray2 - baseline2) / (2 * ray1 * ray2)));

    if (IsNaN(angle)) {
      angles[i] = 0;
    } else {
      // Triangulation is unstable for acute angles (far away points) and
      // obtuse angles (close points), so always compute the minimum angle
      // between the two intersecting rays.
      angles[i] = std::min(angle, M_PI - angle);
    }
  }

  return angles;
}

}  // namespace colmap

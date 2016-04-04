// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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

#include "estimators/p3p.h"

#include <Eigen/Geometry>

#include "util/logging.h"
#include "util/math.h"
#include "util/types.h"

namespace colmap {
namespace {

Eigen::Vector3d LiftImagePoint(const Eigen::Vector2d& point) {
  return point.homogeneous() / std::sqrt(point.squaredNorm() + 1);
}

}  // namespace

std::vector<P3PEstimator::M_t> P3PEstimator::Estimate(
    const std::vector<X_t>& points2D, const std::vector<Y_t>& points3D) {
  CHECK_EQ(points2D.size(), 3);
  CHECK_EQ(points3D.size(), 3);

  Eigen::Matrix3d points3D_world;
  points3D_world.col(0) = points3D[0];
  points3D_world.col(1) = points3D[1];
  points3D_world.col(2) = points3D[2];

  const Eigen::Vector3d u = LiftImagePoint(points2D[0]);
  const Eigen::Vector3d v = LiftImagePoint(points2D[1]);
  const Eigen::Vector3d w = LiftImagePoint(points2D[2]);

  // Angles between 2D points.
  const double cos_uv = u.transpose() * v;
  const double cos_uw = u.transpose() * w;
  const double cos_vw = v.transpose() * w;

  // Distances between 2D points.
  const double dist_AB_2 = (points3D[0] - points3D[1]).squaredNorm();
  const double dist_AC_2 = (points3D[0] - points3D[2]).squaredNorm();
  const double dist_BC_2 = (points3D[1] - points3D[2]).squaredNorm();

  const double dist_AB = std::sqrt(dist_AB_2);

  const double a = dist_BC_2 / dist_AB_2;
  const double b = dist_AC_2 / dist_AB_2;

  // Helper variables for calculation of coefficients.
  const double a2 = a * a;
  const double b2 = b * b;
  const double p = 2 * cos_vw;
  const double q = 2 * cos_uw;
  const double r = 2 * cos_uv;
  const double p2 = p * p;
  const double p3 = p2 * p;
  const double q2 = q * q;
  const double r2 = r * r;
  const double r3 = r2 * r;
  const double r4 = r3 * r;
  const double r5 = r4 * r;

  // Build polynomial coefficients: a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0 = 0.
  std::vector<double> coeffs_a(5);
  coeffs_a[4] = -2 * b + b2 + a2 + 1 + a * b * (2 - r2) - 2 * a;  // a4
  coeffs_a[3] = -2 * q * a2 - r * p * b2 + 4 * q * a + (2 * q + p * r) * b +
                (r2 * q - 2 * q + r * p) * a * b - 2 * q;  // a3
  coeffs_a[2] = (2 + q2) * a2 + (p2 + r2 - 2) * b2 - (4 + 2 * q2) * a -
                (p * q * r + p2) * b - (p * q * r + r2) * a * b + q2 + 2;  // a2
  coeffs_a[1] = -2 * q * a2 - r * p * b2 + 4 * q * a +
                (p * r + q * p2 - 2 * q) * b + (r * p + 2 * q) * a * b -
                2 * q;                                           // a1
  coeffs_a[0] = a2 + b2 - 2 * a + (2 - p2) * b - 2 * a * b + 1;  // a0

  const std::vector<std::complex<double>> roots_a = SolvePolynomialN(coeffs_a);

  std::vector<M_t> models;

  const double kEps = 1e-10;

  for (const auto root_a : roots_a) {
    const double x = root_a.real();

    // Neglect all complex results as degenerate cases.
    if (root_a.imag() > kEps || x < 0) {
      continue;
    }

    const double x2 = x * x;
    const double x3 = x2 * x;

    // Build polynomial coefficients: b1*y + b0 = 0.
    const double _b1 =
        (p2 - p * q * r + r2) * a + (p2 - r2) * b - p2 + p * q * r - r2;
    const double b1 = b * _b1 * _b1;
    const double b0 =
        ((1 - a - b) * x2 + (a - 1) * q * x - a + b + 1) *
        (r3 * (a2 + b2 - 2 * a - 2 * b + (2 - r2) * a * b + 1) * x3 +
         r2 * (p + p * a2 - 2 * r * q * a * b + 2 * r * q * b - 2 * r * q -
               2 * p * a - 2 * p * b + p * r2 * b + 4 * r * q * a +
               q * r3 * a * b - 2 * r * q * a2 + 2 * p * a * b + p * b2 -
               r2 * p * b2) *
             x2 +
         (r5 * (b2 - a * b) - r4 * p * q * b +
          r3 * (q2 - 4 * a - 2 * q2 * a + q2 * a2 + 2 * a2 - 2 * b2 + 2) +
          r2 * (4 * p * q * a - 2 * p * q * a * b + 2 * p * q * b - 2 * p * q -
                2 * p * q * a2) +
          r * (p2 * b2 - 2 * p2 * b + 2 * p2 * a * b - 2 * p2 * a + p2 +
               p2 * a2)) *
             x +
         (2 * p * r2 - 2 * r3 * q + p3 - 2 * p2 * q * r + p * q2 * r2) * a2 +
         (p3 - 2 * p * r2) * b2 +
         (4 * q * r3 - 4 * p * r2 - 2 * p3 + 4 * p2 * q * r - 2 * p * q2 * r2) *
             a +
         (-2 * q * r3 + p * r4 + 2 * p2 * q * r - 2 * p3) * b +
         (2 * p3 + 2 * q * r3 - 2 * p2 * q * r) * a * b + p * q2 * r2 -
         2 * p2 * q * r + 2 * p * r2 + p3 - 2 * r3 * q);

    // Solve for y.
    const double y = b0 / b1;
    const double y2 = y * y;

    const double nu = x2 + y2 - 2 * x * y * cos_uv;

    const double dist_PC = dist_AB / std::sqrt(nu);
    const double dist_PB = y * dist_PC;
    const double dist_PA = x * dist_PC;

    Eigen::Matrix3d points3D_camera;
    points3D_camera.col(0) = u * dist_PA;  // A'
    points3D_camera.col(1) = v * dist_PB;  // B'
    points3D_camera.col(2) = w * dist_PC;  // C'

    // Find transformation from world to camera system (similarity transform
    // without scale - Euclidean transform).
    const Eigen::Matrix4d matrix =
        Eigen::umeyama(points3D_world, points3D_camera, false);

    models.push_back(matrix.topLeftCorner<3, 4>());
  }

  return models;
}

void P3PEstimator::Residuals(const std::vector<X_t>& points2D,
                             const std::vector<Y_t>& points3D,
                             const M_t& proj_matrix,
                             std::vector<double>* residuals) {
  CHECK_EQ(points2D.size(), points3D.size());

  residuals->resize(points2D.size());

  // Note that this code might not be as nice as Eigen expressions,
  // but it is significantly faster in various tests.

  const double P_00 = proj_matrix(0, 0);
  const double P_01 = proj_matrix(0, 1);
  const double P_02 = proj_matrix(0, 2);
  const double P_03 = proj_matrix(0, 3);
  const double P_10 = proj_matrix(1, 0);
  const double P_11 = proj_matrix(1, 1);
  const double P_12 = proj_matrix(1, 2);
  const double P_13 = proj_matrix(1, 3);
  const double P_20 = proj_matrix(2, 0);
  const double P_21 = proj_matrix(2, 1);
  const double P_22 = proj_matrix(2, 2);
  const double P_23 = proj_matrix(2, 3);

  for (size_t i = 0; i < points2D.size(); ++i) {
    const double x_0 = points2D[i](0);
    const double x_1 = points2D[i](1);

    const double X_0 = points3D[i](0);
    const double X_1 = points3D[i](1);
    const double X_2 = points3D[i](2);

    // Project 3D point from world to camera.
    const double px_2 = P_20 * X_0 + P_21 * X_1 + P_22 * X_2 + P_23;

    // Check if 3D point infront of camera.
    if (px_2 > std::numeric_limits<double>::epsilon()) {
      const double px_0 = P_00 * X_0 + P_01 * X_1 + P_02 * X_2 + P_03;
      const double px_1 = P_10 * X_0 + P_11 * X_1 + P_12 * X_2 + P_13;

      const double inv_px_2 = 1.0 / px_2;
      const double dx_0 = x_0 - px_0 * inv_px_2;
      const double dx_1 = x_1 - px_1 * inv_px_2;

      (*residuals)[i] = dx_0 * dx_0 + dx_1 * dx_1;
    } else {
      (*residuals)[i] = std::numeric_limits<double>::max();
    }
  }
}

}  // namespace colmap

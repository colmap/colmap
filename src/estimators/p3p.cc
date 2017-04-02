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

#include "estimators/p3p.h"

#include "base/polynomial.h"
#include "estimators/utils.h"
#include "util/logging.h"

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
  Eigen::Matrix<double, 5, 1> coeffs;
  coeffs(0) = -2 * b + b2 + a2 + 1 + a * b * (2 - r2) - 2 * a;
  coeffs(1) = -2 * q * a2 - r * p * b2 + 4 * q * a + (2 * q + p * r) * b +
              (r2 * q - 2 * q + r * p) * a * b - 2 * q;
  coeffs(2) = (2 + q2) * a2 + (p2 + r2 - 2) * b2 - (4 + 2 * q2) * a -
              (p * q * r + p2) * b - (p * q * r + r2) * a * b + q2 + 2;
  coeffs(3) = -2 * q * a2 - r * p * b2 + 4 * q * a +
              (p * r + q * p2 - 2 * q) * b + (r * p + 2 * q) * a * b - 2 * q;
  coeffs(4) = a2 + b2 - 2 * a + (2 - p2) * b - 2 * a * b + 1;

  Eigen::VectorXd roots_real;
  Eigen::VectorXd roots_imag;
  if (!FindPolynomialRootsCompanionMatrix(coeffs, &roots_real, &roots_imag)) {
    return std::vector<P3PEstimator::M_t>({});
  }

  std::vector<M_t> models;
  models.reserve(roots_real.size());

  for (Eigen::VectorXd::Index i = 0; i < roots_real.size(); ++i) {
    const double kMaxRootImag = 1e-10;
    if (std::abs(roots_imag(i)) > kMaxRootImag) {
      continue;
    }

    const double x = roots_real(i);
    if (x < 0) {
      continue;
    }

    const double x2 = x * x;
    const double x3 = x2 * x;

    // Build polynomial coefficients: b1*y + b0 = 0.
    const double bb1 =
        (p2 - p * q * r + r2) * a + (p2 - r2) * b - p2 + p * q * r - r2;
    const double b1 = b * bb1 * bb1;
    const double b0 =
        ((1 - a - b) * x2 + (a - 1) * q * x - a + b + 1) *
        (r3 * (a2 + b2 - 2 * a - 2 * b + (2 - r2) * a * b + 1) * x3 +
         r2 *
             (p + p * a2 - 2 * r * q * a * b + 2 * r * q * b - 2 * r * q -
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

    // Find transformation from the world to the camera system.
    const Eigen::Matrix4d transform =
        Eigen::umeyama(points3D_world, points3D_camera, false);
    models.push_back(transform.topLeftCorner<3, 4>());
  }

  return models;
}

void P3PEstimator::Residuals(const std::vector<X_t>& points2D,
                             const std::vector<Y_t>& points3D,
                             const M_t& proj_matrix,
                             std::vector<double>* residuals) {
  ComputeSquaredReprojectionError(points2D, points3D, proj_matrix, residuals);
}

}  // namespace colmap

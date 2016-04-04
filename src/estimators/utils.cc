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

#include "estimators/utils.h"

#include "util/logging.h"

namespace colmap {

void CenterAndNormalizeImagePoints(const std::vector<Eigen::Vector2d>& points,
                                   std::vector<Eigen::Vector2d>* normed_points,
                                   Eigen::Matrix3d* matrix) {
  // Calculate centroid
  Eigen::Vector2d centroid(0, 0);
  for (const auto point : points) {
    centroid += point;
  }
  centroid /= points.size();

  // Root mean square error to centroid of all points
  double rms_mean_dist = 0;
  for (const auto point : points) {
    rms_mean_dist += (point - centroid).squaredNorm();
  }
  rms_mean_dist = std::sqrt(rms_mean_dist / points.size());

  // Compose normalization matrix
  const double norm_factor = std::sqrt(2.0) / rms_mean_dist;
  *matrix << norm_factor, 0, -norm_factor * centroid(0), 0, norm_factor,
      -norm_factor * centroid(1), 0, 0, 1;

  // Apply normalization matrix
  normed_points->resize(points.size());

  const double M_00 = (*matrix)(0, 0);
  const double M_01 = (*matrix)(0, 1);
  const double M_02 = (*matrix)(0, 2);
  const double M_10 = (*matrix)(1, 0);
  const double M_11 = (*matrix)(1, 1);
  const double M_12 = (*matrix)(1, 2);
  const double M_20 = (*matrix)(2, 0);
  const double M_21 = (*matrix)(2, 1);
  const double M_22 = (*matrix)(2, 2);

  for (size_t i = 0; i < points.size(); ++i) {
    const double p_0 = points[i](0);
    const double p_1 = points[i](1);

    const double np_0 = M_00 * p_0 + M_01 * p_1 + M_02;
    const double np_1 = M_10 * p_0 + M_11 * p_1 + M_12;
    const double np_2 = M_20 * p_0 + M_21 * p_1 + M_22;

    const double inv_np_2 = 1.0 / np_2;
    (*normed_points)[i](0) = np_0 * inv_np_2;
    (*normed_points)[i](1) = np_1 * inv_np_2;
  }
}

void ComputeSquaredSampsonError(const std::vector<Eigen::Vector2d>& points1,
                                const std::vector<Eigen::Vector2d>& points2,
                                const Eigen::Matrix3d& E,
                                std::vector<double>* residuals) {
  CHECK_EQ(points1.size(), points2.size());

  residuals->resize(points1.size());

  // Note that this code might not be as nice as Eigen expressions,
  // but it is significantly faster in various tests

  const double E_00 = E(0, 0);
  const double E_01 = E(0, 1);
  const double E_02 = E(0, 2);
  const double E_10 = E(1, 0);
  const double E_11 = E(1, 1);
  const double E_12 = E(1, 2);
  const double E_20 = E(2, 0);
  const double E_21 = E(2, 1);
  const double E_22 = E(2, 2);

  for (size_t i = 0; i < points1.size(); ++i) {
    const double x1_0 = points1[i](0);
    const double x1_1 = points1[i](1);
    const double x2_0 = points2[i](0);
    const double x2_1 = points2[i](1);

    // Ex1 = E * points1[i].homogeneous();
    const double Ex1_0 = E_00 * x1_0 + E_01 * x1_1 + E_02;
    const double Ex1_1 = E_10 * x1_0 + E_11 * x1_1 + E_12;
    const double Ex1_2 = E_20 * x1_0 + E_21 * x1_1 + E_22;

    // Etx2 = E.transpose() * points2[i].homogeneous();
    const double Etx2_0 = E_00 * x2_0 + E_10 * x2_1 + E_20;
    const double Etx2_1 = E_01 * x2_0 + E_11 * x2_1 + E_21;

    // x2tEx1 = points2[i].homogeneous().transpose() * Ex1;
    const double x2tEx1 = x2_0 * Ex1_0 + x2_1 * Ex1_1 + Ex1_2;

    // Sampson distance
    (*residuals)[i] = x2tEx1 * x2tEx1 / (Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1 +
                                         Etx2_0 * Etx2_0 + Etx2_1 * Etx2_1);
  }
}
}

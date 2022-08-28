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

#include "estimators/utils.h"

#include "util/logging.h"

namespace colmap {

void CenterAndNormalizeImagePoints(const std::vector<Eigen::Vector2d>& points,
                                   std::vector<Eigen::Vector2d>* normed_points,
                                   Eigen::Matrix3d* matrix) {
  // Calculate centroid
  Eigen::Vector2d centroid(0, 0);
  for (const Eigen::Vector2d& point : points) {
    centroid += point;
  }
  centroid /= points.size();

  // Root mean square error to centroid of all points
  double rms_mean_dist = 0;
  for (const Eigen::Vector2d& point : points) {
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
    (*residuals)[i] =
        x2tEx1 * x2tEx1 /
        (Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1 + Etx2_0 * Etx2_0 + Etx2_1 * Etx2_1);
  }
}

void ComputeSquaredReprojectionError(
    const std::vector<Eigen::Vector2d>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const Eigen::Matrix3x4d& proj_matrix, std::vector<double>* residuals) {
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
    const double X_0 = points3D[i](0);
    const double X_1 = points3D[i](1);
    const double X_2 = points3D[i](2);

    // Project 3D point from world to camera.
    const double px_2 = P_20 * X_0 + P_21 * X_1 + P_22 * X_2 + P_23;

    // Check if 3D point is in front of camera.
    if (px_2 > std::numeric_limits<double>::epsilon()) {
      const double px_0 = P_00 * X_0 + P_01 * X_1 + P_02 * X_2 + P_03;
      const double px_1 = P_10 * X_0 + P_11 * X_1 + P_12 * X_2 + P_13;

      const double x_0 = points2D[i](0);
      const double x_1 = points2D[i](1);

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

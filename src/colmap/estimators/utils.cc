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

#include "colmap/estimators/utils.h"

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <Eigen/Geometry>

namespace colmap {

void CenterAndNormalizeImagePoints(const std::vector<Eigen::Vector2d>& points,
                                   std::vector<Eigen::Vector2d>* normed_points,
                                   Eigen::Matrix3d* normed_from_orig) {
  const size_t num_points = points.size();
  THROW_CHECK_GT(num_points, 0);

  // Calculate centroid.
  Eigen::Vector2d centroid(0, 0);
  for (const Eigen::Vector2d& point : points) {
    centroid += point;
  }
  centroid /= num_points;

  // Root mean square distance to centroid of all points.
  double rms_mean_dist = 0;
  for (const Eigen::Vector2d& point : points) {
    rms_mean_dist += (point - centroid).squaredNorm();
  }
  rms_mean_dist = std::sqrt(rms_mean_dist / num_points);

  // Compose normalization matrix.
  const double norm_factor = std::sqrt(2.0) / rms_mean_dist;
  *normed_from_orig << norm_factor, 0, -norm_factor * centroid(0), 0,
      norm_factor, -norm_factor * centroid(1), 0, 0, 1;

  // Apply normalization matrix.
  normed_points->resize(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    (*normed_points)[i] =
        (*normed_from_orig * points[i].homogeneous()).hnormalized();
  }
}

namespace {

inline double ComputeSquaredSampsonError(const Eigen::Vector3d& ray1,
                                         const Eigen::Vector3d& ray2,
                                         const Eigen::Matrix3d& E) {
  const Eigen::Vector3d epipolar_line1 = E * ray1;
  const double num = ray2.dot(epipolar_line1);
  const Eigen::Vector4d denom(ray2.dot(E.col(0)),
                              ray2.dot(E.col(1)),
                              epipolar_line1.x(),
                              epipolar_line1.y());
  const double denom_sq_norm = denom.squaredNorm();
  if (denom_sq_norm == 0) {
    return std::numeric_limits<double>::max();
  }
  return num * num / denom_sq_norm;
}

}  // namespace

void ComputeSquaredSampsonError(const std::vector<Eigen::Vector2d>& points1,
                                const std::vector<Eigen::Vector2d>& points2,
                                const Eigen::Matrix3d& E,
                                std::vector<double>* residuals) {
  const size_t num_points1 = points1.size();
  THROW_CHECK_EQ(num_points1, points2.size());
  residuals->resize(num_points1);
  for (size_t i = 0; i < num_points1; ++i) {
    (*residuals)[i] = ComputeSquaredSampsonError(
        points1[i].homogeneous(), points2[i].homogeneous(), E);
  }
}

void ComputeSquaredSampsonError(const std::vector<Eigen::Vector3d>& ray1,
                                const std::vector<Eigen::Vector3d>& ray2,
                                const Eigen::Matrix3d& E,
                                std::vector<double>* residuals) {
  const size_t num_ray1 = ray1.size();
  THROW_CHECK_EQ(num_ray1, ray2.size());
  residuals->resize(num_ray1);
  for (size_t i = 0; i < num_ray1; ++i) {
    (*residuals)[i] = ComputeSquaredSampsonError(ray1[i], ray2[i], E);
  }
}

}  // namespace colmap

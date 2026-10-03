// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/geometry/normalization.h"

#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>

namespace colmap {

std::pair<Eigen::AlignedBox3d, Eigen::Vector3d> ComputeBoundingBoxAndCentroid(
    double min_percentile,
    double max_percentile,
    std::vector<double> coords_x,
    std::vector<double> coords_y,
    std::vector<double> coords_z) {
  THROW_CHECK(!coords_x.empty());
  THROW_CHECK_EQ(coords_x.size(), coords_y.size());
  THROW_CHECK_EQ(coords_x.size(), coords_z.size());
  THROW_CHECK_GE(min_percentile, 0);
  THROW_CHECK_LE(min_percentile, 1);
  THROW_CHECK_GE(max_percentile, 0);
  THROW_CHECK_LE(max_percentile, 1);
  THROW_CHECK_LE(min_percentile, max_percentile);

  const size_t end_idx = coords_x.size() - 1;
  const size_t min_idx = std::min<size_t>(
      end_idx, static_cast<size_t>(std::floor(min_percentile * end_idx)));
  const size_t max_idx = std::min<size_t>(
      end_idx, static_cast<size_t>(std::ceil(max_percentile * end_idx)));

  std::nth_element(
      coords_x.begin(), coords_x.begin() + min_idx, coords_x.end());
  std::nth_element(coords_x.begin() + min_idx + 1,
                   coords_x.begin() + max_idx,
                   coords_x.end());
  std::nth_element(
      coords_y.begin(), coords_y.begin() + min_idx, coords_y.end());
  std::nth_element(coords_y.begin() + min_idx + 1,
                   coords_y.begin() + max_idx,
                   coords_y.end());
  std::nth_element(
      coords_z.begin(), coords_z.begin() + min_idx, coords_z.end());
  std::nth_element(coords_z.begin() + min_idx + 1,
                   coords_z.begin() + max_idx,
                   coords_z.end());

  const Eigen::Vector3d bbox_min(
      coords_x[min_idx], coords_y[min_idx], coords_z[min_idx]);
  const Eigen::Vector3d bbox_max(
      coords_x[max_idx], coords_y[max_idx], coords_z[max_idx]);

  Eigen::Vector3d centroid(0, 0, 0);
  const double normalization = 1.0 / (max_idx - min_idx + 1);
  for (size_t i = min_idx; i <= max_idx; ++i) {
    centroid(0) += normalization * coords_x[i];
    centroid(1) += normalization * coords_y[i];
    centroid(2) += normalization * coords_z[i];
  }

  return std::make_pair(Eigen::AlignedBox3d(bbox_min, bbox_max), centroid);
}

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

}  // namespace colmap

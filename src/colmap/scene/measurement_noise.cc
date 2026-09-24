// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/measurement_noise.h"

#include "colmap/util/logging.h"

#include <cmath>

namespace colmap {

Eigen::Matrix2d MeasurementNoiseModel::Covariance(double scale) const {
  THROW_CHECK_GT(scale, 0.0);
  THROW_CHECK_GT(base_sigma_px, 0.0);
  THROW_CHECK_GE(scale_gamma, 0.0);
  const double sigma = base_sigma_px * std::pow(scale, scale_gamma);
  return Eigen::Matrix2d::Identity() * sigma * sigma;
}

Eigen::Matrix2d MeasurementNoiseModel::Covariance(
    const FeatureKeypoint& keypoint) const {
  return Covariance(keypoint.ComputeScale());
}

std::vector<Point2D> KeypointsToPoint2Ds(const FeatureKeypoints& keypoints,
                                         const MeasurementNoiseModel& model) {
  std::vector<Point2D> points2D;
  points2D.reserve(keypoints.size());
  for (const auto& keypoint : keypoints) {
    Point2D point2D;
    point2D.xy = Eigen::Vector2d(keypoint.x, keypoint.y);
    // Degenerate keypoint scales (e.g. from foreign feature sources) are
    // treated as unit scale rather than failing the database load.
    // Legitimate sub-unit scales (fine-octave SIFT features) are preserved.
    const float raw_scale = keypoint.ComputeScale();
    const double scale = raw_scale > 0 ? raw_scale : 1.0;
    point2D.cov = model.Covariance(scale).cast<float>();
    points2D.push_back(point2D);
  }
  return points2D;
}

}  // namespace colmap

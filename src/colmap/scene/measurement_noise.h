// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/types.h"
#include "colmap/scene/point2d.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

// Isotropic 2D measurement noise model: the pixel standard deviation grows
// with the feature keypoint scale as sigma = base_sigma_px * scale^gamma.
// gamma = 1 is linear in scale, gamma = 0.5 sublinear (square-root),
// gamma = 0 a fixed sigma for all keypoints.
struct MeasurementNoiseModel {
  // Base measurement standard deviation in pixels at unit keypoint scale.
  double base_sigma_px = 1.0;

  // Exponent of the keypoint scale. Must be non-negative. Defaults to a
  // sublinear square-root mapping, calibrated on ETH3D DSLR.
  double scale_gamma = 0.5;

  // Isotropic 2x2 measurement covariance for a keypoint scale. Both the scale
  // and base_sigma_px must be positive.
  Eigen::Matrix2d Covariance(double scale) const;
  Eigen::Matrix2d Covariance(const FeatureKeypoint& keypoint) const;
};

// Convert feature keypoints to 2D points, deriving each point's measurement
// covariance from its keypoint scale with the given noise model.
std::vector<Point2D> KeypointsToPoint2Ds(
    const FeatureKeypoints& keypoints,
    const MeasurementNoiseModel& model = MeasurementNoiseModel());

}  // namespace colmap

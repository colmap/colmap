// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/types.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/two_view_geometry.h"

#include <optional>
#include <vector>

#include <Eigen/Core>

namespace colmap {

struct TwoViewRotationCovarianceOptions {
  // Standard deviation of the 2D image point observations in pixels.
  double point2D_stddev_px = 1.0;

  // Maximum standard deviation (in degrees) of the relative rotation along any
  // axis. Poorly or not constrained axes, e.g., of geometrically degenerate
  // configurations, are capped to it.
  double max_rotation_sigma_deg = 60.0;
};

// Estimate the marginal relative-rotation covariance in camera 1's local
// tangent frame (corresponding to right perturbation R21 * Exp(delta_theta))
// from the tangent Sampson Jacobian (or 2D tangent ray alignment for panoramic
// pairs) evaluated at `geometry.cam2_from_cam1` over `geometry.inlier_matches`.
// Returns nullopt if the relative pose is missing, has too few valid inliers,
// or if the eigendecomposition of its information fails.
std::optional<Eigen::Matrix3d> EstimateTwoViewRotationCovariance(
    const Camera& camera1,
    const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2,
    const std::vector<Eigen::Vector2d>& points2,
    const TwoViewGeometry& geometry,
    const TwoViewRotationCovarianceOptions& options =
        TwoViewRotationCovarianceOptions());

// Populate `edge.cam2_from_cam1_rotation_cov` in parallel, with `num_threads`
// threads (-1 = auto), for all valid edges in `pose_graph` that do not already
// have a rotation covariance set. Edges whose rotation covariance cannot be
// estimated are left unset.
void EstimatePoseGraphRotationCovariances(
    const DatabaseCache& database_cache,
    PoseGraph& pose_graph,
    const TwoViewRotationCovarianceOptions& options =
        TwoViewRotationCovarianceOptions(),
    int num_threads = -1);

}  // namespace colmap

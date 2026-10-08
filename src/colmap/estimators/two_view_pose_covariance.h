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

struct TwoViewPoseCovarianceOptions {
  // Number of threads for parallel pose graph covariance estimation (-1 =
  // auto).
  int num_threads = -1;

  // Minimum observation noise standard deviation in pixels, preventing
  // near-zero residuals on minimal configurations from producing infinite
  // information.
  double min_sigma_obs_px = 0.5;

  // Maximum standard deviation (in degrees) of the relative rotation along its
  // least constrained axis, beyond which the rotation is considered
  // geometrically degenerate and no rotation covariance is returned.
  double max_rotation_sigma_deg = 60.0;
};

struct TwoViewPoseCovariance {
  // 3x3 marginal rotation covariance in camera 1's local tangent frame
  // (corresponding to right perturbation R21 * Exp(delta_theta)). Unset if the
  // rotation is degenerate.
  std::optional<Eigen::Matrix3d> cov_rot;

  // 2x2 marginal translation covariance on the tangent space of S^2 at
  // cam2_from_cam1.translation().normalized(). Unset for panoramic pairs or if
  // the translation direction is degenerate.
  std::optional<Eigen::Matrix2d> cov_trans_tangent;

  // Estimated observation noise standard deviation in pixels.
  double sigma_obs_px = 0.0;

  // Number of valid inlier correspondences used.
  size_t num_inliers = 0;
};

// Estimate the marginal relative-pose uncertainty from the tangent Sampson
// Jacobian (or 2D tangent ray alignment for panoramic pairs) evaluated at
// `geometry.cam2_from_cam1` over `geometry.inlier_matches`. Returns nullopt if
// the relative pose is missing or has too few valid inliers.
std::optional<TwoViewPoseCovariance> EstimateTwoViewPoseCovariance(
    const Camera& camera1,
    const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2,
    const std::vector<Eigen::Vector2d>& points2,
    const TwoViewGeometry& geometry,
    const TwoViewPoseCovarianceOptions& options =
        TwoViewPoseCovarianceOptions());

// Populate `edge.cam2_from_cam1_rotation_cov` in parallel for all valid edges
// in `pose_graph` that do not already have a rotation covariance set. Edges
// whose rotation covariance cannot be estimated or is degenerate are left
// unset.
void EstimatePoseGraphCovariances(const DatabaseCache& database_cache,
                                  PoseGraph& pose_graph,
                                  const TwoViewPoseCovarianceOptions& options =
                                      TwoViewPoseCovarianceOptions());

}  // namespace colmap

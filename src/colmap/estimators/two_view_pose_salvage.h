// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/estimators/rotation_averaging.h"
#include "colmap/estimators/two_view_pose_covariance.h"
#include "colmap/feature/types.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/two_view_geometry.h"

#include <optional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colmap {

struct TwoViewPoseSalvageOptions {
  // Number of threads for parallel two-view pose salvage (-1 = auto).
  int num_threads = -1;

  // Random seed for RANSAC (-1 = non-deterministic).
  int random_seed = -1;

  // Maximum epipolar (tangent Sampson) error in pixels for RANSAC inlier
  // selection.
  double max_epipolar_error_px = 4.0;

  // Minimum number of cheirality-consistent inliers required to accept a
  // salvaged two-view pose.
  int min_num_inliers = 30;

  // Minimum median triangulation angle in degrees required to accept a
  // salvaged two-view pose (filters out pure-rotation / degenerate baselines).
  double min_tri_angle_deg = 1.5;

  // Maximum number of 5-point LO-RANSAC trials per candidate image pair.
  int max_num_trials = 1000;

  // Minimum number of 5-point LO-RANSAC trials per candidate image pair.
  int min_num_trials = 100;

  // RANSAC confidence level for dynamic trial termination.
  double confidence = 0.999;

  // Intrinsic 5-point minimal-sample rotation noise standard deviation in
  // degrees, added in quadrature to the leave-one-out prior covariance during
  // the in-loop 5-point hypothesis gate.
  double sample_rotation_sigma_deg = 2.0;

  // Chi-squared (3 DoF) tail significance level for the in-loop 5-point
  // rotation gate.
  double gate_significance = 1e-3;

  // Chi-squared (3 DoF) tail significance level for the a posteriori
  // consistency check between the refined relative rotation and the
  // leave-one-out global rotation prior.
  double posterior_significance = 1e-4;

  // Options for estimating the two-view pose covariance of the salvaged pose.
  TwoViewPoseCovarianceOptions covariance_options;

  bool Check() const;
};

struct SalvagedTwoViewPose {
  // Salvaged calibrated two-view geometry (including cam2_from_cam1, E,
  // tri_angle, and cheirality-filtered inlier_matches).
  TwoViewGeometry geometry;

  // Estimated 3x3 marginal rotation covariance of the salvaged pose in
  // camera 1's tangent frame.
  Eigen::Matrix3d cam2_from_cam1_rotation_cov = Eigen::Matrix3d::Zero();

  // Chi-squared (3 DoF) test statistic of the a posteriori rotation check.
  double posterior_statistic = 0.0;

  // p-value of the a posteriori rotation check.
  double posterior_p_value = 0.0;
};

// Attempt to salvage a two-view relative pose between two calibrated cameras
// using rotation-gated 5-point LO-RANSAC, nonlinear tangent Sampson
// refinement, cheirality and triangulation-angle checks, and an a posteriori
// chi-squared consistency test against the leave-one-out rotation prior.
std::optional<SalvagedTwoViewPose> SalvageTwoViewPose(
    const Camera& camera1,
    const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2,
    const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches,
    const Eigen::Quaterniond& prior_cam2_from_cam1_rotation,
    const Eigen::Matrix3d& prior_cam2_from_cam1_rotation_cov,
    const TwoViewPoseSalvageOptions& options = TwoViewPoseSalvageOptions(),
    double variance_factor = 1.0);

// Attempt to salvage invalid pose graph edges and unverified candidate match
// pairs between registered images in `reconstruction`.
//
// Candidate pairs are collected from:
// 1. Invalid edges in `pose_graph` between two registered images.
// 2. Raw match pairs in `database_cache.Matches()` between two registered
//    images that do not already have a valid edge in `pose_graph`.
//
// Leave-one-out rotation priors and covariances are computed via
// `EstimateRotationAveragingStatistics`, and candidate pairs are processed in
// parallel. Successfully salvaged pairs are enabled/inserted in `pose_graph`
// and updated in `correspondence_graph`. Returns the number of salvaged pairs.
size_t SalvageTwoViewPoses(const TwoViewPoseSalvageOptions& options,
                           const RotationEstimatorOptions& rotation_options,
                           const DatabaseCache& database_cache,
                           Reconstruction& reconstruction,
                           PoseGraph& pose_graph,
                           CorrespondenceGraph& correspondence_graph);

}  // namespace colmap

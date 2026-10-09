// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/types.h"

#include <optional>

namespace colmap {

struct RotationEstimatorOptions;

struct RelativeRotationStatisticsOptions {
  // Minimum redundancy of a rotation axis of a relative rotation for the axis
  // to be tested, in [0, 1]. The redundancy is the fraction of the information
  // about the axis that is provided by the rest of the pose graph, such that
  // the axes of bridge edges, which are only determined by the edge itself, are
  // not testable.
  double min_redundancy = 0.05;

  // Whether to estimate the variance factor of the relative rotation
  // covariances from the test statistics (Pope's tau-test) instead of assuming
  // that the covariances are correctly scaled (Baarda's w-test).
  bool estimate_variance_factor = true;
};

// Leave-one-out consistency test of a relative rotation with the rotation
// averaging solution, accounting for the uncertainty of both.
struct RelativeRotationStatistics {
  // Test statistic, normalized by the variance factor. Follows a chi-squared
  // distribution with num_dofs degrees of freedom for inlier relative
  // rotations.
  double statistic = 0.0;

  // Number of tested rotation axes, whose redundancy is at least
  // RelativeRotationStatisticsOptions::min_redundancy.
  int num_dofs = 0;

  // Minimum redundancy over the rotation axes.
  double min_redundancy = 0.0;

  // Probability of a statistic at least as large for an inlier relative
  // rotation. 1 if no rotation axis is testable.
  double p_value = 1.0;

  // Leave-one-out relative rotation estimated from the rest of the pose graph,
  // and its covariance in camera 1's tangent frame (right perturbation, scaled
  // by the variance factor).
  Eigen::Quaterniond cam2_from_cam1_rotation = Eigen::Quaterniond::Identity();
  Eigen::Matrix3d cam2_from_cam1_rotation_cov = Eigen::Matrix3d::Zero();
};

struct RotationAveragingStatistics {
  // Estimated (or assumed) variance factor of the relative rotation
  // covariances.
  double variance_factor = 1.0;

  // Statistics of the valid edges between images with poses, and of any
  // requested query_pairs in the same connected component.
  FlatHashMap<image_pair_t, RelativeRotationStatistics> edges;
};

// Probability that a chi-squared variable with num_dofs in [1, 3] degrees of
// freedom exceeds x.
double ChiSquaredSurvival(double x, int num_dofs);

// Tests the valid edges between images with poses for consistency with the
// rotations in the reconstruction, which are assumed to minimize the rotation
// averaging problem defined by the options (with the CERES backend). For edge
// k, the relative rotation residual r_k, whitened by the relative rotation
// covariance Sigma_k (identity without COVARIANCE reweighting), is compared
// with its posterior covariance P_k, obtained from the Gauss-Newton
// information matrix of the robustified problem. The influence of the edge on
// the solution is removed with the Woodbury identity, such that the test
// statistic is
//   (r_k^(-k))^T (Sigma_k + P_k^(-k))^-1 r_k^(-k) / sigma_0^2,
// restricted to the testable rotation axes, where r_k^(-k) and P_k^(-k) are the
// leave-one-out residual and posterior covariance and sigma_0^2 is the
// variance factor. Also populates the leave-one-out relative rotation and
// covariance for any additional pair in `query_pairs` whose images are in a
// posed connected component. Returns nullopt if the posterior covariance cannot
// be computed.
std::optional<RotationAveragingStatistics> EstimateRotationAveragingStatistics(
    const RotationEstimatorOptions& options,
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction,
    const FlatHashSet<image_pair_t>& query_pairs = {});

// Marks the edges whose p-value is below the significance level as invalid and
// returns their number.
int FilterEdgesByRelativeRotationStatistics(
    const RotationAveragingStatistics& statistics,
    double significance,
    PoseGraph& pose_graph);

}  // namespace colmap

// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/rotation_averaging_statistics.h"

#include "colmap/estimators/rotation_averaging.h"
#include "colmap/estimators/rotation_averaging_ceres.h"
#include "colmap/math/math.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"
#include "colmap/util/types.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>
#include <ceres/ceres.h>

namespace colmap {
namespace {

// Median of the chi-squared distribution with 3 degrees of freedom.
constexpr double kChiSquaredMedianThreeDof = 2.365973884375338;

// Minimum number of fully testable edges to estimate the variance factor.
constexpr int kMinNumEdgesForVarianceFactor = 10;

// Probability that a chi-squared variable with num_dofs in [1, 3] degrees of
// freedom exceeds x.
double ChiSquaredSurvival(double x, int num_dofs) {
  const double sqrt_half_x = std::sqrt(0.5 * std::max(0.0, x));
  switch (num_dofs) {
    case 1:
      return std::erfc(sqrt_half_x);
    case 2:
      return std::exp(-sqrt_half_x * sqrt_half_x);
    case 3:
      return std::erfc(sqrt_half_x) + 2.0 * sqrt_half_x / std::sqrt(M_PI) *
                                          std::exp(-sqrt_half_x * sqrt_half_x);
    default:
      LOG(FATAL_THROW) << "Unsupported number of degrees of freedom: "
                       << num_dofs;
      return 0.0;
  }
}

using Matrix3dRowMajor = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

// Whitened residual of an edge, with its Jacobians w.r.t. the tangent spaces
// of the variable parameter blocks, and the weight of its robust loss.
struct EdgeLinearization {
  Eigen::Vector3d residual;
  std::vector<const double*> parameter_blocks;
  std::vector<Matrix3dRowMajor> jacobians;
  double loss_weight = 1.0;
};

std::optional<EdgeLinearization> LinearizeEdge(
    const ceres::Problem& problem, ceres::ResidualBlockId residual_block) {
  std::vector<double*> parameter_blocks;
  problem.GetParameterBlocksForResidualBlock(residual_block, &parameter_blocks);

  EdgeLinearization linearization;
  std::vector<double*> jacobian_ptrs(parameter_blocks.size(), nullptr);
  linearization.jacobians.reserve(parameter_blocks.size());
  for (size_t i = 0; i < parameter_blocks.size(); ++i) {
    if (problem.IsParameterBlockConstant(parameter_blocks[i])) {
      continue;
    }
    THROW_CHECK_EQ(problem.ParameterBlockTangentSize(parameter_blocks[i]), 3);
    linearization.parameter_blocks.push_back(parameter_blocks[i]);
    jacobian_ptrs[i] = linearization.jacobians.emplace_back().data();
  }

  double cost = 0.0;
  if (!problem.EvaluateResidualBlock(residual_block,
                                     /*apply_loss_function=*/false,
                                     &cost,
                                     linearization.residual.data(),
                                     jacobian_ptrs.data())) {
    return std::nullopt;
  }

  const ceres::LossFunction* loss_function =
      problem.GetLossFunctionForResidualBlock(residual_block);
  if (loss_function != nullptr) {
    double rho[3];
    loss_function->Evaluate(linearization.residual.squaredNorm(), rho);
    linearization.loss_weight = rho[1];
  }
  return linearization;
}

// Leave-one-out test statistic of an edge before normalization by the variance
// factor, in the whitened residual space.
struct UnnormalizedEdgeStatistics {
  double statistic = 0.0;
  int num_dofs = 0;
  double min_redundancy = 1.0;
};

// With the whitened residual r, its posterior covariance P, and the robust
// loss weight w of the edge in the information matrix, the leave-one-out
// posterior covariance and residual are P (I - w P)^-1 and (I - w P)^-1 r. As
// all terms commute, the statistic is the sum over the eigen-decomposition
// P = sum_i p_i u_i u_i^T of (u_i^T r)^2 / ((1 - w p_i) (1 + (1 - w) p_i)),
// where 1 - w p_i is the redundancy of axis u_i.
UnnormalizedEdgeStatistics ComputeLeaveOneOutStatistics(
    const Eigen::Vector3d& residual,
    const Eigen::Matrix3d& posterior_cov,
    double loss_weight,
    double min_redundancy) {
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(posterior_cov);
  UnnormalizedEdgeStatistics statistics;
  for (int i = 0; i < 3; ++i) {
    const double posterior_var = std::max(0.0, eig.eigenvalues()(i));
    const double redundancy =
        std::clamp(1.0 - loss_weight * posterior_var, 0.0, 1.0);
    statistics.min_redundancy = std::min(statistics.min_redundancy, redundancy);
    if (redundancy < min_redundancy) {
      continue;
    }
    const double projected_residual = eig.eigenvectors().col(i).dot(residual);
    statistics.statistic +=
        projected_residual * projected_residual /
        (redundancy * (1.0 + (1.0 - loss_weight) * posterior_var));
    ++statistics.num_dofs;
  }
  return statistics;
}

// Computes the statistics of the valid edges of a connected pose graph.
bool ComputeComponentStatistics(
    const RotationEstimatorOptions& options,
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction,
    FlatHashMap<image_pair_t, UnnormalizedEdgeStatistics>& edge_statistics) {
  CeresRotationAverager averager(options, pose_graph, reconstruction);
  ceres::Problem& problem = averager.Problem();

  std::vector<std::pair<image_pair_t, EdgeLinearization>> linearizations;
  linearizations.reserve(averager.EdgeResidualBlocks().size());
  std::vector<std::pair<const double*, const double*>> covariance_blocks;
  FlatHashSet<std::pair<const double*, const double*>, PairHash>
      covariance_block_set;
  for (const auto& [pair_id, residual_block] : averager.EdgeResidualBlocks()) {
    std::optional<EdgeLinearization> linearization =
        LinearizeEdge(problem, residual_block);
    if (!linearization.has_value()) {
      return false;
    }
    const std::vector<const double*>& blocks = linearization->parameter_blocks;
    for (size_t i = 0; i < blocks.size(); ++i) {
      for (size_t j = i; j < blocks.size(); ++j) {
        std::pair<const double*, const double*> block_pair =
            std::minmax(blocks[i], blocks[j]);
        if (covariance_block_set.insert(block_pair).second) {
          covariance_blocks.push_back(block_pair);
        }
      }
    }
    linearizations.emplace_back(pair_id, std::move(*linearization));
  }

  // The loss function is applied to obtain the information matrix of the
  // robustified problem, with the gauge fixed by the constant blocks.
  ceres::Covariance::Options covariance_options;
  covariance_options.num_threads = GetEffectiveNumThreads(options.num_threads);
  covariance_options.apply_loss_function = true;
  ceres::Covariance covariance(covariance_options);
  if (!covariance.Compute(covariance_blocks, &problem)) {
    return false;
  }

  Matrix3dRowMajor block_cov;
  for (const auto& [pair_id, linearization] : linearizations) {
    const std::vector<const double*>& blocks = linearization.parameter_blocks;
    Eigen::Matrix3d posterior_cov = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < blocks.size(); ++i) {
      for (size_t j = 0; j < blocks.size(); ++j) {
        covariance.GetCovarianceBlockInTangentSpace(
            blocks[i], blocks[j], block_cov.data());
        posterior_cov += linearization.jacobians[i] * block_cov *
                         linearization.jacobians[j].transpose();
      }
    }
    edge_statistics.emplace(
        pair_id,
        ComputeLeaveOneOutStatistics(
            linearization.residual,
            0.5 * (posterior_cov + posterior_cov.transpose()),
            linearization.loss_weight,
            options.rotation_statistics.min_redundancy));
  }
  return true;
}

}  // namespace

std::optional<RotationAveragingStatistics> EstimateRotationAveragingStatistics(
    const RotationEstimatorOptions& options,
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction) {
  // Evaluate the problem at the given rotations.
  RotationEstimatorOptions problem_options = options;
  problem_options.skip_initialization = true;

  PoseGraph posed_pose_graph = pose_graph;
  for (auto& [pair_id, edge] : posed_pose_graph.Edges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (edge.valid && (!reconstruction.Image(image_id1).HasPose() ||
                       !reconstruction.Image(image_id2).HasPose())) {
      edge.valid = false;
    }
  }
  if (options.reweighting == RotationAveragingReweighting::COVARIANCE) {
    RegularizeRotationCovariances(options, posed_pose_graph);
  }

  FlatHashMap<image_pair_t, UnnormalizedEdgeStatistics> edge_statistics;
  for (const FlatHashSet<image_t>& image_ids :
       posed_pose_graph.ConnectedImageIdsForFrameComponents(
           reconstruction, /*filter_unregistered=*/true)) {
    PoseGraph component_pose_graph = posed_pose_graph;
    component_pose_graph.InvalidatePairsOutsideActiveImageIds(image_ids);
    if (component_pose_graph.ValidEdges().begin() ==
        component_pose_graph.ValidEdges().end()) {
      continue;
    }
    if (!ComputeComponentStatistics(problem_options,
                                    component_pose_graph,
                                    reconstruction,
                                    edge_statistics)) {
      LOG(WARNING) << "Failed to compute the posterior covariance of the "
                      "rotation averaging solution";
      return std::nullopt;
    }
  }

  RotationAveragingStatistics statistics;
  if (options.rotation_statistics.estimate_variance_factor) {
    std::vector<double> full_rank_statistics;
    for (const auto& [_, edge] : edge_statistics) {
      if (edge.num_dofs == 3) {
        full_rank_statistics.push_back(edge.statistic);
      }
    }
    if (full_rank_statistics.size() >= kMinNumEdgesForVarianceFactor) {
      statistics.variance_factor =
          Median(full_rank_statistics) / kChiSquaredMedianThreeDof;
    } else {
      LOG(WARNING) << "Too few testable edges to estimate the variance factor";
    }
  }

  statistics.edges.reserve(edge_statistics.size());
  for (const auto& [pair_id, edge] : edge_statistics) {
    RelativeRotationStatistics& edge_out = statistics.edges[pair_id];
    edge_out.num_dofs = edge.num_dofs;
    edge_out.min_redundancy = edge.min_redundancy;
    if (edge.num_dofs > 0) {
      edge_out.statistic = edge.statistic / statistics.variance_factor;
      edge_out.p_value = ChiSquaredSurvival(edge_out.statistic, edge.num_dofs);
    }
  }
  return statistics;
}

int FilterEdgesByRelativeRotationStatistics(
    const RotationAveragingStatistics& statistics,
    double significance,
    PoseGraph& pose_graph) {
  int num_invalid = 0;
  for (const auto& [pair_id, edge] : statistics.edges) {
    if (edge.p_value < significance && pose_graph.IsValid(pair_id)) {
      pose_graph.SetInvalidEdge(pair_id);
      ++num_invalid;
    }
  }
  LOG(INFO) << "Marked " << num_invalid
            << " image pairs as invalid with statistically inconsistent "
               "relative rotation (variance factor: "
            << statistics.variance_factor << ")";
  return num_invalid;
}

}  // namespace colmap

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

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

namespace colmap {

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

namespace {

// Median of the chi-squared distribution with 3 degrees of freedom.
constexpr double kChiSquaredMedianThreeDof = 2.365973884375338;

// Minimum number of fully testable edges to estimate the variance factor.
constexpr int kMinNumEdgesForVarianceFactor = 10;

// Minimum redundancy used to bound the leave-one-out covariance on untestable
// bridge axes.
constexpr double kMinLeaveOneOutRedundancy = 1e-3;

using Matrix3dRowMajor = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

// Whitened residual of an edge, with its Jacobians w.r.t. the tangent spaces
// of the variable parameter blocks, and the weight of its robust loss.
struct EdgeLinearization {
  Eigen::Vector3d residual;
  std::vector<const double*> parameter_blocks;
  std::vector<Matrix3dRowMajor> jacobians;
  double loss_weight = 1.0;
  Eigen::Quaterniond hat_cam2_from_cam1 = Eigen::Quaterniond::Identity();
  std::optional<Eigen::Matrix3d> edge_cov;
  bool is_valid_edge = true;
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

void AddCovarianceBlocksForLinearization(
    const EdgeLinearization& linearization,
    FlatHashSet<std::pair<const double*, const double*>, PairHash>&
        covariance_block_set,
    std::vector<std::pair<const double*, const double*>>& covariance_blocks) {
  const std::vector<const double*>& blocks = linearization.parameter_blocks;
  for (size_t i = 0; i < blocks.size(); ++i) {
    for (size_t j = i; j < blocks.size(); ++j) {
      std::pair<const double*, const double*> block_pair =
          std::minmax(blocks[i], blocks[j]);
      if (covariance_block_set.insert(block_pair).second) {
        covariance_blocks.push_back(block_pair);
      }
    }
  }
}

// Leave-one-out test statistic and relative rotation prior of an edge before
// normalization by the variance factor.
struct UnnormalizedEdgeStatistics {
  double statistic = 0.0;
  int num_dofs = 0;
  double min_redundancy = 1.0;
  bool is_valid_edge = true;
  Eigen::Quaterniond cam2_from_cam1_rotation = Eigen::Quaterniond::Identity();
  Eigen::Matrix3d cam2_from_cam1_rotation_cov = Eigen::Matrix3d::Zero();
};

// With the whitened residual r, its posterior covariance P, and the robust
// loss weight w of the edge in the information matrix, the leave-one-out
// posterior covariance and residual are P (I - w P)^-1 and (I - w P)^-1 r. As
// all terms commute, the statistic is the sum over the eigen-decomposition
// P = sum_i p_i u_i u_i^T of (u_i^T r)^2 / ((1 - w p_i) (1 + (1 - w) p_i)),
// where 1 - w p_i is the redundancy of axis u_i.
UnnormalizedEdgeStatistics ComputeLeaveOneOutStatistics(
    const EdgeLinearization& linearization,
    const Eigen::Matrix3d& posterior_cov,
    double min_redundancy) {
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(posterior_cov);
  UnnormalizedEdgeStatistics statistics;
  statistics.is_valid_edge = linearization.is_valid_edge;
  Eigen::Vector3d loo_whitened_shift = Eigen::Vector3d::Zero();
  Eigen::Vector3d loo_whitened_variances = Eigen::Vector3d::Zero();
  const double safe_min_redundancy =
      std::max(min_redundancy, kMinLeaveOneOutRedundancy);
  for (int i = 0; i < 3; ++i) {
    const double posterior_var = std::max(0.0, eig.eigenvalues()(i));
    const double redundancy =
        std::clamp(1.0 - linearization.loss_weight * posterior_var, 0.0, 1.0);
    statistics.min_redundancy = std::min(statistics.min_redundancy, redundancy);
    const double safe_redundancy = std::max(redundancy, safe_min_redundancy);
    loo_whitened_variances(i) = posterior_var / safe_redundancy;
    const double projected_residual =
        eig.eigenvectors().col(i).dot(linearization.residual);
    loo_whitened_shift += ((1.0 - safe_redundancy) / safe_redundancy) *
                          projected_residual * eig.eigenvectors().col(i);
    if (redundancy < min_redundancy) {
      continue;
    }
    statistics.statistic +=
        projected_residual * projected_residual /
        (redundancy *
         (1.0 + (1.0 - linearization.loss_weight) * posterior_var));
    ++statistics.num_dofs;
  }

  const Eigen::Matrix3d loo_whitened_cov = eig.eigenvectors() *
                                           loo_whitened_variances.asDiagonal() *
                                           eig.eigenvectors().transpose();
  Eigen::Vector3d loo_shift = loo_whitened_shift;
  if (linearization.edge_cov.has_value()) {
    const Eigen::Matrix3d inv_left_sqrt_info =
        linearization.edge_cov->inverse().llt().matrixL().transpose().solve(
            Eigen::Matrix3d::Identity());
    loo_shift = inv_left_sqrt_info * loo_whitened_shift;
    statistics.cam2_from_cam1_rotation_cov =
        inv_left_sqrt_info * loo_whitened_cov * inv_left_sqrt_info.transpose();
  } else {
    statistics.cam2_from_cam1_rotation_cov = loo_whitened_cov;
  }

  const double shift_angle = loo_shift.norm();
  const Eigen::Quaterniond delta_rot =
      shift_angle > 1e-12 ? Eigen::Quaterniond(Eigen::AngleAxisd(
                                shift_angle, loo_shift / shift_angle))
                          : Eigen::Quaterniond::Identity();
  statistics.cam2_from_cam1_rotation =
      (linearization.hat_cam2_from_cam1 * delta_rot).normalized();
  return statistics;
}

// Computes the statistics of the valid edges of a connected pose graph and of
// any additional query_pairs in the component.
bool ComputeComponentStatistics(
    const RotationEstimatorOptions& options,
    const PoseGraph& pose_graph,
    const FlatHashSet<image_t>& image_ids,
    const FlatHashSet<image_pair_t>& query_pairs,
    Reconstruction& reconstruction,
    FlatHashMap<image_pair_t, UnnormalizedEdgeStatistics>& edge_statistics) {
  CeresRotationAverager averager(options, pose_graph, reconstruction);
  ceres::Problem& problem = averager.Problem();
  const bool use_covariance =
      options.reweighting == RotationAveragingReweighting::COVARIANCE;

  std::vector<std::pair<image_pair_t, EdgeLinearization>> linearizations;
  linearizations.reserve(averager.EdgeResidualBlocks().size() +
                         query_pairs.size());
  std::vector<std::pair<const double*, const double*>> covariance_blocks;
  FlatHashSet<std::pair<const double*, const double*>, PairHash>
      covariance_block_set;
  for (const auto& [pair_id, residual_block] : averager.EdgeResidualBlocks()) {
    std::optional<EdgeLinearization> linearization =
        LinearizeEdge(problem, residual_block);
    if (!linearization.has_value()) {
      return false;
    }
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    linearization->hat_cam2_from_cam1 =
        reconstruction.Image(image_id2).CamFromWorld().rotation() *
        reconstruction.Image(image_id1).CamFromWorld().rotation().inverse();
    if (use_covariance) {
      linearization->edge_cov =
          pose_graph.Edges().at(pair_id).cam2_from_cam1_rotation_cov;
    }
    AddCovarianceBlocksForLinearization(
        *linearization, covariance_block_set, covariance_blocks);
    linearizations.emplace_back(pair_id, std::move(*linearization));
  }

  for (const image_pair_t pair_id : query_pairs) {
    if (averager.EdgeResidualBlocks().count(pair_id) > 0) {
      continue;
    }
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (!image_ids.count(image_id1) || !image_ids.count(image_id2)) {
      continue;
    }
    const Eigen::Quaterniond hat_cam2_from_cam1 =
        reconstruction.Image(image_id2).CamFromWorld().rotation() *
        reconstruction.Image(image_id1).CamFromWorld().rotation().inverse();
    const ceres::ResidualBlockId residual_block =
        averager.AddRelativeRotationResidual(image_id1,
                                             image_id2,
                                             hat_cam2_from_cam1,
                                             /*loss_function=*/nullptr);
    if (residual_block == nullptr) {
      continue;
    }
    std::optional<EdgeLinearization> linearization =
        LinearizeEdge(problem, residual_block);
    problem.RemoveResidualBlock(residual_block);
    if (!linearization.has_value()) {
      return false;
    }
    linearization->loss_weight = 0.0;
    linearization->hat_cam2_from_cam1 = hat_cam2_from_cam1;
    linearization->is_valid_edge = false;
    AddCovarianceBlocksForLinearization(
        *linearization, covariance_block_set, covariance_blocks);
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
            linearization,
            0.5 * (posterior_cov + posterior_cov.transpose()),
            options.rotation_statistics.min_redundancy));
  }
  return true;
}

}  // namespace

std::optional<RotationAveragingStatistics> EstimateRotationAveragingStatistics(
    const RotationEstimatorOptions& options,
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction,
    const FlatHashSet<image_pair_t>& query_pairs) {
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
                                    image_ids,
                                    query_pairs,
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
      if (edge.is_valid_edge && edge.num_dofs == 3) {
        full_rank_statistics.push_back(edge.statistic);
      }
    }
    if (full_rank_statistics.size() >= kMinNumEdgesForVarianceFactor) {
      const double median_statistic = Median(full_rank_statistics);
      if (median_statistic > 0.0) {
        statistics.variance_factor =
            median_statistic / kChiSquaredMedianThreeDof;
      }
    } else {
      LOG(WARNING) << "Too few testable edges to estimate the variance factor";
    }
  }

  statistics.edges.reserve(edge_statistics.size());
  for (const auto& [pair_id, edge] : edge_statistics) {
    RelativeRotationStatistics& edge_out = statistics.edges[pair_id];
    edge_out.num_dofs = edge.num_dofs;
    edge_out.min_redundancy = edge.min_redundancy;
    if (edge.is_valid_edge && edge.num_dofs > 0) {
      edge_out.statistic = edge.statistic / statistics.variance_factor;
      edge_out.p_value = ChiSquaredSurvival(edge_out.statistic, edge.num_dofs);
    }
    edge_out.cam2_from_cam1_rotation = edge.cam2_from_cam1_rotation;
    edge_out.cam2_from_cam1_rotation_cov =
        0.5 * statistics.variance_factor *
        (edge.cam2_from_cam1_rotation_cov +
         edge.cam2_from_cam1_rotation_cov.transpose());
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

// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/rotation_averaging_ceres.h"

#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/motion_averaging.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/logging.h"
#include "colmap/util/string.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace colmap {
namespace {

double* MaybeGetSensorRotationBlock(const Image& image) {
  if (image.IsRefInFrame()) return nullptr;
  const sensor_t sensor_id = image.CameraPtr()->SensorId();
  Rigid3d& sensor_from_rig =
      image.FramePtr()->RigPtr()->SensorFromRig(sensor_id);
  return sensor_from_rig.rotation().coeffs().data();
}

std::optional<PoseGraph> MaybeFilterPoseGraphToActiveImages(
    const PoseGraph& pose_graph, const FlatHashSet<image_t>& active_image_ids) {
  for (const auto& [pair_id, _] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (!active_image_ids.count(image_id1) ||
        !active_image_ids.count(image_id2)) {
      PoseGraph filtered_pose_graph = pose_graph;
      filtered_pose_graph.InvalidatePairsOutsideActiveImageIds(
          active_image_ids);
      return filtered_pose_graph;
    }
  }
  return std::nullopt;
}

}  // namespace

struct CeresRotationAverager::PoseGraphInfo {
  FlatHashSet<image_t> component_image_ids;
  FlatHashSet<image_t> edge_image_ids;
  int max_num_matches = 0;
};

struct CeresRotationAverager::PreparedWeights {
  double loss_function_scale = 0.0;
  std::shared_ptr<ceres::LossFunction> loss;
  // In PoseGraph::ValidEdges() iteration order; empty unless COVARIANCE.
  std::vector<Eigen::Matrix3d> edge_left_sqrt_infos;
};

CeresRotationAverager::PoseGraphInfo CeresRotationAverager::CheckPoseGraph(
    const RotationEstimatorOptions& options,
    const PoseGraph& pose_graph,
    const Reconstruction& reconstruction) {
  THROW_CHECK_NOTNULL(options.ceres);
  PoseGraphInfo info;
  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (!reconstruction.ExistsImage(image_id1) ||
        !reconstruction.ExistsImage(image_id2)) {
      throw std::invalid_argument("rotation edge references unknown image");
    }
    info.edge_image_ids.insert(image_id1);
    info.edge_image_ids.insert(image_id2);
    info.max_num_matches = std::max(info.max_num_matches, edge.num_matches);
  }
  if (info.edge_image_ids.empty()) {
    throw std::invalid_argument("Ceres rotation averaging requires edges");
  }
  std::vector<FlatHashSet<image_t>> image_components =
      pose_graph.ConnectedImageIdsForFrameComponents(
          reconstruction, /*filter_unregistered=*/false);
  if (image_components.size() != 1) {
    throw std::invalid_argument(
        "Ceres rotation averaging requires a connected pose graph");
  }
  info.component_image_ids = std::move(image_components.front());
  // Validate fixed sensors before replacing invalid initial rotations.
  for (const image_t image_id : info.component_image_ids) {
    const auto& image = reconstruction.Image(image_id);
    if (image.IsRefInFrame()) continue;
    const auto& pose = image.FramePtr()->RigPtr()->MaybeSensorFromRig(
        image.CameraPtr()->SensorId());
    const bool refine_sensor =
        options.refine_sensor_from_rig &&
        (!pose.has_value() || pose->translation().hasNaN());
    if (!refine_sensor &&
        (!pose.has_value() || !pose->rotation().coeffs().allFinite())) {
      throw std::invalid_argument(
          "rotation averaging requires finite sensor rotations");
    }
  }
  return info;
}

CeresRotationAverager::PreparedWeights CeresRotationAverager::PrepareWeights(
    const RotationEstimatorOptions& options, const PoseGraph& pose_graph) {
  const CeresRotationAveragerOptions& ceres_options =
      *THROW_CHECK_NOTNULL(options.ceres);
  const bool use_covariance =
      options.reweighting == RotationAveragingReweighting::COVARIANCE;
  PreparedWeights weights;
  // The covariance loss scale is expressed in standard deviations of the
  // whitened residuals, whereas the default loss scale is in degrees.
  weights.loss_function_scale =
      use_covariance ? ceres_options.covariance_loss_scale
                     : DegToRad(ceres_options.loss_function_scale);
  weights.loss = CreateCeresLossFunction(ceres_options.loss_function_type,
                                         weights.loss_function_scale);
  if (!use_covariance) {
    return weights;
  }

  const double floor_sigma_deg = ceres_options.covariance_sigma_floor_deg;
  if (!std::isfinite(floor_sigma_deg) || floor_sigma_deg < 0.0) {
    throw std::invalid_argument(
        "covariance_sigma_floor_deg must be finite and non-negative");
  }
  // Isotropic floor added in quadrature to the relative rotation covariances.
  const double floor_sigma = DegToRad(floor_sigma_deg);
  const Eigen::Matrix3d floor_cov =
      floor_sigma * floor_sigma * Eigen::Matrix3d::Identity();

  weights.edge_left_sqrt_infos.reserve(pose_graph.NumEdges());
  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (!edge.cam2_from_cam1_rotation_cov.has_value()) {
      throw std::invalid_argument(StringPrintf(
          "COVARIANCE reweighting requires a rotation covariance for every "
          "valid edge, but edge (%u, %u) has none. Call "
          "MaybeEstimatePoseGraphRotationCovariances (or "
          "EstimatePoseGraphRotationCovariances) first.",
          image_id1,
          image_id2));
    }
    const Eigen::Matrix3d floored_cov =
        edge.cam2_from_cam1_rotation_cov->cast<double>() + floor_cov;
    std::optional<Eigen::Matrix3d> left_sqrt_info =
        internal::TryComputeLeftSqrtInformation<3>(floored_cov);
    if (!left_sqrt_info.has_value()) {
      throw std::invalid_argument(StringPrintf(
          "Floored rotation covariance of edge (%u, %u) is not finite and "
          "symmetric positive definite",
          image_id1,
          image_id2));
    }
    weights.edge_left_sqrt_infos.push_back(*left_sqrt_info);
  }
  return weights;
}

void CeresRotationAverager::InitializeRotations(
    const RotationEstimatorOptions& options,
    const PoseGraph& pose_graph,
    const FlatHashSet<image_t>& component_image_ids,
    Reconstruction& reconstruction) {
  for (const image_t image_id : component_image_ids) {
    const auto& image = reconstruction.Image(image_id);
    if (image.IsRefInFrame()) continue;
    auto& pose = image.FramePtr()->RigPtr()->MaybeSensorFromRig(
        image.CameraPtr()->SensorId());
    if (pose.has_value() && !pose->rotation().coeffs().allFinite()) {
      pose.reset();
    }
  }
  if (!options.skip_initialization) {
    InitializeFromMaximumSpanningTree(
        pose_graph,
        component_image_ids,
        reconstruction,
        options.refine_sensor_from_rig,
        /*use_rotation_covariance=*/options.reweighting ==
            RotationAveragingReweighting::COVARIANCE);
  }
}

CeresRotationAveragerOptions::CeresRotationAveragerOptions() {
  solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  solver_options.max_num_iterations = 100;
  solver_options.num_threads = -1;
}

CeresRotationAverager::CeresRotationAverager(
    const RotationEstimatorOptions& options,
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction)
    : solver_options(THROW_CHECK_NOTNULL(options.ceres)->solver_options),
      reconstruction_(reconstruction) {
  const PoseGraphInfo graph_info =
      CheckPoseGraph(options, pose_graph, reconstruction);
  PreparedWeights weights = PrepareWeights(options, pose_graph);
  InitializeRotations(
      options, pose_graph, graph_info.component_image_ids, reconstruction);
  BuildProblem(options, pose_graph, graph_info, std::move(weights));
}

CeresRotationAverager::CeresRotationAverager(
    const RotationEstimatorOptions& options,
    const PoseGraph& pose_graph,
    const PoseGraphInfo& graph_info,
    PreparedWeights weights,
    Reconstruction& reconstruction)
    : solver_options(THROW_CHECK_NOTNULL(options.ceres)->solver_options),
      reconstruction_(reconstruction) {
  BuildProblem(options, pose_graph, graph_info, std::move(weights));
}

void CeresRotationAverager::BuildProblem(
    const RotationEstimatorOptions& options,
    const PoseGraph& pose_graph,
    const PoseGraphInfo& graph_info,
    PreparedWeights weights) {
  const CeresRotationAveragerOptions& ceres_options = *options.ceres;
  solver_options.num_threads =
      GetEffectiveNumThreads(solver_options.num_threads);
  if (VLOG_IS_ON(2)) {
    solver_options.minimizer_progress_to_stdout = true;
  }
  const bool use_covariance =
      options.reweighting == RotationAveragingReweighting::COVARIANCE;

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  SetupParameterBlocks(options, graph_info.edge_image_ids);

  std::shared_ptr<ceres::LossFunction> loss = std::move(weights.loss);
  size_t edge_idx = 0;
  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    const Eigen::Matrix3d* left_sqrt_info =
        use_covariance ? &weights.edge_left_sqrt_infos.at(edge_idx++) : nullptr;
    if (options.reweighting ==
            RotationAveragingReweighting::INLIER_MATCH_COUNT &&
        graph_info.max_num_matches > 0) {
      if (edge.num_matches == 0) continue;
      loss = CreateCeresLossFunction(
          ceres_options.loss_function_type,
          weights.loss_function_scale,
          double(edge.num_matches) / graph_info.max_num_matches);
    }
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    AddRelativeRotationResidualImpl(image_id1,
                                    image_id2,
                                    edge.cam2_from_cam1.rotation(),
                                    loss,
                                    /*cam2_from_cam1_cov=*/nullptr,
                                    left_sqrt_info);
  }
  THROW_CHECK_EQ(edge_idx, weights.edge_left_sqrt_infos.size());
}

void CeresRotationAverager::SetupParameterBlocks(
    const RotationEstimatorOptions& options,
    const FlatHashSet<image_t>& image_ids) {
  const auto add_rotation = [&](Eigen::Map<Eigen::Quaterniond> rotation) {
    double* block = rotation.coeffs().data();
    if (!problem_->HasParameterBlock(block)) {
      problem_->AddParameterBlock(
          block, 4, CreateEigenQuaternionManifold().release());
    }
    return block;
  };
  for (const image_t image_id : image_ids) {
    const Image& image = reconstruction_.Image(image_id);
    Frame& frame = *image.FramePtr();
    if (!frame.HasPose()) {
      frame.SetRigFromWorld(Rigid3d(
          Eigen::Quaterniond::Identity(),
          Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN())));
    }
    add_rotation(frame.RigFromWorld().rotation());
    if (!image.IsRefInFrame()) {
      auto& pose =
          frame.RigPtr()->MaybeSensorFromRig(image.CameraPtr()->SensorId());
      if (!pose.has_value()) {
        pose = Rigid3d(Eigen::Quaterniond::Identity(),
                       Eigen::Vector3d::Constant(
                           std::numeric_limits<double>::quiet_NaN()));
      }
      double* sensor = add_rotation(pose->rotation());
      if (!options.refine_sensor_from_rig || !pose->translation().hasNaN()) {
        problem_->SetParameterBlockConstant(sensor);
      }
    }
  }
  const image_t root_image_id =
      *std::min_element(image_ids.begin(), image_ids.end());
  Frame& root_frame =
      reconstruction_.Frame(reconstruction_.Image(root_image_id).FrameId());
  problem_->SetParameterBlockConstant(
      root_frame.RigFromWorld().rotation().coeffs().data());
}

ceres::Solver::Summary CeresRotationAverager::Solve() {
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, problem_.get(), &summary);
  if (summary.IsSolutionUsable()) {
    for (const auto& [frame_id, frame] : reconstruction_.Frames()) {
      if (frame.HasPose() &&
          problem_->HasParameterBlock(
              frame.RigFromWorld().rotation().coeffs().data())) {
        reconstruction_.RegisterFrame(frame_id);
      }
    }
  }
  return summary;
}

ceres::Problem& CeresRotationAverager::Problem() { return *problem_; }
const ceres::Problem& CeresRotationAverager::Problem() const {
  return *problem_;
}

void CeresRotationAverager::AddRelativeRotationResidual(
    const image_t image_id1,
    const image_t image_id2,
    const Eigen::Quaterniond& cam2_from_cam1,
    const std::shared_ptr<ceres::LossFunction>& loss_function,
    const std::optional<Eigen::Matrix3d>& cam2_from_cam1_cov) {
  AddRelativeRotationResidualImpl(
      image_id1,
      image_id2,
      cam2_from_cam1,
      loss_function,
      cam2_from_cam1_cov.has_value() ? &*cam2_from_cam1_cov : nullptr,
      /*left_sqrt_info=*/nullptr);
}

void CeresRotationAverager::AddRelativeRotationResidualImpl(
    const image_t image_id1,
    const image_t image_id2,
    const Eigen::Quaterniond& cam2_from_cam1,
    const std::shared_ptr<ceres::LossFunction>& loss_function,
    const Eigen::Matrix3d* cam2_from_cam1_cov,
    const Eigen::Matrix3d* left_sqrt_info) {
  const Image& image1 = reconstruction_.Image(image_id1);
  const Image& image2 = reconstruction_.Image(image_id2);
  Frame& frame1 = *image1.FramePtr();
  Frame& frame2 = *image2.FramePtr();
  double* rig1_from_world_rot_ptr =
      frame1.RigFromWorld().rotation().coeffs().data();
  double* rig2_from_world_rot_ptr =
      frame2.RigFromWorld().rotation().coeffs().data();
  THROW_CHECK(problem_->HasParameterBlock(rig1_from_world_rot_ptr));
  THROW_CHECK(problem_->HasParameterBlock(rig2_from_world_rot_ptr));
  double* sensor1_from_rig_rot_ptr = MaybeGetSensorRotationBlock(image1);
  double* sensor2_from_rig_rot_ptr = MaybeGetSensorRotationBlock(image2);
  THROW_CHECK(sensor1_from_rig_rot_ptr == nullptr ||
              problem_->HasParameterBlock(sensor1_from_rig_rot_ptr));
  THROW_CHECK(sensor2_from_rig_rot_ptr == nullptr ||
              problem_->HasParameterBlock(sensor2_from_rig_rot_ptr));
  const bool same_frame = image1.FrameId() == image2.FrameId();
  if (same_frame && sensor1_from_rig_rot_ptr == sensor2_from_rig_rot_ptr) {
    LOG(WARNING) << "Skipping self-loop for image pair " << image_id1 << ", "
                 << image_id2;
    return;
  }
  ceres::LossFunction* loss = loss_function.get();
  if (loss != nullptr) {
    losses_.try_emplace(loss, loss_function);
  }
  // Whitens the residual with the covariance of the relative rotation, if any.
  const auto maybe_whiten =
      [&](ceres::CostFunction* cost_function) -> ceres::CostFunction* {
    if (left_sqrt_info != nullptr) {
      return DynamicCovarianceWeightedCostFunction<3>::FromLeftSqrtInformation(
          *left_sqrt_info, cost_function);
    }
    if (cam2_from_cam1_cov != nullptr) {
      return new DynamicCovarianceWeightedCostFunction<3>(*cam2_from_cam1_cov,
                                                          cost_function);
    }
    return cost_function;
  };
  if (sensor1_from_rig_rot_ptr == nullptr &&
      sensor2_from_rig_rot_ptr == nullptr && !same_frame) {
    problem_->AddResidualBlock(
        maybe_whiten(RelativeRotationCostFunctor::Create(cam2_from_cam1)),
        loss,
        rig1_from_world_rot_ptr,
        rig2_from_world_rot_ptr);
    return;
  }
  std::vector<double*> parameter_blocks;
  if (!same_frame)
    parameter_blocks = {rig1_from_world_rot_ptr, rig2_from_world_rot_ptr};
  const auto get_sensor_index_in_parameters =
      [&](double* sensor_from_rig_rot_ptr) {
        if (sensor_from_rig_rot_ptr == nullptr) return -1;
        const auto it = std::find(parameter_blocks.begin(),
                                  parameter_blocks.end(),
                                  sensor_from_rig_rot_ptr);
        const int index = static_cast<int>(it - parameter_blocks.begin());
        if (it == parameter_blocks.end())
          parameter_blocks.push_back(sensor_from_rig_rot_ptr);
        return index;
      };
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<RelativeRotationCostFunctor, 8>(
          new RelativeRotationCostFunctor{
              cam2_from_cam1,
              get_sensor_index_in_parameters(sensor1_from_rig_rot_ptr),
              get_sensor_index_in_parameters(sensor2_from_rig_rot_ptr),
              same_frame});
  for (size_t i = 0; i < parameter_blocks.size(); ++i)
    cost_function->AddParameterBlock(4);
  cost_function->SetNumResiduals(3);
  problem_->AddResidualBlock(
      maybe_whiten(cost_function), loss, parameter_blocks);
}

std::unique_ptr<CeresRotationAverager> CreateDefaultCeresRotationAverager(
    const RotationEstimatorOptions& options,
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction) {
  return std::make_unique<CeresRotationAverager>(
      options, pose_graph, reconstruction);
}

bool RotationEstimator::SolveRotationAveragingWithCeres(
    const PoseGraph& pose_graph,
    const FlatHashSet<image_t>& active_image_ids,
    Reconstruction& reconstruction) {
  THROW_CHECK_NOTNULL(options_.ceres);

  const std::optional<PoseGraph> filtered_pose_graph =
      MaybeFilterPoseGraphToActiveImages(pose_graph, active_image_ids);
  const PoseGraph& active_pose_graph =
      filtered_pose_graph.has_value() ? *filtered_pose_graph : pose_graph;

  const auto solve = [&](const RotationEstimatorOptions& options,
                         const CeresRotationAverager::PoseGraphInfo& graph_info,
                         CeresRotationAverager::PreparedWeights weights) {
    CeresRotationAverager averager(options,
                                   active_pose_graph,
                                   graph_info,
                                   std::move(weights),
                                   reconstruction);
    const ceres::Solver::Summary summary = averager.Solve();
    if (VLOG_IS_ON(2)) {
      LOG(INFO) << summary.FullReport();
    } else {
      VLOG(1) << summary.BriefReport();
    }
    return summary.IsSolutionUsable();
  };

  const CeresRotationAverager::PoseGraphInfo graph_info =
      CeresRotationAverager::CheckPoseGraph(
          options_, active_pose_graph, reconstruction);

  std::optional<RotationEstimatorOptions> warm_start_options;
  std::optional<CeresRotationAverager::PreparedWeights> warm_start_weights;
  if (options_.ceres->max_num_warm_start_iterations > 0) {
    warm_start_options = options_;
    warm_start_options->ceres->loss_function_type =
        CeresLossFunctionType::HUBER;
    warm_start_options->ceres->solver_options.max_num_iterations =
        options_.ceres->max_num_warm_start_iterations;
    if (options_.reweighting == RotationAveragingReweighting::COVARIANCE) {
      // Warm-start on angular residuals: with whitening, confidently wrong
      // relative rotations (e.g., from symmetric structures) dominate the
      // early iterations.
      warm_start_options->reweighting = RotationAveragingReweighting::UNIFORM;
    }
    warm_start_weights = CeresRotationAverager::PrepareWeights(
        *warm_start_options, active_pose_graph);
  }
  CeresRotationAverager::PreparedWeights weights =
      CeresRotationAverager::PrepareWeights(options_, active_pose_graph);

  // Initialize once after validating both stages, since the warm-start
  // overrides the reweighting, which determines how the maximum spanning tree
  // weights edges.
  CeresRotationAverager::InitializeRotations(options_,
                                             active_pose_graph,
                                             graph_info.component_image_ids,
                                             reconstruction);

  if (warm_start_options.has_value() &&
      !solve(*warm_start_options, graph_info, std::move(*warm_start_weights))) {
    return false;
  }

  return solve(options_, graph_info, std::move(weights));
}

}  // namespace colmap

// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/rotation_averaging_ceres.h"

#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/motion_averaging.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"

#include <algorithm>
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

}  // namespace

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
  const CeresRotationAveragerOptions& ceres_options = *options.ceres;
  solver_options.num_threads =
      GetEffectiveNumThreads(solver_options.num_threads);
  if (VLOG_IS_ON(2)) {
    solver_options.minimizer_progress_to_stdout = true;
  }
  std::shared_ptr<ceres::LossFunction> loss(CreateCeresLossFunction(
      ceres_options.loss_function_type, ceres_options.loss_function_scale));
  FlatHashSet<image_t> image_ids;
  int max_num_matches = 0;
  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (!reconstruction.ExistsImage(image_id1) ||
        !reconstruction.ExistsImage(image_id2)) {
      throw std::invalid_argument("rotation edge references unknown image");
    }
    image_ids.insert(image_id1);
    image_ids.insert(image_id2);
    max_num_matches = std::max(max_num_matches, edge.num_matches);
  }
  if (image_ids.empty()) {
    throw std::invalid_argument("Ceres rotation averaging requires edges");
  }
  const auto frame_components = pose_graph.ConnectedFrameComponents(
      reconstruction, /*filter_unregistered=*/false);
  if (frame_components.size() != 1) {
    throw std::invalid_argument(
        "Ceres rotation averaging requires a connected pose graph");
  }

  InitializeRotations(options, pose_graph, image_ids);
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  SetupParameterBlocks(options, image_ids);

  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    if (options.reweighting ==
            RotationAveragingReweighting::INLIER_MATCH_COUNT &&
        max_num_matches > 0) {
      if (edge.num_matches == 0) continue;
      loss =
          CreateCeresLossFunction(ceres_options.loss_function_type,
                                  ceres_options.loss_function_scale,
                                  double(edge.num_matches) / max_num_matches);
    }
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    AddRelativeRotationResidual(
        image_id1, image_id2, edge.cam2_from_cam1.rotation(), loss);
  }
}

void CeresRotationAverager::InitializeRotations(
    const RotationEstimatorOptions& options,
    const PoseGraph& pose_graph,
    const FlatHashSet<image_t>& image_ids) {
  // Validate fixed sensors before replacing invalid initial rotations.
  for (const image_t image_id : image_ids) {
    const auto& image = reconstruction_.Image(image_id);
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
  for (const image_t image_id : image_ids) {
    const auto& image = reconstruction_.Image(image_id);
    if (image.IsRefInFrame()) continue;
    auto& pose = image.FramePtr()->RigPtr()->MaybeSensorFromRig(
        image.CameraPtr()->SensorId());
    if (pose.has_value() && !pose->rotation().coeffs().allFinite()) {
      pose.reset();
    }
  }
  if (!options.skip_initialization) {
    InitializeFromMaximumSpanningTree(
        pose_graph, image_ids, reconstruction_, options.refine_sensor_from_rig);
  }
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
    const std::shared_ptr<ceres::LossFunction>& loss_function) {
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
  if (sensor1_from_rig_rot_ptr == nullptr &&
      sensor2_from_rig_rot_ptr == nullptr && !same_frame) {
    problem_->AddResidualBlock(
        RelativeRotationCostFunctor::Create(cam2_from_cam1),
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
  problem_->AddResidualBlock(cost_function, loss, parameter_blocks);
}

std::unique_ptr<CeresRotationAverager> CreateDefaultCeresRotationAverager(
    const RotationEstimatorOptions& options,
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction) {
  return std::make_unique<CeresRotationAverager>(
      options, pose_graph, reconstruction);
}

}  // namespace colmap

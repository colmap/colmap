// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/rotation_averaging_ceres.h"

#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/quaternion_utils.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace colmap {
namespace {

struct RelativeRotationError {
  template <typename T>
  bool operator()(const T* const rotation1,
                  const T* const rotation2,
                  T* residuals) const {
    const T* parameters[] = {rotation1, rotation2};
    return (*this)(parameters, residuals);
  }

  static ceres::CostFunction* Create(const Eigen::Quaterniond& cam2_from_cam1) {
    return new ceres::AutoDiffCostFunction<RelativeRotationError, 3, 4, 4>(
        new RelativeRotationError{cam2_from_cam1});
  }

  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    Eigen::Quaternion<T> error = measurement.cast<T>();
    if (sensor2_index >= 0) {
      error =
          EigenQuaternionMap<T>(parameters[sensor2_index]).conjugate() * error;
    }
    if (sensor1_index >= 0) {
      error = error * EigenQuaternionMap<T>(parameters[sensor1_index]);
    }
    // The shared frame rotation cancels from the angular cost.
    if (!same_frame) {
      error = EigenQuaternionMap<T>(parameters[1]).conjugate() * error *
              EigenQuaternionMap<T>(parameters[0]);
    }
    AngleAxisFromEigenQuaternion(error.coeffs().data(), residuals);
    return true;
  }

  Eigen::Quaterniond measurement;
  int sensor1_index = -1;
  int sensor2_index = -1;
  bool same_frame = false;
};

}  // namespace

CeresRotationAveragerOptions::CeresRotationAveragerOptions() {
  solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
}

CeresRotationAverager::CeresRotationAverager(
    const CeresRotationAveragerOptions& options,
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction)
    : solver_options_(options.solver_options), reconstruction_(reconstruction) {
  std::shared_ptr<ceres::LossFunction> loss(CreateCeresLossFunction(
      options.loss_function_type, options.loss_function_scale));
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

  if (!options.refine_sensor_from_rig) {
    for (const image_t image_id : image_ids) {
      const auto& image = reconstruction.Image(image_id);
      if (image.IsRefInFrame()) continue;
      THROW_CHECK(image.FramePtr()->RigPtr()->HasSensorFromRig(
          image.CameraPtr()->SensorId()))
          << "rotation averaging requires sensor rotations";
    }
  }
  if (!options.skip_initialization) {
    InitializeFromMaximumSpanningTree(
        pose_graph, image_ids, reconstruction, options.refine_sensor_from_rig);
  }
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  const auto add_rotation = [&](Eigen::Map<Eigen::Quaterniond> rotation) {
    double* block = rotation.coeffs().data();
    if (!problem_->HasParameterBlock(block)) {
      problem_->AddParameterBlock(
          block, 4, CreateEigenQuaternionManifold().release());
    }
    return block;
  };
  for (const image_t image_id : image_ids) {
    const Image& image = reconstruction.Image(image_id);
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
      if (!pose->rotation().coeffs().allFinite()) {
        throw std::invalid_argument(
            "rotation averaging requires sensor rotations");
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
      reconstruction.Frame(reconstruction.Image(root_image_id).FrameId());
  problem_->SetParameterBlockConstant(
      root_frame.RigFromWorld().rotation().coeffs().data());

  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    if (options.reweighting ==
            RotationAveragingReweighting::INLIER_MATCH_COUNT &&
        max_num_matches > 0) {
      if (edge.num_matches == 0) continue;
      loss =
          CreateCeresLossFunction(options.loss_function_type,
                                  options.loss_function_scale,
                                  double(edge.num_matches) / max_num_matches);
    }
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    AddRelativeRotationResidual(
        image_id1, image_id2, edge.cam2_from_cam1.rotation(), loss);
  }
}

ceres::Solver::Summary CeresRotationAverager::Solve() {
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options_, problem_.get(), &summary);
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
const ceres::Solver::Options& CeresRotationAverager::SolverOptions() const {
  return solver_options_;
}

void CeresRotationAverager::AddRelativeRotationResidual(
    const image_t image_id1,
    const image_t image_id2,
    const Eigen::Quaterniond& cam2_from_cam1,
    std::shared_ptr<ceres::LossFunction> loss_function) {
  const Image& image1 = reconstruction_.Image(image_id1);
  const Image& image2 = reconstruction_.Image(image_id2);
  Frame& frame1 = *image1.FramePtr();
  Frame& frame2 = *image2.FramePtr();
  double* rotation1 = frame1.RigFromWorld().rotation().coeffs().data();
  double* rotation2 = frame2.RigFromWorld().rotation().coeffs().data();
  THROW_CHECK(problem_->HasParameterBlock(rotation1));
  THROW_CHECK(problem_->HasParameterBlock(rotation2));
  const auto sensor_block = [](const Image& image) -> double* {
    if (image.IsRefInFrame()) return nullptr;
    const sensor_t sensor_id = image.CameraPtr()->SensorId();
    Rigid3d& sensor_from_rig =
        image.FramePtr()->RigPtr()->SensorFromRig(sensor_id);
    return sensor_from_rig.rotation().coeffs().data();
  };
  double* sensor1 = sensor_block(image1);
  double* sensor2 = sensor_block(image2);
  THROW_CHECK(sensor1 == nullptr || problem_->HasParameterBlock(sensor1));
  THROW_CHECK(sensor2 == nullptr || problem_->HasParameterBlock(sensor2));
  const bool same_frame = image1.FrameId() == image2.FrameId();
  if (same_frame && sensor1 == sensor2) {
    LOG(WARNING) << "Skipping self-loop for image pair " << image_id1 << ", "
                 << image_id2;
    return;
  }
  ceres::LossFunction* loss = loss_function.get();
  if (loss != nullptr) {
    losses_.try_emplace(loss, std::move(loss_function));
  }
  if (sensor1 == nullptr && sensor2 == nullptr && !same_frame) {
    problem_->AddResidualBlock(RelativeRotationError::Create(cam2_from_cam1),
                               loss,
                               rotation1,
                               rotation2);
    return;
  }
  std::vector<double*> blocks;
  if (!same_frame) blocks = {rotation1, rotation2};
  const auto sensor_index = [&](double* sensor) {
    if (sensor == nullptr) return -1;
    const auto it = std::find(blocks.begin(), blocks.end(), sensor);
    const int index = static_cast<int>(it - blocks.begin());
    if (it == blocks.end()) blocks.push_back(sensor);
    return index;
  };
  auto* cost = new ceres::DynamicAutoDiffCostFunction<RelativeRotationError, 8>(
      new RelativeRotationError{cam2_from_cam1,
                                sensor_index(sensor1),
                                sensor_index(sensor2),
                                same_frame});
  for (size_t i = 0; i < blocks.size(); ++i) cost->AddParameterBlock(4);
  cost->SetNumResiduals(3);
  problem_->AddResidualBlock(cost, loss, blocks);
}

std::unique_ptr<CeresRotationAverager> CreateDefaultCeresRotationAverager(
    const CeresRotationAveragerOptions& options,
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction) {
  return std::make_unique<CeresRotationAverager>(
      options, pose_graph, reconstruction);
}

}  // namespace colmap

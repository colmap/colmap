#include "colmap/estimators/rotation_averaging_ceres.h"

#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/quaternion_utils.h"
#include "colmap/estimators/rotation_averaging.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include <ceres/rotation.h>

namespace colmap {
namespace {

struct RelativeRotationError {
  explicit RelativeRotationError(const Eigen::Quaterniond& cam2_from_cam1)
      : cam2_from_cam1(cam2_from_cam1) {}

  template <typename T>
  bool operator()(const T* const rotation1,
                  const T* const rotation2,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 3> error =
        EigenQuaternionMap<T>(rotation2).toRotationMatrix().transpose() *
        cam2_from_cam1.cast<T>().toRotationMatrix() *
        EigenQuaternionMap<T>(rotation1).toRotationMatrix();
    ceres::RotationMatrixToAngleAxis(error.data(), residuals);
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Quaterniond& cam2_from_cam1) {
    return new ceres::AutoDiffCostFunction<RelativeRotationError, 3, 4, 4>(
        new RelativeRotationError(cam2_from_cam1));
  }

  const Eigen::Quaterniond cam2_from_cam1;
};

}  // namespace

CeresRotationAveragerOptions::CeresRotationAveragerOptions() {
  solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
}

CeresRotationAverager::CeresRotationAverager(
    std::unique_ptr<ceres::Problem> problem,
    ceres::Solver::Options solver_options,
    Reconstruction& reconstruction)
    : solver_options_(std::move(solver_options)),
      reconstruction_(reconstruction),
      problem_(std::move(problem)) {}

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
  Frame& frame1 =
      reconstruction_.Frame(reconstruction_.Image(image_id1).FrameId());
  Frame& frame2 =
      reconstruction_.Frame(reconstruction_.Image(image_id2).FrameId());
  double* rotation1 = frame1.RigFromWorld().rotation().coeffs().data();
  double* rotation2 = frame2.RigFromWorld().rotation().coeffs().data();
  if (!problem_->HasParameterBlock(rotation1) ||
      !problem_->HasParameterBlock(rotation2)) {
    throw std::invalid_argument(
        "relative rotation residual references an image outside the Ceres "
        "problem");
  }
  ceres::LossFunction* loss = loss_function.get();
  if (loss != nullptr) {
    losses_.try_emplace(loss, std::move(loss_function));
  }
  problem_->AddResidualBlock(RelativeRotationError::Create(cam2_from_cam1),
                             loss,
                             rotation1,
                             rotation2);
}

std::unique_ptr<CeresRotationAverager> CreateDefaultCeresRotationAverager(
    const CeresRotationAveragerOptions& options,
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction) {
  std::shared_ptr<ceres::LossFunction> loss(CreateCeresLossFunction(
      options.loss_function_type, options.loss_function_scale));
  FlatHashSet<image_t> image_ids;
  std::vector<image_pair_t> pair_ids;
  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (!reconstruction.ExistsImage(image_id1) ||
        !reconstruction.ExistsImage(image_id2)) {
      throw std::invalid_argument("rotation edge references unknown image");
    }
    const Image& image1 = reconstruction.Image(image_id1);
    const Image& image2 = reconstruction.Image(image_id2);
    if (image1.FrameId() == image2.FrameId() || !image1.IsRefInFrame() ||
        !image2.IsRefInFrame() || image1.FramePtr()->NumDataIds() != 1 ||
        image2.FramePtr()->NumDataIds() != 1) {
      throw std::invalid_argument(
          "Ceres rotation averaging does not support multi-camera rigs");
    }
    image_ids.insert(image_id1);
    image_ids.insert(image_id2);
    pair_ids.push_back(pair_id);
  }
  if (pair_ids.empty()) {
    throw std::invalid_argument("Ceres rotation averaging requires edges");
  }
  if (pose_graph
          .ConnectedFrameComponents(reconstruction,
                                    /*filter_unregistered=*/false)
          .size() != 1) {
    throw std::invalid_argument(
        "Ceres rotation averaging requires a connected pose graph");
  }
  std::sort(pair_ids.begin(), pair_ids.end());

  if (!options.skip_initialization) {
    for (const auto& [image_id, cam_from_world] :
         ComputeImageRotationsFromMaximumSpanningTree(pose_graph, image_ids)) {
      Frame& frame =
          reconstruction.Frame(reconstruction.Image(image_id).FrameId());
      if (!frame.HasPose()) {
        Rigid3d rig_from_world;
        rig_from_world.translation().setConstant(
            std::numeric_limits<double>::quiet_NaN());
        frame.SetRigFromWorld(rig_from_world);
      }
      frame.RigFromWorld().rotation() = cam_from_world.rotation();
    }
  }

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  auto problem = std::make_unique<ceres::Problem>(problem_options);
  for (const image_t image_id : image_ids) {
    Frame& frame =
        reconstruction.Frame(reconstruction.Image(image_id).FrameId());
    if (!frame.HasPose()) {
      throw std::invalid_argument(
          "rotation averaging frame has no initial pose");
    }
    double* rotation = frame.RigFromWorld().rotation().coeffs().data();
    problem->AddParameterBlock(
        rotation, 4, CreateEigenQuaternionManifold().release());
  }
  const image_t root_image_id =
      *std::min_element(image_ids.begin(), image_ids.end());
  Frame& root_frame =
      reconstruction.Frame(reconstruction.Image(root_image_id).FrameId());
  problem->SetParameterBlockConstant(
      root_frame.RigFromWorld().rotation().coeffs().data());

  std::unique_ptr<CeresRotationAverager> owner(new CeresRotationAverager(
      std::move(problem), options.solver_options, reconstruction));
  for (const image_pair_t pair_id : pair_ids) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    owner->AddRelativeRotationResidual(
        image_id1,
        image_id2,
        pose_graph.Edges().at(pair_id).cam2_from_cam1.rotation(),
        loss);
  }
  return owner;
}

}  // namespace colmap

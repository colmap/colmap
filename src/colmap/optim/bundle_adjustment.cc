// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "colmap/optim/bundle_adjustment.h"

#include <iomanip>

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#include "colmap/base/cost_functions.h"
#include "colmap/camera/models.h"
#include "colmap/geometry/projection.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"
#include "colmap/util/timer.h"

namespace colmap {

////////////////////////////////////////////////////////////////////////////////
// BundleAdjustmentOptions
////////////////////////////////////////////////////////////////////////////////

ceres::LossFunction* BundleAdjustmentOptions::CreateLossFunction() const {
  ceres::LossFunction* loss_function = nullptr;
  switch (loss_function_type) {
    case LossFunctionType::TRIVIAL:
      loss_function = new ceres::TrivialLoss();
      break;
    case LossFunctionType::SOFT_L1:
      loss_function = new ceres::SoftLOneLoss(loss_function_scale);
      break;
    case LossFunctionType::CAUCHY:
      loss_function = new ceres::CauchyLoss(loss_function_scale);
      break;
  }
  CHECK_NOTNULL(loss_function);
  return loss_function;
}

bool BundleAdjustmentOptions::Check() const {
  CHECK_OPTION_GE(loss_function_scale, 0);
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// BundleAdjustmentConfig
////////////////////////////////////////////////////////////////////////////////

BundleAdjustmentConfig::BundleAdjustmentConfig() {}

size_t BundleAdjustmentConfig::NumImages() const { return image_ids_.size(); }

size_t BundleAdjustmentConfig::NumPoints() const {
  return variable_point3D_ids_.size() + constant_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantCamIntrinsics() const {
  return constant_intrinsics_.size();
}

size_t BundleAdjustmentConfig::NumConstantCamPoses() const {
  return constant_cam_poses_.size();
}

size_t BundleAdjustmentConfig::NumConstantCamPositionss() const {
  return constant_cam_positions_.size();
}

size_t BundleAdjustmentConfig::NumVariablePoints() const {
  return variable_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantPoints() const {
  return constant_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumResiduals(
    const Reconstruction& reconstruction) const {
  // Count the number of observations for all added images.
  size_t num_observations = 0;
  for (const image_t image_id : image_ids_) {
    num_observations += reconstruction.Image(image_id).NumPoints3D();
  }

  // Count the number of observations for all added 3D points that are not
  // already added as part of the images above.

  auto NumObservationsForPoint = [this,
                                  &reconstruction](const point3D_t point3D_id) {
    size_t num_observations_for_point = 0;
    const auto& point3D = reconstruction.Point3D(point3D_id);
    for (const auto& track_el : point3D.Track().Elements()) {
      if (image_ids_.count(track_el.image_id) == 0) {
        num_observations_for_point += 1;
      }
    }
    return num_observations_for_point;
  };

  for (const auto point3D_id : variable_point3D_ids_) {
    num_observations += NumObservationsForPoint(point3D_id);
  }
  for (const auto point3D_id : constant_point3D_ids_) {
    num_observations += NumObservationsForPoint(point3D_id);
  }

  return 2 * num_observations;
}

void BundleAdjustmentConfig::AddImage(const image_t image_id) {
  image_ids_.insert(image_id);
}

bool BundleAdjustmentConfig::HasImage(const image_t image_id) const {
  return image_ids_.find(image_id) != image_ids_.end();
}

void BundleAdjustmentConfig::RemoveImage(const image_t image_id) {
  image_ids_.erase(image_id);
}

void BundleAdjustmentConfig::SetConstantCamIntrinsics(
    const camera_t camera_id) {
  constant_intrinsics_.insert(camera_id);
}

void BundleAdjustmentConfig::SetVariableCamIntrinsics(
    const camera_t camera_id) {
  constant_intrinsics_.erase(camera_id);
}

bool BundleAdjustmentConfig::HasConstantCamIntrinsics(
    const camera_t camera_id) const {
  return constant_intrinsics_.find(camera_id) != constant_intrinsics_.end();
}

void BundleAdjustmentConfig::SetConstantCamPose(const image_t image_id) {
  CHECK(HasImage(image_id));
  CHECK(!HasConstantCamPositions(image_id));
  constant_cam_poses_.insert(image_id);
}

void BundleAdjustmentConfig::SetVariableCamPose(const image_t image_id) {
  constant_cam_poses_.erase(image_id);
}

bool BundleAdjustmentConfig::HasConstantCamPose(const image_t image_id) const {
  return constant_cam_poses_.find(image_id) != constant_cam_poses_.end();
}

void BundleAdjustmentConfig::SetConstantCamPositions(
    const image_t image_id, const std::vector<int>& idxs) {
  CHECK_GT(idxs.size(), 0);
  CHECK_LE(idxs.size(), 3);
  CHECK(HasImage(image_id));
  CHECK(!HasConstantCamPose(image_id));
  CHECK(!VectorContainsDuplicateValues(idxs))
      << "Tvec indices must not contain duplicates";
  constant_cam_positions_.emplace(image_id, idxs);
}

void BundleAdjustmentConfig::RemoveConstantCamPositions(
    const image_t image_id) {
  constant_cam_positions_.erase(image_id);
}

bool BundleAdjustmentConfig::HasConstantCamPositions(
    const image_t image_id) const {
  return constant_cam_positions_.find(image_id) !=
         constant_cam_positions_.end();
}

const std::unordered_set<image_t>& BundleAdjustmentConfig::Images() const {
  return image_ids_;
}

const std::unordered_set<point3D_t>& BundleAdjustmentConfig::VariablePoints()
    const {
  return variable_point3D_ids_;
}

const std::unordered_set<point3D_t>& BundleAdjustmentConfig::ConstantPoints()
    const {
  return constant_point3D_ids_;
}

const std::vector<int>& BundleAdjustmentConfig::ConstantCamPositions(
    const image_t image_id) const {
  return constant_cam_positions_.at(image_id);
}

void BundleAdjustmentConfig::AddVariablePoint(const point3D_t point3D_id) {
  CHECK(!HasConstantPoint(point3D_id));
  variable_point3D_ids_.insert(point3D_id);
}

void BundleAdjustmentConfig::AddConstantPoint(const point3D_t point3D_id) {
  CHECK(!HasVariablePoint(point3D_id));
  constant_point3D_ids_.insert(point3D_id);
}

bool BundleAdjustmentConfig::HasPoint(const point3D_t point3D_id) const {
  return HasVariablePoint(point3D_id) || HasConstantPoint(point3D_id);
}

bool BundleAdjustmentConfig::HasVariablePoint(
    const point3D_t point3D_id) const {
  return variable_point3D_ids_.find(point3D_id) != variable_point3D_ids_.end();
}

bool BundleAdjustmentConfig::HasConstantPoint(
    const point3D_t point3D_id) const {
  return constant_point3D_ids_.find(point3D_id) != constant_point3D_ids_.end();
}

void BundleAdjustmentConfig::RemoveVariablePoint(const point3D_t point3D_id) {
  variable_point3D_ids_.erase(point3D_id);
}

void BundleAdjustmentConfig::RemoveConstantPoint(const point3D_t point3D_id) {
  constant_point3D_ids_.erase(point3D_id);
}

////////////////////////////////////////////////////////////////////////////////
// BundleAdjuster
////////////////////////////////////////////////////////////////////////////////

BundleAdjuster::BundleAdjuster(const BundleAdjustmentOptions& options,
                               const BundleAdjustmentConfig& config)
    : options_(options), config_(config) {
  CHECK(options_.Check());
}

bool BundleAdjuster::Solve(Reconstruction* reconstruction) {
  CHECK_NOTNULL(reconstruction);
  CHECK(!problem_) << "Cannot use the same BundleAdjuster multiple times";

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);

  const auto loss_function =
      std::unique_ptr<ceres::LossFunction>(options_.CreateLossFunction());
  SetUp(reconstruction, loss_function.get());

  if (problem_->NumResiduals() == 0) {
    return false;
  }

  ceres::Solver::Options solver_options = options_.solver_options;
  const bool has_sparse =
      solver_options.sparse_linear_algebra_library_type != ceres::NO_SPARSE;

  // Empirical choice.
  const size_t kMaxNumImagesDirectDenseSolver = 50;
  const size_t kMaxNumImagesDirectSparseSolver = 1000;
  const size_t num_images = config_.NumImages();
  if (num_images <= kMaxNumImagesDirectDenseSolver) {
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;
  } else if (num_images <= kMaxNumImagesDirectSparseSolver && has_sparse) {
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  } else {  // Indirect sparse (preconditioned CG) solver.
    solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
  }

  if (problem_->NumResiduals() <
      options_.min_num_residuals_for_multi_threading) {
    solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR
  } else {
    solver_options.num_threads =
        GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads =
        GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR
  }

  std::string solver_error;
  CHECK(solver_options.IsValid(&solver_error)) << solver_error;

  ceres::Solve(solver_options, problem_.get(), &summary_);

  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (options_.print_summary) {
    PrintHeading2("Bundle adjustment report");
    PrintSolverSummary(summary_);
  }

  TearDown(reconstruction);

  return true;
}

const ceres::Solver::Summary& BundleAdjuster::Summary() const {
  return summary_;
}

void BundleAdjuster::SetUp(Reconstruction* reconstruction,
                           ceres::LossFunction* loss_function) {
  // Warning: AddPointsToProblem assumes that AddImageToProblem is called first.
  // Do not change order of instructions!
  for (const image_t image_id : config_.Images()) {
    AddImageToProblem(image_id, reconstruction, loss_function);
  }
  for (const auto point3D_id : config_.VariablePoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }
  for (const auto point3D_id : config_.ConstantPoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  ParameterizeCameras(reconstruction);
  ParameterizePoints(reconstruction);
}

void BundleAdjuster::TearDown(Reconstruction*) {
  // Nothing to do
}

void BundleAdjuster::AddImageToProblem(const image_t image_id,
                                       Reconstruction* reconstruction,
                                       ceres::LossFunction* loss_function) {
  Image& image = reconstruction->Image(image_id);
  Camera& camera = reconstruction->Camera(image.CameraId());

  // CostFunction assumes unit quaternions.
  image.CamFromWorld().rotation.normalize();

  double* cam_from_world_rotation =
      image.CamFromWorld().rotation.coeffs().data();
  double* cam_from_world_translation = image.CamFromWorld().translation.data();
  double* camera_params = camera.ParamsData();

  const bool constant_cam_pose =
      !options_.refine_extrinsics || config_.HasConstantCamPose(image_id);

  // Add residuals to bundle adjustment problem.
  size_t num_observations = 0;
  for (const Point2D& point2D : image.Points2D()) {
    if (!point2D.HasPoint3D()) {
      continue;
    }

    num_observations += 1;
    point3D_num_observations_[point2D.point3D_id] += 1;

    Point3D& point3D = reconstruction->Point3D(point2D.point3D_id);
    assert(point3D.Track().Length() > 1);

    ceres::CostFunction* cost_function = nullptr;

    if (constant_cam_pose) {
      switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::kModelId:                                          \
    cost_function =                                                    \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
            image.CamFromWorld(), point2D.xy);                         \
    break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
      }

      problem_->AddResidualBlock(
          cost_function, loss_function, point3D.XYZ().data(), camera_params);
    } else {
      switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::kModelId:                                          \
    cost_function =                                                    \
        BundleAdjustmentCostFunction<CameraModel>::Create(point2D.xy); \
    break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
      }

      problem_->AddResidualBlock(cost_function,
                                 loss_function,
                                 cam_from_world_rotation,
                                 cam_from_world_translation,
                                 point3D.XYZ().data(),
                                 camera_params);
    }
  }

  if (num_observations > 0) {
    camera_ids_.insert(image.CameraId());

    // Set pose parameterization.
    if (!constant_cam_pose) {
      SetQuaternionManifold(problem_.get(), cam_from_world_rotation);
      if (config_.HasConstantCamPositions(image_id)) {
        const std::vector<int>& constant_position_idxs =
            config_.ConstantCamPositions(image_id);
        SetSubsetManifold(3,
                          constant_position_idxs,
                          problem_.get(),
                          cam_from_world_translation);
      }
    }
  }
}

void BundleAdjuster::AddPointToProblem(const point3D_t point3D_id,
                                       Reconstruction* reconstruction,
                                       ceres::LossFunction* loss_function) {
  Point3D& point3D = reconstruction->Point3D(point3D_id);

  // Is 3D point already fully contained in the problem? I.e. its entire track
  // is contained in `variable_image_ids`, `constant_image_ids`,
  // `constant_x_image_ids`.
  if (point3D_num_observations_[point3D_id] == point3D.Track().Length()) {
    return;
  }

  for (const auto& track_el : point3D.Track().Elements()) {
    // Skip observations that were already added in `FillImages`.
    if (config_.HasImage(track_el.image_id)) {
      continue;
    }

    point3D_num_observations_[point3D_id] += 1;

    Image& image = reconstruction->Image(track_el.image_id);
    Camera& camera = reconstruction->Camera(image.CameraId());
    const Point2D& point2D = image.Point2D(track_el.point2D_idx);

    // CostFunction assumes unit quaternions.
    image.CamFromWorld().rotation.normalize();

    // We do not want to refine the camera of images that are not
    // part of `constant_image_ids_`, `constant_image_ids_`,
    // `constant_x_image_ids_`.
    if (camera_ids_.count(image.CameraId()) == 0) {
      camera_ids_.insert(image.CameraId());
      config_.SetConstantCamIntrinsics(image.CameraId());
    }

    ceres::CostFunction* cost_function = nullptr;

    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::kModelId:                                          \
    cost_function =                                                    \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
            image.CamFromWorld(), point2D.xy);                         \
    break;

      CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }
    problem_->AddResidualBlock(cost_function,
                               loss_function,
                               point3D.XYZ().data(),
                               camera.ParamsData());
  }
}

void BundleAdjuster::ParameterizeCameras(Reconstruction* reconstruction) {
  const bool constant_camera = !options_.refine_focal_length &&
                               !options_.refine_principal_point &&
                               !options_.refine_extra_params;
  for (const camera_t camera_id : camera_ids_) {
    Camera& camera = reconstruction->Camera(camera_id);

    if (constant_camera || config_.HasConstantCamIntrinsics(camera_id)) {
      problem_->SetParameterBlockConstant(camera.ParamsData());
      continue;
    } else {
      std::vector<int> const_camera_params;

      if (!options_.refine_focal_length) {
        const std::vector<size_t>& params_idxs = camera.FocalLengthIdxs();
        const_camera_params.insert(
            const_camera_params.end(), params_idxs.begin(), params_idxs.end());
      }
      if (!options_.refine_principal_point) {
        const std::vector<size_t>& params_idxs = camera.PrincipalPointIdxs();
        const_camera_params.insert(
            const_camera_params.end(), params_idxs.begin(), params_idxs.end());
      }
      if (!options_.refine_extra_params) {
        const std::vector<size_t>& params_idxs = camera.ExtraParamsIdxs();
        const_camera_params.insert(
            const_camera_params.end(), params_idxs.begin(), params_idxs.end());
      }

      if (const_camera_params.size() > 0) {
        SetSubsetManifold(static_cast<int>(camera.NumParams()),
                          const_camera_params,
                          problem_.get(),
                          camera.ParamsData());
      }
    }
  }
}

void BundleAdjuster::ParameterizePoints(Reconstruction* reconstruction) {
  for (const auto elem : point3D_num_observations_) {
    Point3D& point3D = reconstruction->Point3D(elem.first);
    if (point3D.Track().Length() > elem.second) {
      problem_->SetParameterBlockConstant(point3D.XYZ().data());
    }
  }

  for (const point3D_t point3D_id : config_.ConstantPoints()) {
    Point3D& point3D = reconstruction->Point3D(point3D_id);
    problem_->SetParameterBlockConstant(point3D.XYZ().data());
  }
}

////////////////////////////////////////////////////////////////////////////////
// RigBundleAdjuster
////////////////////////////////////////////////////////////////////////////////

RigBundleAdjuster::RigBundleAdjuster(const BundleAdjustmentOptions& options,
                                     const Options& rig_options,
                                     const BundleAdjustmentConfig& config)
    : BundleAdjuster(options, config), rig_options_(rig_options) {}

bool RigBundleAdjuster::Solve(Reconstruction* reconstruction,
                              std::vector<CameraRig>* camera_rigs) {
  CHECK_NOTNULL(reconstruction);
  CHECK_NOTNULL(camera_rigs);
  CHECK(!problem_) << "Cannot use the same BundleAdjuster multiple times";

  // Check the validity of the provided camera rigs.
  std::unordered_set<camera_t> rig_camera_ids;
  for (auto& camera_rig : *camera_rigs) {
    camera_rig.Check(*reconstruction);
    for (const auto& camera_id : camera_rig.GetCameraIds()) {
      CHECK_EQ(rig_camera_ids.count(camera_id), 0)
          << "Camera must not be part of multiple camera rigs";
      rig_camera_ids.insert(camera_id);
    }

    for (const auto& snapshot : camera_rig.Snapshots()) {
      for (const auto& image_id : snapshot) {
        CHECK_EQ(image_id_to_camera_rig_.count(image_id), 0)
            << "Image must not be part of multiple camera rigs";
        image_id_to_camera_rig_.emplace(image_id, &camera_rig);
      }
    }
  }

  problem_ = std::make_unique<ceres::Problem>();

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);

  const auto loss_function =
      std::unique_ptr<ceres::LossFunction>(options_.CreateLossFunction());
  SetUp(reconstruction, camera_rigs, loss_function.get());

  if (problem_->NumResiduals() == 0) {
    return false;
  }

  ceres::Solver::Options solver_options = options_.solver_options;
  const bool has_sparse =
      solver_options.sparse_linear_algebra_library_type != ceres::NO_SPARSE;

  // Empirical choice.
  const size_t kMaxNumImagesDirectDenseSolver = 50;
  const size_t kMaxNumImagesDirectSparseSolver = 1000;
  const size_t num_images = config_.NumImages();
  if (num_images <= kMaxNumImagesDirectDenseSolver) {
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;
  } else if (num_images <= kMaxNumImagesDirectSparseSolver && has_sparse) {
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  } else {  // Indirect sparse (preconditioned CG) solver.
    solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
  }

  solver_options.num_threads =
      GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR

  std::string solver_error;
  CHECK(solver_options.IsValid(&solver_error)) << solver_error;

  ceres::Solve(solver_options, problem_.get(), &summary_);

  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (options_.print_summary) {
    PrintHeading2("Rig Bundle adjustment report");
    PrintSolverSummary(summary_);
  }

  TearDown(reconstruction, *camera_rigs);

  return true;
}

void RigBundleAdjuster::SetUp(Reconstruction* reconstruction,
                              std::vector<CameraRig>* camera_rigs,
                              ceres::LossFunction* loss_function) {
  ComputeCameraRigPoses(*reconstruction, *camera_rigs);

  for (const image_t image_id : config_.Images()) {
    AddImageToProblem(image_id, reconstruction, camera_rigs, loss_function);
  }
  for (const auto point3D_id : config_.VariablePoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }
  for (const auto point3D_id : config_.ConstantPoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  ParameterizeCameras(reconstruction);
  ParameterizePoints(reconstruction);
  ParameterizeCameraRigs(reconstruction);
}

void RigBundleAdjuster::TearDown(Reconstruction* reconstruction,
                                 const std::vector<CameraRig>& camera_rigs) {
  for (const auto& elem : image_id_to_camera_rig_) {
    const auto image_id = elem.first;
    const auto& camera_rig = *elem.second;
    auto& image = reconstruction->Image(image_id);
    image.CamFromWorld() = camera_rig.CamFromRig(image.CameraId()) *
                           (*image_id_to_rig_from_world_.at(image_id));
  }
}

void RigBundleAdjuster::AddImageToProblem(const image_t image_id,
                                          Reconstruction* reconstruction,
                                          std::vector<CameraRig>* camera_rigs,
                                          ceres::LossFunction* loss_function) {
  const double max_squared_reproj_error =
      rig_options_.max_reproj_error * rig_options_.max_reproj_error;

  Image& image = reconstruction->Image(image_id);
  Camera& camera = reconstruction->Camera(image.CameraId());

  const bool constant_cam_pose = config_.HasConstantCamPose(image_id);
  const bool constant_cam_position = config_.HasConstantCamPositions(image_id);

  double* camera_params = camera.ParamsData();
  double* cam_from_rig_rotation = nullptr;
  double* cam_from_rig_translation = nullptr;
  double* rig_from_world_rotation = nullptr;
  double* rig_from_world_translation = nullptr;
  CameraRig* camera_rig = nullptr;
  Eigen::Matrix3x4d cam_from_world_mat = Eigen::Matrix3x4d::Zero();

  if (image_id_to_camera_rig_.count(image_id) > 0) {
    CHECK(!constant_cam_pose)
        << "Images contained in a camera rig must not have constant pose";
    CHECK(!constant_cam_position)
        << "Images contained in a camera rig must not have constant tvec";
    camera_rig = image_id_to_camera_rig_.at(image_id);
    Rigid3d& rig_from_world = *image_id_to_rig_from_world_.at(image_id);
    rig_from_world_rotation = rig_from_world.rotation.coeffs().data();
    rig_from_world_translation = rig_from_world.translation.data();
    Rigid3d& cam_from_rig = camera_rig->CamFromRig(image.CameraId());
    cam_from_rig_rotation = cam_from_rig.rotation.coeffs().data();
    cam_from_rig_translation = cam_from_rig.translation.data();
    cam_from_world_mat = (cam_from_rig * rig_from_world).ToMatrix();
  } else {
    // CostFunction assumes unit quaternions.
    image.CamFromWorld().rotation.normalize();
    cam_from_rig_rotation = image.CamFromWorld().rotation.coeffs().data();
    cam_from_rig_translation = image.CamFromWorld().translation.data();
  }

  // Collect cameras for final parameterization.
  CHECK(image.HasCamera());
  camera_ids_.insert(image.CameraId());

  // The number of added observations for the current image.
  size_t num_observations = 0;

  // Add residuals to bundle adjustment problem.
  for (const Point2D& point2D : image.Points2D()) {
    if (!point2D.HasPoint3D()) {
      continue;
    }

    Point3D& point3D = reconstruction->Point3D(point2D.point3D_id);
    assert(point3D.Track().Length() > 1);

    if (camera_rig != nullptr &&
        CalculateSquaredReprojectionError(
            point2D.xy, point3D.XYZ(), cam_from_world_mat, camera) >
            max_squared_reproj_error) {
      continue;
    }

    num_observations += 1;
    point3D_num_observations_[point2D.point3D_id] += 1;

    ceres::CostFunction* cost_function = nullptr;

    if (camera_rig == nullptr) {
      if (constant_cam_pose) {
        switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::kModelId:                                          \
    cost_function =                                                    \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
            image.CamFromWorld(), point2D.xy);                         \
    break;

          CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
        }

        problem_->AddResidualBlock(
            cost_function, loss_function, point3D.XYZ().data(), camera_params);
      } else {
        switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::kModelId:                                          \
    cost_function =                                                    \
        BundleAdjustmentCostFunction<CameraModel>::Create(point2D.xy); \
    break;

          CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
        }

        problem_->AddResidualBlock(cost_function,
                                   loss_function,
                                   cam_from_rig_rotation,     // rig == world
                                   cam_from_rig_translation,  // rig == world
                                   point3D.XYZ().data(),
                                   camera_params);
      }
    } else {
      switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                    \
  case CameraModel::kModelId:                                             \
    cost_function =                                                       \
        RigBundleAdjustmentCostFunction<CameraModel>::Create(point2D.xy); \
                                                                          \
    break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
      }
      problem_->AddResidualBlock(cost_function,
                                 loss_function,
                                 cam_from_rig_rotation,
                                 cam_from_rig_translation,
                                 rig_from_world_rotation,
                                 rig_from_world_translation,
                                 point3D.XYZ().data(),
                                 camera_params);
    }
  }

  if (num_observations > 0) {
    parameterized_quats_.insert(cam_from_rig_rotation);

    if (camera_rig != nullptr) {
      parameterized_quats_.insert(rig_from_world_rotation);

      // Set the relative pose of the camera constant if relative pose
      // refinement is disabled or if it is the reference camera to avoid over-
      // parameterization of the camera pose.
      if (!rig_options_.refine_relative_poses ||
          image.CameraId() == camera_rig->RefCameraId()) {
        problem_->SetParameterBlockConstant(cam_from_rig_rotation);
        problem_->SetParameterBlockConstant(cam_from_rig_translation);
      }
    }

    // Set pose parameterization.
    if (!constant_cam_pose && constant_cam_position) {
      const std::vector<int>& constant_position_idxs =
          config_.ConstantCamPositions(image_id);
      SetSubsetManifold(
          3, constant_position_idxs, problem_.get(), cam_from_rig_translation);
    }
  }
}

void RigBundleAdjuster::AddPointToProblem(const point3D_t point3D_id,
                                          Reconstruction* reconstruction,
                                          ceres::LossFunction* loss_function) {
  Point3D& point3D = reconstruction->Point3D(point3D_id);

  // Is 3D point already fully contained in the problem? I.e. its entire track
  // is contained in `variable_image_ids`, `constant_image_ids`,
  // `constant_x_image_ids`.
  if (point3D_num_observations_[point3D_id] == point3D.Track().Length()) {
    return;
  }

  for (const auto& track_el : point3D.Track().Elements()) {
    // Skip observations that were already added in `AddImageToProblem`.
    if (config_.HasImage(track_el.image_id)) {
      continue;
    }

    point3D_num_observations_[point3D_id] += 1;

    Image& image = reconstruction->Image(track_el.image_id);
    Camera& camera = reconstruction->Camera(image.CameraId());
    const Point2D& point2D = image.Point2D(track_el.point2D_idx);

    // We do not want to refine the camera of images that are not
    // part of `constant_image_ids_`, `constant_image_ids_`,
    // `constant_x_image_ids_`.
    if (camera_ids_.count(image.CameraId()) == 0) {
      camera_ids_.insert(image.CameraId());
      config_.SetConstantCamIntrinsics(image.CameraId());
    }

    ceres::CostFunction* cost_function = nullptr;

    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::kModelId:                                          \
    cost_function =                                                    \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
            image.CamFromWorld(), point2D.xy);                         \
    problem_->AddResidualBlock(cost_function,                          \
                               loss_function,                          \
                               point3D.XYZ().data(),                   \
                               camera.ParamsData());                   \
    break;

      CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }
  }
}

void RigBundleAdjuster::ComputeCameraRigPoses(
    const Reconstruction& reconstruction,
    const std::vector<CameraRig>& camera_rigs) {
  rigs_from_world_.reserve(camera_rigs.size());
  for (const auto& camera_rig : camera_rigs) {
    rigs_from_world_.emplace_back();
    auto& rig_from_world = rigs_from_world_.back();
    const size_t num_snapshots = camera_rig.NumSnapshots();
    rig_from_world.resize(num_snapshots);
    for (size_t snapshot_idx = 0; snapshot_idx < num_snapshots;
         ++snapshot_idx) {
      rig_from_world[snapshot_idx] =
          camera_rig.ComputeRigFromWorld(snapshot_idx, reconstruction);
      for (const auto image_id : camera_rig.Snapshots()[snapshot_idx]) {
        image_id_to_rig_from_world_.emplace(image_id,
                                            &rig_from_world[snapshot_idx]);
      }
    }
  }
}

void RigBundleAdjuster::ParameterizeCameraRigs(Reconstruction* reconstruction) {
  for (double* cam_from_rig_rotation : parameterized_quats_) {
    SetQuaternionManifold(problem_.get(), cam_from_rig_rotation);
  }
}

void PrintSolverSummary(const ceres::Solver::Summary& summary) {
  std::cout << std::right << std::setw(16) << "Residuals : ";
  std::cout << std::left << summary.num_residuals_reduced << std::endl;

  std::cout << std::right << std::setw(16) << "Parameters : ";
  std::cout << std::left << summary.num_effective_parameters_reduced
            << std::endl;

  std::cout << std::right << std::setw(16) << "Iterations : ";
  std::cout << std::left
            << summary.num_successful_steps + summary.num_unsuccessful_steps
            << std::endl;

  std::cout << std::right << std::setw(16) << "Time : ";
  std::cout << std::left << summary.total_time_in_seconds << " [s]"
            << std::endl;

  std::cout << std::right << std::setw(16) << "Initial cost : ";
  std::cout << std::right << std::setprecision(6)
            << std::sqrt(summary.initial_cost / summary.num_residuals_reduced)
            << " [px]" << std::endl;

  std::cout << std::right << std::setw(16) << "Final cost : ";
  std::cout << std::right << std::setprecision(6)
            << std::sqrt(summary.final_cost / summary.num_residuals_reduced)
            << " [px]" << std::endl;

  std::cout << std::right << std::setw(16) << "Termination : ";

  std::string termination = "";

  switch (summary.termination_type) {
    case ceres::CONVERGENCE:
      termination = "Convergence";
      break;
    case ceres::NO_CONVERGENCE:
      termination = "No convergence";
      break;
    case ceres::FAILURE:
      termination = "Failure";
      break;
    case ceres::USER_SUCCESS:
      termination = "User success";
      break;
    case ceres::USER_FAILURE:
      termination = "User failure";
      break;
    default:
      termination = "Unknown";
      break;
  }

  std::cout << std::right << termination << std::endl;
  std::cout << std::endl;
}

}  // namespace colmap

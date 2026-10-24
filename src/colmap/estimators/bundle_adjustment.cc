// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#include "colmap/estimators/bundle_adjustment.h"

#include "colmap/estimators/alignment.h"
#include "colmap/estimators/cost_functions.h"
#include "colmap/estimators/manifold.h"
#include "colmap/scene/projection.h"
#include "colmap/sensor/models.h"
#include "colmap/util/cuda.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"
#include "colmap/util/timer.h"

#include <iomanip>

namespace colmap {

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

size_t BundleAdjustmentConfig::NumConstantCamPositions() const {
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
    for (const auto& track_el : point3D.track.Elements()) {
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
  THROW_CHECK(HasImage(image_id));
  THROW_CHECK(!HasConstantCamPositions(image_id));
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
  THROW_CHECK_GT(idxs.size(), 0);
  THROW_CHECK_LT(idxs.size(), 3)
      << "Set the entire parameter block as constant instead";
  THROW_CHECK(HasImage(image_id));
  THROW_CHECK(!HasConstantCamPose(image_id));
  THROW_CHECK(!VectorContainsDuplicateValues(idxs))
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

const std::unordered_set<camera_t> BundleAdjustmentConfig::ConstantIntrinsics()
    const {
  return constant_intrinsics_;
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

const std::unordered_set<image_t>& BundleAdjustmentConfig::ConstantCamPoses()
    const {
  return constant_cam_poses_;
}

const std::vector<int>& BundleAdjustmentConfig::ConstantCamPositions(
    const image_t image_id) const {
  return constant_cam_positions_.at(image_id);
}

void BundleAdjustmentConfig::AddVariablePoint(const point3D_t point3D_id) {
  THROW_CHECK(!HasConstantPoint(point3D_id));
  variable_point3D_ids_.insert(point3D_id);
}

void BundleAdjustmentConfig::AddConstantPoint(const point3D_t point3D_id) {
  THROW_CHECK(!HasVariablePoint(point3D_id));
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

BundleAdjuster::BundleAdjuster(BundleAdjustmentOptions options,
                               BundleAdjustmentConfig config)
    : options_(std::move(options)), config_(std::move(config)) {
  THROW_CHECK(options_.Check());
}

const BundleAdjustmentOptions& BundleAdjuster::Options() const {
  return options_;
}

const BundleAdjustmentConfig& BundleAdjuster::Config() const { return config_; }

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
  THROW_CHECK_NOTNULL(loss_function);
  return loss_function;
}

ceres::Solver::Options BundleAdjustmentOptions::CreateSolverOptions(
    const BundleAdjustmentConfig& config, const ceres::Problem& problem) const {
  ceres::Solver::Options custom_solver_options = solver_options;
  if (VLOG_IS_ON(2)) {
    custom_solver_options.minimizer_progress_to_stdout = true;
    custom_solver_options.logging_type =
        ceres::LoggingType::PER_MINIMIZER_ITERATION;
  }

  const int num_images = config.NumImages();
  const bool has_sparse =
      custom_solver_options.sparse_linear_algebra_library_type !=
      ceres::NO_SPARSE;

  int max_num_images_direct_dense_solver =
      max_num_images_direct_dense_cpu_solver;
  int max_num_images_direct_sparse_solver =
      max_num_images_direct_sparse_cpu_solver;

#ifdef COLMAP_CUDA_ENABLED
  bool cuda_solver_enabled = false;

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 2)) && \
    !defined(CERES_NO_CUDA)
  if (use_gpu && num_images >= min_num_images_gpu_solver) {
    cuda_solver_enabled = true;
    custom_solver_options.dense_linear_algebra_library_type = ceres::CUDA;
    max_num_images_direct_dense_solver = max_num_images_direct_dense_gpu_solver;
  }
#else
  if (use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but Ceres was "
           "compiled without CUDA support. Falling back to CPU-based dense "
           "solvers.";
  }
#endif

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 3)) && \
    !defined(CERES_NO_CUDSS)
  if (use_gpu && num_images >= min_num_images_gpu_solver) {
    cuda_solver_enabled = true;
    custom_solver_options.sparse_linear_algebra_library_type =
        ceres::CUDA_SPARSE;
    max_num_images_direct_sparse_solver =
        max_num_images_direct_sparse_gpu_solver;
  }
#else
  if (use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but Ceres was "
           "compiled without cuDSS support. Falling back to CPU-based sparse "
           "solvers.";
  }
#endif

  if (cuda_solver_enabled) {
    const std::vector<int> gpu_indices = CSVToVector<int>(gpu_index);
    THROW_CHECK_GT(gpu_indices.size(), 0);
    SetBestCudaDevice(gpu_indices[0]);
  }
#else
  if (use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but COLMAP was "
           "compiled without CUDA support. Falling back to CPU-based "
           "solvers.";
  }
#endif  // COLMAP_CUDA_ENABLED

  if (num_images <= max_num_images_direct_dense_solver) {
    custom_solver_options.linear_solver_type = ceres::DENSE_SCHUR;
  } else if (has_sparse && num_images <= max_num_images_direct_sparse_solver) {
    custom_solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  } else {  // Indirect sparse (preconditioned CG) solver.
    custom_solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    custom_solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
  }

  if (problem.NumResiduals() < min_num_residuals_for_cpu_multi_threading) {
    custom_solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
    custom_solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR
  } else {
    custom_solver_options.num_threads =
        GetEffectiveNumThreads(custom_solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
    custom_solver_options.num_linear_solver_threads =
        GetEffectiveNumThreads(custom_solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR
  }

  std::string solver_error;
  THROW_CHECK(custom_solver_options.IsValid(&solver_error)) << solver_error;
  return custom_solver_options;
}

bool BundleAdjustmentOptions::Check() const {
  CHECK_OPTION_GE(loss_function_scale, 0);
  CHECK_OPTION_LT(max_num_images_direct_dense_cpu_solver,
                  max_num_images_direct_sparse_cpu_solver);
  CHECK_OPTION_LT(max_num_images_direct_dense_gpu_solver,
                  max_num_images_direct_sparse_gpu_solver);
  return true;
}

namespace {

void ParameterizeCameras(const BundleAdjustmentOptions& options,
                         const BundleAdjustmentConfig& config,
                         const std::unordered_set<camera_t>& camera_ids,
                         Reconstruction& reconstruction,
                         ceres::Problem& problem) {
  const bool constant_camera = !options.refine_focal_length &&
                               !options.refine_principal_point &&
                               !options.refine_extra_params;
  for (const camera_t camera_id : camera_ids) {
    Camera& camera = reconstruction.Camera(camera_id);

    if (constant_camera || config.HasConstantCamIntrinsics(camera_id)) {
      problem.SetParameterBlockConstant(camera.params.data());
    } else {
      std::vector<int> const_camera_params;

      if (!options.refine_focal_length) {
        const span<const size_t> params_idxs = camera.FocalLengthIdxs();
        const_camera_params.insert(
            const_camera_params.end(), params_idxs.begin(), params_idxs.end());
      }
      if (!options.refine_principal_point) {
        const span<const size_t> params_idxs = camera.PrincipalPointIdxs();
        const_camera_params.insert(
            const_camera_params.end(), params_idxs.begin(), params_idxs.end());
      }
      if (!options.refine_extra_params) {
        const span<const size_t> params_idxs = camera.ExtraParamsIdxs();
        const_camera_params.insert(
            const_camera_params.end(), params_idxs.begin(), params_idxs.end());
      }

      if (const_camera_params.size() > 0) {
        SetSubsetManifold(static_cast<int>(camera.params.size()),
                          const_camera_params,
                          &problem,
                          camera.params.data());
      }
    }
  }
}

void ParameterizePoints(
    const BundleAdjustmentConfig& config,
    const std::unordered_map<point3D_t, size_t>& point3D_num_observations,
    Reconstruction& reconstruction,
    ceres::Problem& problem) {
  for (const auto& [point3D_id, num_observations] : point3D_num_observations) {
    Point3D& point3D = reconstruction.Point3D(point3D_id);
    if (point3D.track.Length() > num_observations) {
      problem.SetParameterBlockConstant(point3D.xyz.data());
    }
  }

  for (const point3D_t point3D_id : config.ConstantPoints()) {
    Point3D& point3D = reconstruction.Point3D(point3D_id);
    problem.SetParameterBlockConstant(point3D.xyz.data());
  }
}

class DefaultBundleAdjuster : public BundleAdjuster {
 public:
  DefaultBundleAdjuster(BundleAdjustmentOptions options,
                        BundleAdjustmentConfig config,
                        Reconstruction& reconstruction)
      : BundleAdjuster(std::move(options), std::move(config)),
        loss_function_(std::unique_ptr<ceres::LossFunction>(
            options_.CreateLossFunction())) {
    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem_ = std::make_shared<ceres::Problem>(problem_options);

    // Set up problem
    // Warning: AddPointsToProblem assumes that AddImageToProblem is called
    // first. Do not change order of instructions!
    for (const image_t image_id : config_.Images()) {
      AddImageToProblem(image_id, reconstruction);
    }
    for (const auto point3D_id : config_.VariablePoints()) {
      AddPointToProblem(point3D_id, reconstruction);
    }
    for (const auto point3D_id : config_.ConstantPoints()) {
      AddPointToProblem(point3D_id, reconstruction);
    }

    ParameterizeCameras(
        options_, config_, camera_ids_, reconstruction, *problem_);
    ParameterizePoints(
        config_, point3D_num_observations_, reconstruction, *problem_);
  }

  ceres::Solver::Summary Solve() override {
    ceres::Solver::Summary summary;
    if (problem_->NumResiduals() == 0) {
      return summary;
    }

    const ceres::Solver::Options solver_options =
        options_.CreateSolverOptions(config_, *problem_);

    ceres::Solve(solver_options, problem_.get(), &summary);

    if (options_.print_summary || VLOG_IS_ON(1)) {
      PrintSolverSummary(summary, "Bundle adjustment report");
    }

    return summary;
  }

  std::shared_ptr<ceres::Problem>& Problem() override { return problem_; }

  void AddImageToProblem(const image_t image_id,
                         Reconstruction& reconstruction) {
    Image& image = reconstruction.Image(image_id);
    Camera& camera = *image.CameraPtr();

    // CostFunction assumes unit quaternions.
    image.CamFromWorld().rotation.normalize();

    double* cam_from_world_rotation =
        image.CamFromWorld().rotation.coeffs().data();
    double* cam_from_world_translation =
        image.CamFromWorld().translation.data();
    double* camera_params = camera.params.data();

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

      Point3D& point3D = reconstruction.Point3D(point2D.point3D_id);
      assert(point3D.track.Length() > 1);

      if (constant_cam_pose) {
        problem_->AddResidualBlock(
            CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor>(
                camera.model_id, point2D.xy, image.CamFromWorld()),
            loss_function_.get(),
            point3D.xyz.data(),
            camera_params);
      } else {
        problem_->AddResidualBlock(
            CreateCameraCostFunction<ReprojErrorCostFunctor>(camera.model_id,
                                                             point2D.xy),
            loss_function_.get(),
            cam_from_world_rotation,
            cam_from_world_translation,
            point3D.xyz.data(),
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

  void AddPointToProblem(const point3D_t point3D_id,
                         Reconstruction& reconstruction) {
    Point3D& point3D = reconstruction.Point3D(point3D_id);

    // Is 3D point already fully contained in the problem? I.e. its entire track
    // is contained in `variable_image_ids`, `constant_image_ids`,
    // `constant_x_image_ids`.
    if (point3D_num_observations_[point3D_id] == point3D.track.Length()) {
      return;
    }

    for (const auto& track_el : point3D.track.Elements()) {
      // Skip observations that were already added in `FillImages`.
      if (config_.HasImage(track_el.image_id)) {
        continue;
      }

      point3D_num_observations_[point3D_id] += 1;

      Image& image = reconstruction.Image(track_el.image_id);
      Camera& camera = *image.CameraPtr();
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
      problem_->AddResidualBlock(
          CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor>(
              camera.model_id, point2D.xy, image.CamFromWorld()),
          loss_function_.get(),
          point3D.xyz.data(),
          camera.params.data());
    }
  }

 private:
  std::shared_ptr<ceres::Problem> problem_;
  std::unique_ptr<ceres::LossFunction> loss_function_;

  std::unordered_set<camera_t> camera_ids_;
  std::unordered_map<point3D_t, size_t> point3D_num_observations_;
};

class RigBundleAdjuster : public BundleAdjuster {
 public:
  RigBundleAdjuster(BundleAdjustmentOptions options,
                    RigBundleAdjustmentOptions rig_options,
                    BundleAdjustmentConfig config,
                    Reconstruction& reconstruction,
                    std::vector<CameraRig>& camera_rigs)
      : BundleAdjuster(std::move(options), std::move(config)),
        rig_options_(rig_options),
        reconstruction_(reconstruction),
        loss_function_(std::unique_ptr<ceres::LossFunction>(
            options_.CreateLossFunction())) {
    // Check the validity of the provided camera rigs.
    std::unordered_set<camera_t> rig_camera_ids;
    for (CameraRig& camera_rig : camera_rigs) {
      camera_rig.Check(reconstruction);
      for (const auto& camera_id : camera_rig.GetCameraIds()) {
        THROW_CHECK_EQ(rig_camera_ids.count(camera_id), 0)
            << "Camera must not be part of multiple camera rigs";
        rig_camera_ids.insert(camera_id);
      }

      for (const auto& snapshot : camera_rig.Snapshots()) {
        for (const auto& image_id : snapshot) {
          THROW_CHECK_EQ(image_id_to_camera_rig_.count(image_id), 0)
              << "Image must not be part of multiple camera rigs";
          image_id_to_camera_rig_.emplace(image_id, &camera_rig);
        }
      }
    }

    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem_ = std::make_shared<ceres::Problem>(problem_options);

    ExtractRigsFromWorld(reconstruction, camera_rigs);

    for (const image_t image_id : config_.Images()) {
      AddImageToProblem(image_id, reconstruction);
    }
    for (const auto point3D_id : config_.VariablePoints()) {
      AddPointToProblem(point3D_id, reconstruction);
    }
    for (const auto point3D_id : config_.ConstantPoints()) {
      AddPointToProblem(point3D_id, reconstruction);
    }

    ParameterizeCameras(
        options_, config_, camera_ids_, reconstruction, *problem_);
    ParameterizeCameraRigs();
    ParameterizePoints(
        config_, point3D_num_observations_, reconstruction, *problem_);
  }

  ceres::Solver::Summary Solve() override {
    ceres::Solver::Summary summary;
    if (problem_->NumResiduals() == 0) {
      return summary;
    }

    const ceres::Solver::Options solver_options =
        options_.CreateSolverOptions(config_, *problem_);

    ceres::Solve(solver_options, problem_.get(), &summary);

    if (options_.print_summary || VLOG_IS_ON(1)) {
      PrintSolverSummary(summary, "Rig Bundle adjustment report");
    }

    ComputeCamsFromWorld();

    return summary;
  }

  std::shared_ptr<ceres::Problem>& Problem() override { return problem_; }

  void AddImageToProblem(const image_t image_id,
                         Reconstruction& reconstruction) {
    const double max_squared_reproj_error =
        rig_options_.max_reproj_error * rig_options_.max_reproj_error;

    Image& image = reconstruction.Image(image_id);
    Camera& camera = *image.CameraPtr();

    const bool constant_cam_pose = config_.HasConstantCamPose(image_id);
    const bool constant_cam_position =
        config_.HasConstantCamPositions(image_id);

    double* camera_params = camera.params.data();
    double* cam_from_rig_rotation = nullptr;
    double* cam_from_rig_translation = nullptr;
    double* rig_from_world_rotation = nullptr;
    double* rig_from_world_translation = nullptr;
    CameraRig* camera_rig = nullptr;
    Eigen::Matrix3x4d cam_from_world_mat = Eigen::Matrix3x4d::Zero();

    if (const auto it = image_id_to_camera_rig_.find(image_id);
        it != image_id_to_camera_rig_.end()) {
      THROW_CHECK(!constant_cam_pose)
          << "Images contained in a camera rig must not have constant pose";
      THROW_CHECK(!constant_cam_position)
          << "Images contained in a camera rig must not have constant tvec";
      camera_rig = it->second;
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
    THROW_CHECK(image.HasCameraId());
    camera_ids_.insert(image.CameraId());

    // The number of added observations for the current image.
    size_t num_observations = 0;

    // Add residuals to bundle adjustment problem.
    for (const Point2D& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D()) {
        continue;
      }

      Point3D& point3D = reconstruction.Point3D(point2D.point3D_id);
      assert(point3D.track.Length() > 1);

      if (camera_rig != nullptr &&
          CalculateSquaredReprojectionError(
              point2D.xy, point3D.xyz, cam_from_world_mat, camera) >
              max_squared_reproj_error) {
        continue;
      }

      num_observations += 1;
      point3D_num_observations_[point2D.point3D_id] += 1;

      if (camera_rig == nullptr) {
        if (constant_cam_pose) {
          problem_->AddResidualBlock(
              CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor>(
                  camera.model_id, point2D.xy, image.CamFromWorld()),
              loss_function_.get(),
              point3D.xyz.data(),
              camera_params);
        } else {
          problem_->AddResidualBlock(
              CreateCameraCostFunction<ReprojErrorCostFunctor>(camera.model_id,
                                                               point2D.xy),
              loss_function_.get(),
              cam_from_rig_rotation,     // rig == world
              cam_from_rig_translation,  // rig == world
              point3D.xyz.data(),
              camera_params);
        }
      } else {
        problem_->AddResidualBlock(
            CreateCameraCostFunction<RigReprojErrorCostFunctor>(camera.model_id,
                                                                point2D.xy),
            loss_function_.get(),
            cam_from_rig_rotation,
            cam_from_rig_translation,
            rig_from_world_rotation,
            rig_from_world_translation,
            point3D.xyz.data(),
            camera_params);
      }
    }

    if (num_observations > 0) {
      parameterized_cams_from_rig_rotations_.insert(cam_from_rig_rotation);

      if (camera_rig != nullptr) {
        parameterized_cams_from_rig_rotations_.insert(rig_from_world_rotation);

        // Set the relative pose of the camera constant if relative pose
        // refinement is disabled or if it is the reference camera to avoid
        // over- parameterization of the camera pose.
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
        SetSubsetManifold(3,
                          constant_position_idxs,
                          problem_.get(),
                          cam_from_rig_translation);
      }
    }
  }

  void AddPointToProblem(const point3D_t point3D_id,
                         Reconstruction& reconstruction) {
    Point3D& point3D = reconstruction.Point3D(point3D_id);

    // Is 3D point already fully contained in the problem? I.e. its entire track
    // is contained in `variable_image_ids`, `constant_image_ids`,
    // `constant_x_image_ids`.
    if (point3D_num_observations_[point3D_id] == point3D.track.Length()) {
      return;
    }

    for (const auto& track_el : point3D.track.Elements()) {
      // Skip observations that were already added in `AddImageToProblem`.
      if (config_.HasImage(track_el.image_id)) {
        continue;
      }

      point3D_num_observations_[point3D_id] += 1;

      Image& image = reconstruction.Image(track_el.image_id);
      Camera& camera = *image.CameraPtr();
      const Point2D& point2D = image.Point2D(track_el.point2D_idx);

      // We do not want to refine the camera of images that are not
      // part of `constant_image_ids_`, `constant_image_ids_`,
      // `constant_x_image_ids_`.
      if (camera_ids_.count(image.CameraId()) == 0) {
        camera_ids_.insert(image.CameraId());
        config_.SetConstantCamIntrinsics(image.CameraId());
      }

      problem_->AddResidualBlock(
          CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor>(
              camera.model_id, point2D.xy, image.CamFromWorld()),
          loss_function_.get(),
          point3D.xyz.data(),
          camera.params.data());
    }
  }

  void ExtractRigsFromWorld(const Reconstruction& reconstruction,
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

  void ComputeCamsFromWorld() {
    for (const auto& [image_id, camera_rig] : image_id_to_camera_rig_) {
      Image& image = reconstruction_.Image(image_id);
      image.CamFromWorld() = camera_rig->CamFromRig(image.CameraId()) *
                             (*image_id_to_rig_from_world_.at(image_id));
    }
  }

  void ParameterizeCameraRigs() {
    for (double* cam_from_rig_rotation :
         parameterized_cams_from_rig_rotations_) {
      SetQuaternionManifold(problem_.get(), cam_from_rig_rotation);
    }
  }

 private:
  const RigBundleAdjustmentOptions rig_options_;

  Reconstruction& reconstruction_;

  std::shared_ptr<ceres::Problem> problem_;
  std::unique_ptr<ceres::LossFunction> loss_function_;

  std::unordered_set<camera_t> camera_ids_;
  std::unordered_map<point3D_t, size_t> point3D_num_observations_;

  // Mapping from images to camera rigs.
  std::unordered_map<image_t, CameraRig*> image_id_to_camera_rig_;
  std::unordered_map<image_t, Rigid3d*> image_id_to_rig_from_world_;

  // For each camera rig, the absolute camera rig poses for all snapshots.
  std::vector<std::vector<Rigid3d>> rigs_from_world_;

  // The Quaternions added to the problem, used to set the local
  // parameterization once after setting up the problem.
  std::unordered_set<double*> parameterized_cams_from_rig_rotations_;
};

class PosePriorBundleAdjuster : public BundleAdjuster {
 public:
  PosePriorBundleAdjuster(BundleAdjustmentOptions options,
                          PosePriorBundleAdjustmentOptions prior_options,
                          BundleAdjustmentConfig config,
                          std::unordered_map<image_t, PosePrior> pose_priors,
                          Reconstruction& reconstruction)
      : BundleAdjuster(std::move(options), std::move(config)),
        prior_options_(prior_options),
        pose_priors_(std::move(pose_priors)),
        reconstruction_(reconstruction) {
    const bool use_prior_position = AlignReconstruction();

    // Fix 7-DOFs of BA problem if not enough valid pose priors.
    if (!use_prior_position) {
      auto reg_image_ids_it = reconstruction_.RegImageIds().begin();
      config_.SetConstantCamPose(*reg_image_ids_it);
      config_.SetConstantCamPositions(*(++reg_image_ids_it), {0});
    }

    default_bundle_adjuster_ = std::make_unique<DefaultBundleAdjuster>(
        options_, config_, reconstruction);

    if (use_prior_position) {
      // Normalize the reconstruction to avoid any numerical instability but do
      // not transform priors as they will be transformed when added to
      // ceres::Problem.
      normalized_from_metric_ = reconstruction_.Normalize(/*fixed_scale=*/true);

      if (prior_options_.use_robust_loss_on_prior_position) {
        prior_loss_function_ = std::make_unique<ceres::CauchyLoss>(
            prior_options_.prior_position_loss_scale);
      }

      for (const image_t image_id : config_.Images()) {
        const auto pose_prior_it = pose_priors_.find(image_id);
        if (pose_prior_it != pose_priors_.end()) {
          AddPosePriorToProblem(
              image_id, pose_prior_it->second, reconstruction);
        }
      }
    }
  }

  ceres::Solver::Summary Solve() override {
    ceres::Solver::Summary summary;
    std::shared_ptr<ceres::Problem> problem =
        default_bundle_adjuster_->Problem();
    if (problem->NumResiduals() == 0) {
      return summary;
    }

    const ceres::Solver::Options solver_options =
        options_.CreateSolverOptions(config_, *problem);

    ceres::Solve(solver_options, problem.get(), &summary);

    reconstruction_.Transform(Inverse(normalized_from_metric_));

    if (options_.print_summary || VLOG_IS_ON(1)) {
      PrintSolverSummary(summary, "Pose Prior Bundle adjustment report");
    }

    return summary;
  }

  std::shared_ptr<ceres::Problem>& Problem() override {
    return default_bundle_adjuster_->Problem();
  }

  void AddPosePriorToProblem(image_t image_id,
                             const PosePrior& prior,
                             Reconstruction& reconstruction) {
    if (!prior.IsValid() || !prior.IsCovarianceValid()) {
      LOG(ERROR) << "Could not add prior for image #" << image_id;
      return;
    }

    std::shared_ptr<ceres::Problem> problem =
        default_bundle_adjuster_->Problem();

    Image& image = reconstruction.Image(image_id);
    THROW_CHECK(image.HasPose());

    double* cam_from_world_translation =
        image.CamFromWorld().translation.data();
    if (!problem->HasParameterBlock(cam_from_world_translation)) {
      return;
    }

    // image.CamFromWorld().rotation is already normalized in
    // AddImageToProblem()
    double* cam_from_world_rotation =
        image.CamFromWorld().rotation.coeffs().data();

    problem->AddResidualBlock(
        CovarianceWeightedCostFunctor<AbsolutePosePositionPriorCostFunctor>::
            Create(prior.position_covariance,
                   normalized_from_metric_ * prior.position),
        prior_loss_function_.get(),
        cam_from_world_rotation,
        cam_from_world_translation);
  }

  bool AlignReconstruction() {
    RANSACOptions ransac_options;
    if (prior_options_.ransac_max_error > 0) {
      ransac_options.max_error = prior_options_.ransac_max_error;
    } else {
      double max_stddev_sum = 0;
      size_t num_valid_covs = 0;
      for (const auto& [_, pose_prior] : pose_priors_) {
        if (pose_prior.IsCovarianceValid()) {
          const double max_stddev =
              std::sqrt(Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(
                            pose_prior.position_covariance)
                            .eigenvalues()
                            .maxCoeff());
          max_stddev_sum += max_stddev;
          ++num_valid_covs;
        }
      }
      if (num_valid_covs == 0) {
        LOG(WARNING) << "No pose priors with valid covariance found.";
        return false;
      }
      // Set max error at the 3 sigma confidence interval. Assumes no outliers.
      ransac_options.max_error = 3 * max_stddev_sum / num_valid_covs;
    }

    Sim3d metric_from_orig;
    const bool success = AlignReconstructionToPosePriors(
        reconstruction_, pose_priors_, ransac_options, &metric_from_orig);

    if (success) {
      reconstruction_.Transform(metric_from_orig);
    } else {
      LOG(WARNING) << "Alignment w.r.t. prior positions failed";
    }

    if (VLOG_IS_ON(2) && success) {
      std::vector<double> verr2_wrt_prior;
      verr2_wrt_prior.reserve(reconstruction_.NumRegImages());
      for (const image_t image_id : reconstruction_.RegImageIds()) {
        const auto pose_prior_it = pose_priors_.find(image_id);
        if (pose_prior_it != pose_priors_.end() &&
            pose_prior_it->second.IsValid()) {
          const auto& image = reconstruction_.Image(image_id);
          verr2_wrt_prior.push_back(
              (image.ProjectionCenter() - pose_prior_it->second.position)
                  .squaredNorm());
        }
      }

      VLOG(2) << "Alignment error w.r.t. prior positions:\n"
              << "  - rmse:   " << std::sqrt(Mean(verr2_wrt_prior)) << "\n"
              << "  - median: " << std::sqrt(Median(verr2_wrt_prior)) << "\n";
    }

    return success;
  }

 private:
  const PosePriorBundleAdjustmentOptions prior_options_;
  const std::unordered_map<image_t, PosePrior> pose_priors_;
  Reconstruction& reconstruction_;

  std::unique_ptr<DefaultBundleAdjuster> default_bundle_adjuster_;
  std::unique_ptr<ceres::LossFunction> prior_loss_function_;

  Sim3d normalized_from_metric_;
};

}  // namespace

std::unique_ptr<BundleAdjuster> CreateDefaultBundleAdjuster(
    BundleAdjustmentOptions options,
    BundleAdjustmentConfig config,
    Reconstruction& reconstruction) {
  return std::make_unique<DefaultBundleAdjuster>(
      std::move(options), std::move(config), reconstruction);
}

std::unique_ptr<BundleAdjuster> CreateRigBundleAdjuster(
    BundleAdjustmentOptions options,
    RigBundleAdjustmentOptions rig_options,
    BundleAdjustmentConfig config,
    Reconstruction& reconstruction,
    std::vector<CameraRig>& camera_rigs) {
  return std::make_unique<RigBundleAdjuster>(std::move(options),
                                             rig_options,
                                             std::move(config),
                                             reconstruction,
                                             camera_rigs);
}

std::unique_ptr<BundleAdjuster> CreatePosePriorBundleAdjuster(
    BundleAdjustmentOptions options,
    PosePriorBundleAdjustmentOptions prior_options,
    BundleAdjustmentConfig config,
    std::unordered_map<image_t, PosePrior> pose_priors,
    Reconstruction& reconstruction) {
  return std::make_unique<PosePriorBundleAdjuster>(std::move(options),
                                                   prior_options,
                                                   std::move(config),
                                                   std::move(pose_priors),
                                                   reconstruction);
}

void PrintSolverSummary(const ceres::Solver::Summary& summary,
                        const std::string& header) {
  if (VLOG_IS_ON(3)) {
    LOG(INFO) << summary.FullReport();
  }

  std::ostringstream log;
  log << header << "\n";
  log << std::right << std::setw(16) << "Residuals : ";
  log << std::left << summary.num_residuals_reduced << "\n";

  log << std::right << std::setw(16) << "Parameters : ";
  log << std::left << summary.num_effective_parameters_reduced << "\n";

  log << std::right << std::setw(16) << "Iterations : ";
  log << std::left
      << summary.num_successful_steps + summary.num_unsuccessful_steps << "\n";

  log << std::right << std::setw(16) << "Time : ";
  log << std::left << summary.total_time_in_seconds << " [s]\n";

  log << std::right << std::setw(16) << "Initial cost : ";
  log << std::right << std::setprecision(6)
      << std::sqrt(summary.initial_cost / summary.num_residuals_reduced)
      << " [px]\n";

  log << std::right << std::setw(16) << "Final cost : ";
  log << std::right << std::setprecision(6)
      << std::sqrt(summary.final_cost / summary.num_residuals_reduced)
      << " [px]\n";

  log << std::right << std::setw(16) << "Termination : ";

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

  log << std::right << termination << "\n\n";
  LOG(INFO) << log.str();
}

}  // namespace colmap

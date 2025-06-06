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

void BundleAdjustmentConfig::FixGauge(BundleAdjustmentGauge gauge) {
  fixed_gauge_ = gauge;
}

BundleAdjustmentGauge BundleAdjustmentConfig::FixedGauge() const {
  return fixed_gauge_;
}

size_t BundleAdjustmentConfig::NumImages() const { return image_ids_.size(); }

size_t BundleAdjustmentConfig::NumPoints() const {
  return variable_point3D_ids_.size() + constant_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantCamIntrinsics() const {
  return constant_cam_intrinsics_.size();
}

size_t BundleAdjustmentConfig::NumConstantSensorFromRigPoses() const {
  return constant_sensor_from_rig_poses_.size();
}

size_t BundleAdjustmentConfig::NumConstantRigFromWorldPoses() const {
  return constant_rig_from_world_poses_.size();
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
        ++num_observations_for_point;
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
  constant_cam_intrinsics_.insert(camera_id);
}

void BundleAdjustmentConfig::SetVariableCamIntrinsics(
    const camera_t camera_id) {
  constant_cam_intrinsics_.erase(camera_id);
}

bool BundleAdjustmentConfig::HasConstantCamIntrinsics(
    const camera_t camera_id) const {
  return constant_cam_intrinsics_.find(camera_id) !=
         constant_cam_intrinsics_.end();
}

void BundleAdjustmentConfig::SetConstantSensorFromRigPose(
    const sensor_t sensor_id) {
  constant_sensor_from_rig_poses_.insert(sensor_id);
}

void BundleAdjustmentConfig::SetVariableSensorFromRigPose(
    const sensor_t sensor_id) {
  constant_sensor_from_rig_poses_.erase(sensor_id);
}

bool BundleAdjustmentConfig::HasConstantSensorFromRigPose(
    const sensor_t sensor_id) const {
  return constant_sensor_from_rig_poses_.find(sensor_id) !=
         constant_sensor_from_rig_poses_.end();
}

void BundleAdjustmentConfig::SetConstantRigFromWorldPose(
    const frame_t frame_id) {
  THROW_CHECK(HasImage(frame_id));
  constant_rig_from_world_poses_.insert(frame_id);
}

void BundleAdjustmentConfig::SetVariableRigFromWorldPose(
    const frame_t frame_id) {
  constant_rig_from_world_poses_.erase(frame_id);
}

bool BundleAdjustmentConfig::HasConstantRigFromWorldPose(
    const frame_t frame_id) const {
  return constant_rig_from_world_poses_.find(frame_id) !=
         constant_rig_from_world_poses_.end();
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

const std::unordered_set<camera_t>
BundleAdjustmentConfig::ConstantCamIntrinsics() const {
  return constant_cam_intrinsics_;
}

const std::unordered_set<sensor_t>&
BundleAdjustmentConfig::ConstantSensorFromRigPoses() const {
  return constant_sensor_from_rig_poses_;
}

const std::unordered_set<frame_t>&
BundleAdjustmentConfig::ConstantRigFromWorldPoses() const {
  return constant_rig_from_world_poses_;
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
                         const std::set<camera_t>& camera_ids,
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

void FixGaugeWithTwoCamsFromWorld(const BundleAdjustmentOptions& options,
                                  const BundleAdjustmentConfig& config,
                                  const std::set<image_t>& image_ids,
                                  Reconstruction& reconstruction,
                                  ceres::Problem& problem) {
  const size_t num_constant_images = std::count_if(
      image_ids.begin(),
      image_ids.end(),
      [&config, &reconstruction](image_t image_id) {
        Image& image = reconstruction.Image(image_id);
        return config.HasConstantRigFromWorldPose(image.FrameId());
      });
  if (num_constant_images >= 2) {
    return;
  }

  Image* image1 = nullptr;
  Image* image2 = nullptr;
  Eigen::Index frame2_from_world_fixed_dim = 0;
  for (const image_t image_id : image_ids) {
    Image& image = reconstruction.Image(image_id);
    if (image1 == nullptr && image.FramePtr()->RigPtr()->IsRefSensor(
                                 image.CameraPtr()->SensorId())) {
      image1 = &image;
    } else if (image1 != nullptr && image1->FrameId() != image.FrameId() &&
               image.FramePtr()->RigPtr()->IsRefSensor(
                   image.CameraPtr()->SensorId())) {
      // Check if one of the baseline dimensions is large enough and
      // choose it as the fixed coordinate. If there is no such pair of
      // frames, then the scale is not constrained well.
      const Eigen::Vector3d baseline =
          (image1->FramePtr()->RigFromWorld() *
           Inverse(image.FramePtr()->RigFromWorld()))
              .translation;
      if (baseline.cwiseAbs().maxCoeff(&frame2_from_world_fixed_dim) > 1e-9) {
        image2 = &image;
        break;
      }
    }
  }

  // TODO(jsch): Once we support IMUs or other sensors, we have to
  // fix the Gauge differently, as we are not guaranteed to find two
  // images/cameras that are reference sensors in different frames.
  THROW_CHECK(image1 != nullptr && image2 != nullptr);

  Rigid3d& frame1_from_world = image1->FramePtr()->RigFromWorld();
  if (!config.HasConstantRigFromWorldPose(image1->FrameId())) {
    problem.SetParameterBlockConstant(
        frame1_from_world.rotation.coeffs().data());
    problem.SetParameterBlockConstant(frame1_from_world.translation.data());
  }

  Rigid3d& frame2_from_world = image2->FramePtr()->RigFromWorld();
  if (!config.HasConstantRigFromWorldPose(image2->FrameId())) {
    SetSubsetManifold(3,
                      {static_cast<int>(frame2_from_world_fixed_dim)},
                      &problem,
                      frame2_from_world.translation.data());
  }
}

void ParameterizeImages(const BundleAdjustmentOptions& options,
                        const BundleAdjustmentConfig& config,
                        const std::set<image_t>& image_ids,
                        Reconstruction& reconstruction,
                        ceres::Problem& problem) {
  std::unordered_set<rig_t> parameterized_rig_ids;
  std::unordered_set<sensor_t> parameterized_sensor_ids;
  std::unordered_set<frame_t> parameterized_frame_ids;
  for (const image_t image_id : image_ids) {
    Image& image = reconstruction.Image(image_id);
    parameterized_rig_ids.insert(image.FramePtr()->RigId());

    // Parameterize sensor_from_rig.
    const sensor_t sensor_id = image.CameraPtr()->SensorId();
    const bool not_parameterized_before =
        parameterized_sensor_ids.insert(sensor_id).second;
    if (not_parameterized_before && !image.HasTrivialFrame()) {
      Rigid3d& sensor_from_rig =
          image.FramePtr()->RigPtr()->SensorFromRig(sensor_id);
      // CostFunction assumes unit quaternions.
      sensor_from_rig.rotation.normalize();
      if (problem.HasParameterBlock(sensor_from_rig.rotation.coeffs().data())) {
        SetQuaternionManifold(&problem,
                              sensor_from_rig.rotation.coeffs().data());
        if (!options.refine_sensor_from_rig ||
            config.HasConstantSensorFromRigPose(sensor_id)) {
          problem.SetParameterBlockConstant(
              sensor_from_rig.rotation.coeffs().data());
          problem.SetParameterBlockConstant(sensor_from_rig.translation.data());
        }
      }
    }

    // Parameterize rig_from_world.
    if (parameterized_frame_ids.insert(image.FrameId()).second) {
      Rigid3d& rig_from_world = image.FramePtr()->RigFromWorld();
      // CostFunction assumes unit quaternions.
      rig_from_world.rotation.normalize();
      if (problem.HasParameterBlock(rig_from_world.rotation.coeffs().data())) {
        SetQuaternionManifold(&problem,
                              rig_from_world.rotation.coeffs().data());
        if (!options.refine_rig_from_world ||
            config.HasConstantRigFromWorldPose(image.FrameId())) {
          problem.SetParameterBlockConstant(
              rig_from_world.rotation.coeffs().data());
          problem.SetParameterBlockConstant(rig_from_world.translation.data());
        }
      }
    }
  }

  // Set the rig poses as constant, if the reference sensor is not part of the
  // problem. Otherwise, the relative pose between the sensors is not well
  // constrained. Notice that this does not handle degenerate configurations and
  // assumes the observations in the problem constrain the relative poses
  // sufficiently.
  for (const rig_t rig_id : parameterized_rig_ids) {
    Rig& rig = reconstruction.Rig(rig_id);
    if (parameterized_sensor_ids.count(rig.RefSensorId()) != 0) {
      continue;
    }
    for (auto& [_, sensor_from_rig] : rig.Sensors()) {
      if (sensor_from_rig.has_value() &&
          problem.HasParameterBlock(sensor_from_rig->translation.data())) {
        problem.SetParameterBlockConstant(
            sensor_from_rig->rotation.coeffs().data());
        problem.SetParameterBlockConstant(sensor_from_rig->translation.data());
      }
    }
  }

  if (config.FixedGauge() == BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD &&
      options.refine_rig_from_world) {
    FixGaugeWithTwoCamsFromWorld(
        options, config, image_ids, reconstruction, problem);
  }
}

struct FixedGaugeWithThreePoints {
  // The number of fixed points for the Gauge.
  Eigen::Index num_fixed_points = 0;
  // The coordinates of the fixed points as columns.
  Eigen::Matrix3d fixed_points = Eigen::Matrix3d::Zero();
  bool MaybeAddPoint(const Eigen::Vector3d& point) {
    if (num_fixed_points >= 3) {
      return false;
    }
    fixed_points.col(num_fixed_points) = point;
    if (fixed_points.colPivHouseholderQr().rank() > num_fixed_points) {
      ++num_fixed_points;
      return true;
    } else {
      fixed_points.col(num_fixed_points).setZero();
      return false;
    }
  }
};

void FixGaugeWithThreePoints(
    const std::unordered_map<point3D_t, size_t>& point3D_num_observations,
    FixedGaugeWithThreePoints& fixed_gauge,
    Reconstruction& reconstruction,
    ceres::Problem& problem) {
  for (const auto& [point3D_id, num_observations] : point3D_num_observations) {
    Point3D& point3D = reconstruction.Point3D(point3D_id);
    if (point3D.track.Length() == num_observations &&
        fixed_gauge.MaybeAddPoint(point3D.xyz)) {
      problem.SetParameterBlockConstant(point3D.xyz.data());
      if (fixed_gauge.num_fixed_points >= 3) {
        break;
      }
    }
  }

  LOG_IF(WARNING, fixed_gauge.num_fixed_points < 3)
      << "Failed to fix Gauge due to insufficient number of fixed points";
}

void ParameterizePoints(
    const BundleAdjustmentConfig& config,
    const std::unordered_map<point3D_t, size_t>& point3D_num_observations,
    Reconstruction& reconstruction,
    ceres::Problem& problem) {
  FixedGaugeWithThreePoints fixed_gauge;

  for (const auto& [point3D_id, num_observations] : point3D_num_observations) {
    Point3D& point3D = reconstruction.Point3D(point3D_id);
    if (point3D.track.Length() > num_observations) {
      problem.SetParameterBlockConstant(point3D.xyz.data());
      fixed_gauge.MaybeAddPoint(point3D.xyz);
    }
  }

  for (const point3D_t point3D_id : config.ConstantPoints()) {
    Point3D& point3D = reconstruction.Point3D(point3D_id);
    problem.SetParameterBlockConstant(point3D.xyz.data());
    fixed_gauge.MaybeAddPoint(point3D.xyz);
  }

  if (config.FixedGauge() == BundleAdjustmentGauge::THREE_POINTS &&
      fixed_gauge.num_fixed_points < 3) {
    FixGaugeWithThreePoints(
        point3D_num_observations, fixed_gauge, reconstruction, problem);
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

    ParameterizeCameras(options_,
                        config_,
                        parameterized_camera_ids_,
                        reconstruction,
                        *problem_);
    ParameterizeImages(
        options_, config_, parameterized_image_ids_, reconstruction, *problem_);
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

    if (image.HasTrivialFrame()) {
      AddImageWithTrivialFrame(image, reconstruction);
    } else {
      AddImageWithNonTrivialFrame(image, reconstruction);
    }
  }

  void AddImageWithTrivialFrame(Image& image, Reconstruction& reconstruction) {
    Camera& camera = *image.CameraPtr();

    THROW_CHECK(image.HasTrivialFrame());
    Rigid3d& cam_from_world = image.FramePtr()->RigFromWorld();

    const bool constant_cam_from_world =
        !options_.refine_rig_from_world ||
        config_.HasConstantRigFromWorldPose(image.FrameId());

    // Add residuals to bundle adjustment problem.
    size_t num_observations = 0;
    for (const Point2D& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D()) {
        continue;
      }

      num_observations += 1;
      point3D_num_observations_[point2D.point3D_id] += 1;

      Point3D& point3D = reconstruction.Point3D(point2D.point3D_id);
      THROW_CHECK_GT(point3D.track.Length(), 1);

      if (constant_cam_from_world) {
        problem_->AddResidualBlock(
            CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor>(
                camera.model_id, point2D.xy, cam_from_world),
            loss_function_.get(),
            point3D.xyz.data(),
            camera.params.data());
      } else {
        problem_->AddResidualBlock(
            CreateCameraCostFunction<ReprojErrorCostFunctor>(camera.model_id,
                                                             point2D.xy),
            loss_function_.get(),
            cam_from_world.rotation.coeffs().data(),
            cam_from_world.translation.data(),
            point3D.xyz.data(),
            camera.params.data());
      }
    }

    if (num_observations > 0) {
      parameterized_camera_ids_.insert(image.CameraId());
      parameterized_image_ids_.insert(image.ImageId());
    }
  }

  void AddImageWithNonTrivialFrame(Image& image,
                                   Reconstruction& reconstruction) {
    Camera& camera = *image.CameraPtr();
    const sensor_t sensor_id = camera.SensorId();

    THROW_CHECK(!image.HasTrivialFrame());
    Rigid3d& cam_from_rig =
        image.FramePtr()->RigPtr()->SensorFromRig(sensor_id);
    Rigid3d& rig_from_world = image.FramePtr()->RigFromWorld();

    const bool constant_sensor_from_rig =
        !options_.refine_sensor_from_rig ||
        config_.HasConstantSensorFromRigPose(sensor_id);
    const bool constant_rig_from_world =
        !options_.refine_rig_from_world ||
        config_.HasConstantRigFromWorldPose(image.FrameId());

    // Add residuals to bundle adjustment problem.
    size_t num_observations = 0;
    for (const Point2D& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D()) {
        continue;
      }

      num_observations += 1;
      point3D_num_observations_[point2D.point3D_id] += 1;

      Point3D& point3D = reconstruction.Point3D(point2D.point3D_id);
      THROW_CHECK_GT(point3D.track.Length(), 1);

      // The !constant_sensor_from_rig && constant_rig_from_world is
      // rare enough that we do not have a specialized cost function for it.
      if (constant_sensor_from_rig && constant_rig_from_world) {
        problem_->AddResidualBlock(
            CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor>(
                camera.model_id, point2D.xy, cam_from_rig * rig_from_world),
            loss_function_.get(),
            point3D.xyz.data(),
            camera.params.data());
      } else if (!constant_rig_from_world && constant_sensor_from_rig) {
        problem_->AddResidualBlock(
            CreateCameraCostFunction<RigReprojErrorConstantRigCostFunctor>(
                camera.model_id, point2D.xy, cam_from_rig),
            loss_function_.get(),
            rig_from_world.rotation.coeffs().data(),
            rig_from_world.translation.data(),
            point3D.xyz.data(),
            camera.params.data());
      } else {
        problem_->AddResidualBlock(
            CreateCameraCostFunction<RigReprojErrorCostFunctor>(camera.model_id,
                                                                point2D.xy),
            loss_function_.get(),
            cam_from_rig.rotation.coeffs().data(),
            cam_from_rig.translation.data(),
            rig_from_world.rotation.coeffs().data(),
            rig_from_world.translation.data(),
            point3D.xyz.data(),
            camera.params.data());
      }
    }

    if (num_observations > 0) {
      parameterized_camera_ids_.insert(image.CameraId());
      parameterized_image_ids_.insert(image.ImageId());
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

      if (image.HasTrivialFrame()) {
        Rigid3d& cam_from_world = image.FramePtr()->RigFromWorld();

        problem_->AddResidualBlock(
            CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor>(
                camera.model_id, point2D.xy, cam_from_world),
            loss_function_.get(),
            point3D.xyz.data(),
            camera.params.data());
      } else {
        Rigid3d& cam_from_rig = image.FramePtr()->RigPtr()->SensorFromRig(
            image.CameraPtr()->SensorId());
        Rigid3d& rig_from_world = image.FramePtr()->RigFromWorld();

        problem_->AddResidualBlock(
            CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor>(
                camera.model_id, point2D.xy, cam_from_rig * rig_from_world),
            loss_function_.get(),
            point3D.xyz.data(),
            camera.params.data());
      }

      // Do not optimize intrinsics if th corresponding images
      // were not included explicitly in the config.
      if (parameterized_camera_ids_.insert(image.CameraId()).second) {
        config_.SetConstantCamIntrinsics(image.CameraId());
      }
    }
  }

 private:
  std::shared_ptr<ceres::Problem> problem_;
  std::unique_ptr<ceres::LossFunction> loss_function_;

  std::set<camera_t> parameterized_camera_ids_;
  std::set<image_t> parameterized_image_ids_;
  std::unordered_map<point3D_t, size_t> point3D_num_observations_;
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
    if (use_prior_position) {
      // Normalize the reconstruction to avoid any numerical instability but do
      // not transform priors as they will be transformed when added to
      // ceres::Problem.
      normalized_from_metric_ = reconstruction_.Normalize(/*fixed_scale=*/true);
    } else {
      config_.FixGauge(BundleAdjustmentGauge::THREE_POINTS);
    }

    default_bundle_adjuster_ = std::make_unique<DefaultBundleAdjuster>(
        options_, config_, reconstruction);

    if (use_prior_position) {
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

    Image& image = reconstruction.Image(image_id);
    if (!image.HasTrivialFrame()) {
      // TODO(jsch): Only enforce the pose prior on the reference sensor. This
      // fails if only a non-reference sensor image has a corresponding pose
      // prior stored. This will be replaced with dedicated modeling of a
      // GNSS/GPS sensor.
      return;
    }

    THROW_CHECK(image.HasPose());
    Rigid3d& cam_from_world = image.FramePtr()->RigFromWorld();

    std::shared_ptr<ceres::Problem>& problem =
        default_bundle_adjuster_->Problem();

    double* cam_from_world_translation = cam_from_world.translation.data();
    if (!problem->HasParameterBlock(cam_from_world_translation)) {
      return;
    }

    // cam_from_world.rotation is normalized in AddImageToProblem()
    double* cam_from_world_rotation = cam_from_world.rotation.coeffs().data();

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

    VLOG(2) << "Robustly aligning reconstruction with max_error="
            << ransac_options.max_error;

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
              << "  - rmse:   " << std::sqrt(Mean(verr2_wrt_prior)) << '\n'
              << "  - median: " << std::sqrt(Median(verr2_wrt_prior)) << '\n';
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
  log << header << '\n';
  log << std::right << std::setw(16) << "Residuals : ";
  log << std::left << summary.num_residuals_reduced << '\n';

  log << std::right << std::setw(16) << "Parameters : ";
  log << std::left << summary.num_effective_parameters_reduced << '\n';

  log << std::right << std::setw(16) << "Iterations : ";
  log << std::left
      << summary.num_successful_steps + summary.num_unsuccessful_steps << '\n';

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

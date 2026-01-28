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

#include "colmap/estimators/bundle_adjustment_ceres.h"

#include "colmap/estimators/alignment.h"
#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/pose_prior.h"
#include "colmap/estimators/cost_functions/reprojection_error.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/util/cuda.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <iomanip>

namespace colmap {

namespace {

BundleAdjustmentTerminationType CeresTerminationTypeToTerminationType(
    ceres::TerminationType ceres_type) {
  switch (ceres_type) {
    case ceres::CONVERGENCE:
      return BundleAdjustmentTerminationType::CONVERGENCE;
    case ceres::NO_CONVERGENCE:
      return BundleAdjustmentTerminationType::NO_CONVERGENCE;
    case ceres::FAILURE:
      return BundleAdjustmentTerminationType::FAILURE;
    case ceres::USER_SUCCESS:
      return BundleAdjustmentTerminationType::USER_SUCCESS;
    case ceres::USER_FAILURE:
      return BundleAdjustmentTerminationType::USER_FAILURE;
  }
  LOG(FATAL_THROW) << "Unknown Ceres termination type: " << ceres_type;
  return BundleAdjustmentTerminationType::FAILURE;
}

std::unique_ptr<ceres::LossFunction> CreateLossFunction(
    CeresBundleAdjustmentOptions::LossFunctionType loss_function_type,
    double loss_function_scale) {
  switch (loss_function_type) {
    case CeresBundleAdjustmentOptions::LossFunctionType::TRIVIAL:
      return std::make_unique<ceres::TrivialLoss>();
    case CeresBundleAdjustmentOptions::LossFunctionType::SOFT_L1:
      return std::make_unique<ceres::SoftLOneLoss>(loss_function_scale);
    case CeresBundleAdjustmentOptions::LossFunctionType::CAUCHY:
      return std::make_unique<ceres::CauchyLoss>(loss_function_scale);
    case CeresBundleAdjustmentOptions::LossFunctionType::HUBER:
      return std::make_unique<ceres::HuberLoss>(loss_function_scale);
  }
  return nullptr;
}

}  // namespace

std::shared_ptr<CeresBundleAdjustmentSummary>
CeresBundleAdjustmentSummary::Create(
    const ceres::Solver::Summary& ceres_summary) {
  auto summary = std::make_shared<CeresBundleAdjustmentSummary>();
  summary->termination_type =
      CeresTerminationTypeToTerminationType(ceres_summary.termination_type);
  summary->num_residuals = ceres_summary.num_residuals_reduced;
  summary->ceres_summary = ceres_summary;
  return summary;
}

std::string CeresBundleAdjustmentSummary::BriefReport() const {
  return ceres_summary.BriefReport();
}

////////////////////////////////////////////////////////////////////////////////
// CeresBundleAdjustmentOptions
////////////////////////////////////////////////////////////////////////////////

CeresBundleAdjustmentOptions::CeresBundleAdjustmentOptions() {
  solver_options.function_tolerance = 0.0;
  solver_options.gradient_tolerance = 1e-4;
  solver_options.parameter_tolerance = 0.0;
  solver_options.logging_type = ceres::LoggingType::SILENT;
  solver_options.max_num_iterations = 100;
  solver_options.max_linear_solver_iterations = 200;
  solver_options.max_num_consecutive_invalid_steps = 10;
  solver_options.max_consecutive_nonmonotonic_steps = 10;
  solver_options.num_threads = -1;
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads = -1;
#endif  // CERES_VERSION_MAJOR
}

std::unique_ptr<ceres::LossFunction>
CeresBundleAdjustmentOptions::CreateLossFunction() const {
  return colmap::CreateLossFunction(loss_function_type, loss_function_scale);
}

ceres::Solver::Options CeresBundleAdjustmentOptions::CreateSolverOptions(
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

  // Auto-select solver type based on problem size, unless disabled.
  if (auto_select_solver_type) {
    if (num_images <= max_num_images_direct_dense_solver) {
      custom_solver_options.linear_solver_type = ceres::DENSE_SCHUR;
    } else if (has_sparse &&
               num_images <= max_num_images_direct_sparse_solver) {
      custom_solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    } else {  // Indirect sparse (preconditioned CG) solver.
      custom_solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
      custom_solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
    }
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

bool CeresBundleAdjustmentOptions::Check() const {
  CHECK_OPTION_GE(loss_function_scale, 0);
  CHECK_OPTION_LT(max_num_images_direct_dense_cpu_solver,
                  max_num_images_direct_sparse_cpu_solver);
  CHECK_OPTION_LT(max_num_images_direct_dense_gpu_solver,
                  max_num_images_direct_sparse_gpu_solver);
  return true;
}

bool CeresPosePriorBundleAdjustmentOptions::Check() const {
  CHECK_OPTION_GT(prior_position_loss_scale, 0);
  return true;
}

namespace {

struct FixedGaugeWithThreePoints {
  // The number of fixed points for the Gauge.
  Eigen::Index num_fixed_points = 0;
  // The coordinates of the fixed points as columns.
  Eigen::Matrix3d fixed_points = Eigen::Matrix3d::Zero();
  bool MaybeAddFixedPoint(const Eigen::Vector3d& point) {
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
    Reconstruction& reconstruction,
    ceres::Problem& problem) {
  FixedGaugeWithThreePoints fixed_gauge;

  // First check if we already fixed enough points in the problem.
  for (const auto& [point3D_id, num_observations] : point3D_num_observations) {
    const Point3D& point3D = reconstruction.Point3D(point3D_id);
    if (problem.IsParameterBlockConstant(point3D.xyz.data()) &&
        fixed_gauge.MaybeAddFixedPoint(point3D.xyz) &&
        fixed_gauge.num_fixed_points >= 3) {
      return;
    }
  }

  // Otherwise, fix sufficient points in the problem.
  for (const auto& [point3D_id, num_observations] : point3D_num_observations) {
    Point3D& point3D = reconstruction.Point3D(point3D_id);
    if (!problem.IsParameterBlockConstant(point3D.xyz.data()) &&
        fixed_gauge.MaybeAddFixedPoint(point3D.xyz)) {
      problem.SetParameterBlockConstant(point3D.xyz.data());
      if (fixed_gauge.num_fixed_points >= 3) {
        return;
      }
    }
  }

  LOG(WARNING)
      << "Failed to fix Gauge due to insufficient number of fixed points: "
      << fixed_gauge.num_fixed_points;
}

// Note that the following implementation does not handle all degenerate edge
// cases well, e.g., where the selected two cameras are not well constrained
// with respect to each other with shared observations. Furthermore, the
// implementation could be more sophisticated for multi-camera rigs by selecting
// camera pairs within a rig, etc.
void FixGaugeWithTwoCamsFromWorld(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    const std::set<image_t>& image_ids,
    const std::unordered_map<point3D_t, size_t>& point3D_num_observations,
    Reconstruction& reconstruction,
    ceres::Problem& problem) {
  // No need to fix the Gauge if all frames are constant.
  if (!options.refine_rig_from_world) {
    return;
  }

  Image* image1 = nullptr;
  Image* image2 = nullptr;

  // Check if a sensor is either a reference sensor, or a non-reference sensor
  // with sensor_from_rig fixed.
  auto IsParameterizedConstSensor =
      [&problem, &config, &options](const Image& image) {
        const sensor_t sensor_id = image.CameraPtr()->SensorId();
        if (image.FramePtr()->RigPtr()->IsRefSensor(sensor_id)) {
          return true;
        }
        const Rigid3d& sensor_from_rig =
            image.FramePtr()->RigPtr()->SensorFromRig(sensor_id);
        if (problem.HasParameterBlock(sensor_from_rig.params.data()) &&
            problem.IsParameterBlockConstant(sensor_from_rig.params.data())) {
          return true;
        }
        // Cover corner case when ReprojErrorConstantPoseCostFunctor is used
        if (config.HasConstantSensorFromRigPose(sensor_id) ||
            !options.refine_sensor_from_rig) {
          return true;
        }
        return false;
      };

  // First, search through the already fixed cameras in the problem.
  for (const image_t image_id : image_ids) {
    Image& image = reconstruction.Image(image_id);
    if (config.HasConstantRigFromWorldPose(image.FrameId()) &&
        IsParameterizedConstSensor(image)) {
      if (image1 == nullptr) {
        image1 = &image;
      } else if (image1 != nullptr && image1->FrameId() != image.FrameId()) {
        // No need to fix the Gauge if two frames are already fixed.
        return;
      }
    }
  }

  // Otherwise, search through the variable cameras in the problem.
  int frame2_from_world_fixed_dim = 0;
  for (const image_t image_id : image_ids) {
    Image& image = reconstruction.Image(image_id);
    const Rigid3d& rig_from_world = image.FramePtr()->RigFromWorld();
    if (image1 == nullptr && IsParameterizedConstSensor(image)) {
      image1 = &image;
    } else if (image1 != nullptr && image1->FrameId() != image.FrameId() &&
               IsParameterizedConstSensor(image) &&
               problem.HasParameterBlock(rig_from_world.params.data())) {
      // Check if one of the baseline dimensions is large enough and
      // choose it as the fixed coordinate. If there is no such pair of
      // frames, then the scale is not constrained well.
      const Eigen::Vector3d baseline =
          (image1->FramePtr()->RigFromWorld() *
           Inverse(image.FramePtr()->RigFromWorld()))
              .translation();
      Eigen::Index max_coeff_idx = 0;
      if (baseline.cwiseAbs().maxCoeff(&max_coeff_idx) > 1e-9) {
        image2 = &image;
        frame2_from_world_fixed_dim = max_coeff_idx;
        break;
      }
    }
  }

  // TODO(jsch): Notice that we could alternatively fall back to fixing the
  // Gauge between two cameras in the same frame or in different frames. Since
  // there are many different combinations to iterate through, we instead fall
  // back to fixing the Gauge with three points for simplicity. Furthermore,
  // once we support IMUs or other sensors, we should fix the Gauge differently.
  if (image1 == nullptr || image2 == nullptr) {
    LOG(WARNING) << "Failed to fix Gauge with two cameras. "
                    "Falling back to fixing Gauge with three points.";
    FixGaugeWithThreePoints(point3D_num_observations, reconstruction, problem);
    return;
  }

  if (!config.HasConstantRigFromWorldPose(image1->FrameId())) {
    const Rigid3d& frame1_from_world = image1->FramePtr()->RigFromWorld();
    problem.SetParameterBlockConstant(frame1_from_world.params.data());
  }

  if (!config.HasConstantRigFromWorldPose(image2->FrameId())) {
    Rigid3d& frame2_from_world = image2->FramePtr()->RigFromWorld();
    if (options.constant_rig_from_world_rotation) {
      SetManifold(&problem,
                  frame2_from_world.params.data(),
                  CreateSubsetManifold(
                      7, {0, 1, 2, 3, 4 + frame2_from_world_fixed_dim}));
    } else {
      SetManifold(&problem,
                  frame2_from_world.params.data(),
                  CreateProductManifold(
                      CreateEigenQuaternionManifold(),
                      CreateSubsetManifold(3, {frame2_from_world_fixed_dim})));
    }
  }
}

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
      const_camera_params.reserve(camera.params.size());

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

      if (!const_camera_params.empty()) {
        SetManifold(
            &problem,
            camera.params.data(),
            CreateSubsetManifold(camera.params.size(), const_camera_params));
      }
    }
  }
}

void ParameterizeRigsAndFrames(const BundleAdjustmentOptions& options,
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
    if (not_parameterized_before && !image.IsRefInFrame()) {
      Rigid3d& sensor_from_rig =
          image.FramePtr()->RigPtr()->SensorFromRig(sensor_id);
      // CostFunction assumes unit quaternions.
      sensor_from_rig.rotation().normalize();
      if (problem.HasParameterBlock(sensor_from_rig.params.data())) {
        SetManifold(&problem,
                    sensor_from_rig.params.data(),
                    CreateProductManifold(CreateEigenQuaternionManifold(),
                                          CreateEuclideanManifold<3>()));
        if (!options.refine_sensor_from_rig ||
            config.HasConstantSensorFromRigPose(sensor_id)) {
          problem.SetParameterBlockConstant(sensor_from_rig.params.data());
        }
      }
    }

    // Parameterize rig_from_world.
    if (parameterized_frame_ids.insert(image.FrameId()).second) {
      Rigid3d& rig_from_world = image.FramePtr()->RigFromWorld();
      // CostFunction assumes unit quaternions.
      rig_from_world.rotation().normalize();
      if (problem.HasParameterBlock(rig_from_world.params.data())) {
        if (!options.refine_rig_from_world ||
            config.HasConstantRigFromWorldPose(image.FrameId())) {
          problem.SetParameterBlockConstant(rig_from_world.params.data());
        } else if (options.constant_rig_from_world_rotation) {
          SetManifold(&problem,
                      rig_from_world.params.data(),
                      CreateSubsetManifold(7, {0, 1, 2, 3}));
        } else {
          SetManifold(&problem,
                      rig_from_world.params.data(),
                      CreateProductManifold(CreateEigenQuaternionManifold(),
                                            CreateEuclideanManifold<3>()));
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
    for (auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
      THROW_CHECK(sensor_from_rig.has_value());
      if (problem.HasParameterBlock(sensor_from_rig->params.data())) {
        problem.SetParameterBlockConstant(sensor_from_rig->params.data());
      }
    }
  }
}

void ParameterizePoints(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    const std::unordered_map<point3D_t, size_t>& point3D_num_observations,
    Reconstruction& reconstruction,
    ceres::Problem& problem) {
  for (const auto& [point3D_id, num_observations] : point3D_num_observations) {
    Point3D& point3D = reconstruction.Point3D(point3D_id);
    if (!options.refine_points3D || point3D.track.Length() > num_observations) {
      problem.SetParameterBlockConstant(point3D.xyz.data());
    }
  }

  for (const point3D_t point3D_id : config.ConstantPoints()) {
    Point3D& point3D = reconstruction.Point3D(point3D_id);
    problem.SetParameterBlockConstant(point3D.xyz.data());
  }
}

class DefaultBundleAdjuster : public CeresBundleAdjuster {
 public:
  DefaultBundleAdjuster(const BundleAdjustmentOptions& options,
                        const BundleAdjustmentConfig& config,
                        Reconstruction& reconstruction)
      : CeresBundleAdjuster(options, config),
        loss_function_(options_.ceres->CreateLossFunction()) {
    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem_ = std::make_shared<ceres::Problem>(problem_options);

    // Verify that reconstruction is internally consistent.
    THROW_CHECK(reconstruction.IsValid());

    // Set up problem.
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
    ParameterizeRigsAndFrames(
        options_, config_, parameterized_image_ids_, reconstruction, *problem_);
    ParameterizePoints(options_,
                       config_,
                       point3D_num_observations_,
                       reconstruction,
                       *problem_);

    switch (config_.FixedGauge()) {
      case BundleAdjustmentGauge::UNSPECIFIED:
        break;
      case BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD:
        FixGaugeWithTwoCamsFromWorld(options_,
                                     config_,
                                     parameterized_image_ids_,
                                     point3D_num_observations_,
                                     reconstruction,
                                     *problem_);
        break;
      case BundleAdjustmentGauge::THREE_POINTS:
        FixGaugeWithThreePoints(
            point3D_num_observations_, reconstruction, *problem_);
        break;
      default:
        LOG(FATAL_THROW) << "Unknown BundleAdjustmentGauge";
    }
  }

  std::shared_ptr<BundleAdjustmentSummary> Solve() override {
    if (problem_->NumResiduals() == 0) {
      return std::make_shared<BundleAdjustmentSummary>();
    }

    const ceres::Solver::Options solver_options =
        options_.ceres->CreateSolverOptions(config_, *problem_);

    ceres::Solver::Summary ceres_summary;
    ceres::Solve(solver_options, problem_.get(), &ceres_summary);

    if (options_.print_summary || VLOG_IS_ON(1)) {
      PrintSolverSummary(ceres_summary, "Bundle adjustment report");
    }

    return CeresBundleAdjustmentSummary::Create(ceres_summary);
  }

  std::shared_ptr<ceres::Problem>& Problem() override { return problem_; }

  const std::set<image_t>& ParameterizedImageIds() const {
    return parameterized_image_ids_;
  }

  void AddImageToProblem(const image_t image_id,
                         Reconstruction& reconstruction) {
    Image& image = reconstruction.Image(image_id);

    if (image.IsRefInFrame()) {
      AddImageWithTrivialFrame(image, reconstruction);
    } else {
      AddImageWithNonTrivialFrame(image, reconstruction);
    }
  }

  void AddImageWithTrivialFrame(Image& image, Reconstruction& reconstruction) {
    Camera& camera = *image.CameraPtr();

    const bool constant_cam_from_world =
        !options_.refine_rig_from_world ||
        config_.HasConstantRigFromWorldPose(image.FrameId());

    THROW_CHECK(image.IsRefInFrame());
    Rigid3d& rig_from_world = image.FramePtr()->RigFromWorld();

    // Add residuals to bundle adjustment problem.
    size_t num_observations = 0;
    for (const Point2D& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D() || config_.IsIgnoredPoint(point2D.point3D_id)) {
        continue;
      }

      Point3D& point3D = reconstruction.Point3D(point2D.point3D_id);
      THROW_CHECK_GT(point3D.track.Length(), 1);

      // Skip points with track length below minimum.
      if (options_.min_track_length > 0 &&
          static_cast<int>(point3D.track.Length()) <
              options_.min_track_length) {
        continue;
      }

      num_observations += 1;
      point3D_num_observations_[point2D.point3D_id] += 1;

      if (constant_cam_from_world) {
        problem_->AddResidualBlock(
            CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor>(
                camera.model_id, point2D.xy, rig_from_world),
            loss_function_.get(),
            point3D.xyz.data(),
            camera.params.data());
      } else {
        problem_->AddResidualBlock(
            CreateCameraCostFunction<ReprojErrorCostFunctor>(camera.model_id,
                                                             point2D.xy),
            loss_function_.get(),
            point3D.xyz.data(),
            rig_from_world.params.data(),
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

    const bool constant_sensor_from_rig =
        !options_.refine_sensor_from_rig ||
        config_.HasConstantSensorFromRigPose(sensor_id);
    const bool constant_rig_from_world =
        !options_.refine_rig_from_world ||
        config_.HasConstantRigFromWorldPose(image.FrameId());

    THROW_CHECK(!image.IsRefInFrame());
    Rigid3d& sensor_from_rig =
        image.FramePtr()->RigPtr()->SensorFromRig(sensor_id);
    Rigid3d& rig_from_world = image.FramePtr()->RigFromWorld();
    const std::optional<Rigid3d> cam_from_world =
        (constant_sensor_from_rig && constant_rig_from_world)
            ? std::make_optional<Rigid3d>(sensor_from_rig * rig_from_world)
            : std::nullopt;

    // Add residuals to bundle adjustment problem.
    size_t num_observations = 0;
    for (const Point2D& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D() || config_.IsIgnoredPoint(point2D.point3D_id)) {
        continue;
      }

      Point3D& point3D = reconstruction.Point3D(point2D.point3D_id);
      THROW_CHECK_GT(point3D.track.Length(), 1);

      // Skip points with track length below minimum.
      if (options_.min_track_length > 0 &&
          static_cast<int>(point3D.track.Length()) <
              options_.min_track_length) {
        continue;
      }

      num_observations += 1;
      point3D_num_observations_[point2D.point3D_id] += 1;

      // The !constant_sensor_from_rig && constant_rig_from_world is
      // rare enough that we do not have a specialized cost function for it.
      if (constant_sensor_from_rig && constant_rig_from_world) {
        problem_->AddResidualBlock(
            CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor>(
                camera.model_id, point2D.xy, cam_from_world.value()),
            loss_function_.get(),
            point3D.xyz.data(),
            camera.params.data());
      } else if (!constant_rig_from_world && constant_sensor_from_rig) {
        problem_->AddResidualBlock(
            CreateCameraCostFunction<RigReprojErrorConstantRigCostFunctor>(
                camera.model_id, point2D.xy, sensor_from_rig),
            loss_function_.get(),
            point3D.xyz.data(),
            rig_from_world.params.data(),
            camera.params.data());
      } else {
        problem_->AddResidualBlock(
            CreateCameraCostFunction<RigReprojErrorCostFunctor>(camera.model_id,
                                                                point2D.xy),
            loss_function_.get(),
            point3D.xyz.data(),
            sensor_from_rig.params.data(),
            rig_from_world.params.data(),
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
    THROW_CHECK(!config_.IsIgnoredPoint(point3D_id));
    Point3D& point3D = reconstruction.Point3D(point3D_id);

    // Skip points with track length below minimum.
    if (options_.min_track_length > 0 &&
        static_cast<int>(point3D.track.Length()) < options_.min_track_length) {
      return;
    }

    size_t& num_observations = point3D_num_observations_[point3D_id];

    // Is 3D point already fully contained in the problem? I.e. its entire
    // track is contained in `variable_image_ids`, `constant_image_ids`,
    // `constant_x_image_ids`.
    if (num_observations == point3D.track.Length()) {
      return;
    }

    for (const auto& track_el : point3D.track.Elements()) {
      // Skip observations that were already added in `FillImages`.
      if (config_.HasImage(track_el.image_id)) {
        continue;
      }

      num_observations += 1;

      Image& image = reconstruction.Image(track_el.image_id);
      Camera& camera = *image.CameraPtr();
      const Point2D& point2D = image.Point2D(track_el.point2D_idx);

      if (image.IsRefInFrame()) {
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

class PosePriorBundleAdjuster : public CeresBundleAdjuster {
 public:
  PosePriorBundleAdjuster(const BundleAdjustmentOptions& options,
                          const PosePriorBundleAdjustmentOptions& prior_options,
                          const BundleAdjustmentConfig& config,
                          std::vector<PosePrior> pose_priors,
                          Reconstruction& reconstruction)
      : CeresBundleAdjuster(options, config),
        prior_options_(prior_options),
        pose_priors_(std::move(pose_priors)),
        reconstruction_(reconstruction) {
    THROW_CHECK(prior_options_.Check());

    // Filter irrelevant pose priors.
    pose_priors_.erase(
        std::remove_if(pose_priors_.begin(),
                       pose_priors_.end(),
                       [this](const auto& pose_prior) {
                         return !pose_prior.HasPosition() ||
                                pose_prior.corr_data_id.sensor_id.type !=
                                    SensorType::CAMERA ||
                                !config_.HasImage(pose_prior.corr_data_id.id);
                       }),
        pose_priors_.end());

    const bool use_prior_position = AlignReconstruction();

    // Fix 7-DOFs of BA problem if not enough valid pose priors.
    if (use_prior_position) {
      // Normalize the reconstruction to avoid any numerical instability but
      // do not transform priors as they will be transformed when added to
      // ceres::Problem.
      normalized_from_metric_ = reconstruction_.Normalize(/*fixed_scale=*/true);
    } else {
      config_.FixGauge(BundleAdjustmentGauge::THREE_POINTS);
    }

    // WARNING: Do not move this above the reconstruction normalization.
    default_bundle_adjuster_ = std::make_unique<DefaultBundleAdjuster>(
        options_, config_, reconstruction);

    if (use_prior_position) {
      prior_loss_function_ = CreateLossFunction(
          prior_options_.ceres->prior_position_loss_function_type,
          prior_options_.ceres->prior_position_loss_scale);

      // Only consider parameterized images for pose priors. Notice that some
      // images may be configured to be included in the BA problem but have no
      // reprojection constraints, etc.
      const std::set<image_t>& parameterized_image_ids =
          default_bundle_adjuster_->ParameterizedImageIds();
      for (const auto& pose_prior : pose_priors_) {
        if (parameterized_image_ids.count(pose_prior.corr_data_id.id) > 0) {
          AddImagePosePriorToProblem(
              pose_prior.corr_data_id.id, pose_prior, reconstruction);
        }
      }
    }
  }

  std::shared_ptr<BundleAdjustmentSummary> Solve() override {
    std::shared_ptr<ceres::Problem> problem =
        default_bundle_adjuster_->Problem();
    if (problem->NumResiduals() == 0) {
      return std::make_shared<BundleAdjustmentSummary>();
    }

    const ceres::Solver::Options solver_options =
        options_.ceres->CreateSolverOptions(config_, *problem);

    ceres::Solver::Summary ceres_summary;
    ceres::Solve(solver_options, problem.get(), &ceres_summary);

    reconstruction_.Transform(Inverse(normalized_from_metric_));

    if (options_.print_summary || VLOG_IS_ON(1)) {
      PrintSolverSummary(ceres_summary, "Pose Prior Bundle adjustment report");
    }

    return CeresBundleAdjustmentSummary::Create(ceres_summary);
  }

  std::shared_ptr<ceres::Problem>& Problem() override {
    return default_bundle_adjuster_->Problem();
  }

  void AddImagePosePriorToProblem(image_t image_id,
                                  const PosePrior& pose_prior,
                                  Reconstruction& reconstruction) {
    Image& image = reconstruction.Image(image_id);

    const bool constant_sensor_from_rig =
        !options_.refine_sensor_from_rig ||
        config_.HasConstantSensorFromRigPose(image.CameraPtr()->SensorId());
    const bool constant_rig_from_world =
        !options_.refine_rig_from_world ||
        config_.HasConstantRigFromWorldPose(image.FrameId());
    if (constant_sensor_from_rig && constant_rig_from_world) {
      return;
    }

    ceres::Problem& problem = *default_bundle_adjuster_->Problem();
    Frame& frame = *image.FramePtr();

    Rigid3d& rig_from_world = frame.RigFromWorld();

    const Eigen::Vector3d normalized_position =
        normalized_from_metric_ * pose_prior.position;
    const Eigen::Matrix3d normalized_from_metric_scaled_rotation =
        normalized_from_metric_.scale() *
        normalized_from_metric_.rotation().toRotationMatrix();
    const Eigen::Matrix3d position_cov =
        pose_prior.HasPositionCov()
            ? pose_prior.position_covariance
            : (prior_options_.prior_position_fallback_stddev *
               prior_options_.prior_position_fallback_stddev *
               Eigen::Matrix3d::Identity());
    const Eigen::Matrix3d normalized_position_cov =
        normalized_from_metric_scaled_rotation * position_cov *
        normalized_from_metric_scaled_rotation.transpose();

    if (image.IsRefInFrame()) {
      problem.AddResidualBlock(
          CovarianceWeightedCostFunctor<AbsolutePosePositionPriorCostFunctor>::
              Create(normalized_position_cov, normalized_position),
          prior_loss_function_.get(),
          rig_from_world.params.data());
    } else {
      Rigid3d& cam_from_rig =
          frame.RigPtr()->SensorFromRig(image.CameraPtr()->SensorId());
      problem.AddResidualBlock(
          CovarianceWeightedCostFunctor<
              AbsoluteRigPosePositionPriorCostFunctor>::
              Create(normalized_position_cov, normalized_position),
          prior_loss_function_.get(),
          cam_from_rig.params.data(),
          rig_from_world.params.data());
    }
  }

  bool AlignReconstruction() {
    RANSACOptions ransac_options = prior_options_.alignment_ransac_options;
    if (ransac_options.max_error <= 0) {
      std::vector<double> rms_vars;
      rms_vars.reserve(pose_priors_.size());
      for (const auto& pose_prior : pose_priors_) {
        const double trace = pose_prior.position_covariance.trace();
        if (trace <= 0.0) {
          continue;
        }
        rms_vars.push_back(trace / 3.0);
      }

      if (rms_vars.empty()) {
        LOG(WARNING) << "No pose priors with valid covariance found.";
        rms_vars.push_back(prior_options_.prior_position_fallback_stddev *
                           prior_options_.prior_position_fallback_stddev);
      }

      // Set max error using the median RMS variance of valid pose priors.
      // Scaled by sqrt(chi-square 95% quantile, 3 DOF) to approximate a 95%
      // confidence radius.
      ransac_options.max_error =
          std::sqrt(kChiSquare95ThreeDof * Median(rms_vars));
    }

    VLOG(2) << "Robustly aligning reconstruction with max_error="
            << ransac_options.max_error;

    Sim3d metric_from_orig;
    if (!AlignReconstructionToPosePriors(
            reconstruction_, pose_priors_, ransac_options, &metric_from_orig)) {
      LOG(WARNING) << "Alignment w.r.t. prior positions failed";
      return false;
    }
    reconstruction_.Transform(metric_from_orig);

    // Compute alignment error w.r.t. prior positions.
    if (VLOG_IS_ON(2)) {
      std::vector<double> verr2_wrt_prior;
      verr2_wrt_prior.reserve(config_.NumImages());
      for (const auto& pose_prior : pose_priors_) {
        const auto& image = reconstruction_.Image(pose_prior.corr_data_id.id);
        verr2_wrt_prior.push_back(
            (image.ProjectionCenter() - pose_prior.position).squaredNorm());
      }
      VLOG(2) << "Alignment error w.r.t. prior positions:\n"
              << "  - rmse:   " << std::sqrt(Mean(verr2_wrt_prior)) << '\n'
              << "  - median: " << std::sqrt(Median(verr2_wrt_prior)) << '\n';
    }

    return true;
  }

 private:
  PosePriorBundleAdjustmentOptions prior_options_;
  std::vector<PosePrior> pose_priors_;
  Reconstruction& reconstruction_;

  std::unique_ptr<DefaultBundleAdjuster> default_bundle_adjuster_;
  std::unique_ptr<ceres::LossFunction> prior_loss_function_;

  Sim3d normalized_from_metric_;
};

}  // namespace

std::unique_ptr<BundleAdjuster> CreateDefaultCeresBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    Reconstruction& reconstruction) {
  return std::make_unique<DefaultBundleAdjuster>(
      options, config, reconstruction);
}

std::unique_ptr<BundleAdjuster> CreatePosePriorCeresBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const PosePriorBundleAdjustmentOptions& prior_options,
    const BundleAdjustmentConfig& config,
    std::vector<PosePrior> pose_priors,
    Reconstruction& reconstruction) {
  return std::make_unique<PosePriorBundleAdjuster>(
      options, prior_options, config, std::move(pose_priors), reconstruction);
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
  log << std::right << ceres::TerminationTypeToString(summary.termination_type)
      << "\n\n";
  LOG(INFO) << log.str();
}

}  // namespace colmap

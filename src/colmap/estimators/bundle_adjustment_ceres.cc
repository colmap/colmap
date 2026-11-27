// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/bundle_adjustment_ceres.h"

#include "colmap/estimators/alignment.h"
#include "colmap/estimators/ceres_loss_function.h"
#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/pose_prior.h"
#include "colmap/estimators/cost_functions/reprojection_error.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/util/cuda.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <cmath>
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

}  // namespace

std::shared_ptr<CeresBundleAdjustmentSummary>
CeresBundleAdjustmentSummary::Create(ceres::Solver::Summary ceres_summary) {
  auto summary = std::make_shared<CeresBundleAdjustmentSummary>();
  summary->termination_type =
      CeresTerminationTypeToTerminationType(ceres_summary.termination_type);
  summary->num_residuals = ceres_summary.num_residuals_reduced;
  summary->ceres_summary = std::move(ceres_summary);
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
  // Bound the trust region so that LM keeps a minimum damping of
  // clamp(diag(J^T J), min_lm_diagonal, max_lm_diagonal) /
  // max_trust_region_radius. This regularizes directions that the problem
  // leaves unconstrained (e.g., the gauge when it is not fixed) and keeps the
  // free scale from sliding into a collapsed (zero-baseline) solution in
  // low-parallax problems.
  solver_options.max_trust_region_radius = 1e8;
  solver_options.num_threads = -1;
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads = -1;
#endif  // CERES_VERSION_MAJOR
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
  const bool cuda_solver_requested =
      use_gpu && num_images >= min_num_images_gpu_solver;
  const bool use_cuda_solver = cuda_solver_requested && GetNumCudaDevices() > 0;
  if (cuda_solver_requested && !use_cuda_solver) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but no CUDA GPU is "
           "available. Falling back to CPU-based solvers.";
  }

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 2)) && \
    !defined(CERES_NO_CUDA)
  if (use_cuda_solver) {
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
  if (use_cuda_solver) {
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
  CHECK_OPTION(IsValidCeresLossFunction(
      loss_function_type, loss_function_scale, loss_function_weight));
  CHECK_OPTION_GT(prior_position_loss_scale, 0);
  CHECK_OPTION(IsValidCeresLossFunction(prior_position_loss_function_type,
                                        prior_position_loss_scale));
  CHECK_OPTION_LT(max_num_images_direct_dense_cpu_solver,
                  max_num_images_direct_sparse_cpu_solver);
  CHECK_OPTION_LT(max_num_images_direct_dense_gpu_solver,
                  max_num_images_direct_sparse_gpu_solver);
  return true;
}

bool CeresPosePriorBundleAdjustmentOptions::Check() const {
  CHECK_OPTION_GT(prior_position_loss_scale, 0);
  CHECK_OPTION(IsValidCeresLossFunction(prior_position_loss_function_type,
                                        prior_position_loss_scale));
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
      const_camera_params.reserve(camera.params.size());

      {
        // Metadata parameters (e.g. the (w, h) image dimensions of spherical
        // models) are sensor properties and are never optimized.
        const span<const size_t> params_idxs = camera.MetaDataParamsIdxs();
        const_camera_params.insert(
            const_camera_params.end(), params_idxs.begin(), params_idxs.end());
      }
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

      if (const_camera_params.size() == camera.params.size()) {
        problem.SetParameterBlockConstant(camera.params.data());
      } else if (!const_camera_params.empty()) {
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
  FlatHashSet<rig_t> parameterized_rig_ids;
  FlatHashSet<sensor_t> parameterized_sensor_ids;
  FlatHashSet<frame_t> parameterized_frame_ids;
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
    const FlatHashMap<point3D_t, size_t>& point3D_num_observations,
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

std::shared_ptr<CeresBundleAdjustmentSummary> CreateSummaryAndLogFailure(
    ceres::Solver::Summary ceres_summary, const std::string& context) {
  auto summary = CeresBundleAdjustmentSummary::Create(std::move(ceres_summary));
  if (!summary->IsSolutionUsable()) {
    LOG(ERROR) << context << " failed: " << summary->ceres_summary.message;
  }
  return summary;
}

class CancellationCallback : public ceres::IterationCallback {
 public:
  explicit CancellationCallback(std::function<bool()> check_if_stopped)
      : check_if_stopped_(std::move(check_if_stopped)) {}

  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary&) override {
    return check_if_stopped_ && check_if_stopped_()
               ? ceres::SOLVER_TERMINATE_SUCCESSFULLY
               : ceres::SOLVER_CONTINUE;
  }

 private:
  std::function<bool()> check_if_stopped_;
};

ceres::Solver::Summary SolveWithGpuFallback(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    ceres::Problem* problem) {
  CancellationCallback cancellation_callback(options.check_if_stopped);
  ceres::Solver::Options solver_options =
      options.ceres->CreateSolverOptions(config, *problem);
  if (options.check_if_stopped) {
    solver_options.callbacks.push_back(&cancellation_callback);
  }

  ceres::Solver::Summary ceres_summary;
  ceres::Solve(solver_options, problem, &ceres_summary);

  if (ceres_summary.termination_type == ceres::FAILURE &&
      options.ceres->use_gpu) {
    const std::string& msg = ceres_summary.message;
    if (msg.find("CUDA initialization failed") != std::string::npos ||
        msg.find("non-numeric") != std::string::npos ||
        msg.find("Unable to create Jacobian") != std::string::npos) {
      LOG(WARNING) << "GPU bundle adjustment failed (" << msg
                   << "), retrying with CPU.";
      auto cpu_options =
          std::make_shared<CeresBundleAdjustmentOptions>(*options.ceres);
      cpu_options->use_gpu = false;
      ceres::Solver::Options cpu_solver_options =
          cpu_options->CreateSolverOptions(config, *problem);
      if (options.check_if_stopped) {
        cpu_solver_options.callbacks.push_back(&cancellation_callback);
      }
      ceres::Solve(cpu_solver_options, problem, &ceres_summary);
    }
  }

  return ceres_summary;
}

bool HasVariableRigFromWorldPose(const BundleAdjustmentOptions& options,
                                 const BundleAdjustmentConfig& config,
                                 const Reconstruction& reconstruction) {
  return options.refine_rig_from_world &&
         !options.constant_rig_from_world_rotation &&
         std::any_of(config.Images().begin(),
                     config.Images().end(),
                     [&](const image_t image_id) {
                       return !config.HasConstantRigFromWorldPose(
                           reconstruction.Image(image_id).FrameId());
                     });
}

bool HasVariableCameraPose(const Image& image,
                           const BundleAdjustmentConfig& config,
                           const BundleAdjustmentOptions& options) {
  const bool constant_sensor_from_rig =
      image.IsRefInFrame() || !options.refine_sensor_from_rig ||
      config.HasConstantSensorFromRigPose(image.CameraPtr()->SensorId());
  const bool constant_rig_from_world =
      !options.refine_rig_from_world ||
      config.HasConstantRigFromWorldPose(image.FrameId());
  return !constant_sensor_from_rig || !constant_rig_from_world;
}

std::vector<PosePrior> FilterPosePriorsInProblem(
    std::vector<PosePrior> pose_priors,
    const BundleAdjustmentConfig& config,
    const Eigen::Vector3d& position_stddev,
    const bool overwrite_position_covariance) {
  // Convert WGS84 priors to ENU before filtering so the ENU origin is
  // deterministic regardless of which subset of images is optimized.
  ConvertPosePriorsToENU(pose_priors);

  pose_priors.erase(
      std::remove_if(pose_priors.begin(),
                     pose_priors.end(),
                     [&](const PosePrior& pose_prior) {
                       return !pose_prior.HasPosition() ||
                              pose_prior.corr_data_id.sensor_id.type !=
                                  SensorType::CAMERA ||
                              !config.HasImage(pose_prior.corr_data_id.id);
                     }),
      pose_priors.end());

  SetPosePriorPositionCovariances(
      position_stddev, overwrite_position_covariance, pose_priors);
  return pose_priors;
}

bool AlignReconstruction(const RANSACOptions& ransac_options,
                         const std::vector<PosePrior>& pose_priors,
                         Reconstruction& reconstruction) {
  Sim3d metric_from_orig;
  if (!AlignReconstructionToPosePriors(
          reconstruction, pose_priors, ransac_options, &metric_from_orig)) {
    LOG(WARNING) << "Alignment w.r.t. prior positions failed";
    return false;
  }
  reconstruction.Transform(metric_from_orig);

  // Compute alignment error w.r.t. prior positions.
  if (VLOG_IS_ON(2)) {
    std::vector<double> verr2_wrt_prior;
    verr2_wrt_prior.reserve(pose_priors.size());
    for (const auto& pose_prior : pose_priors) {
      const auto& image = reconstruction.Image(pose_prior.corr_data_id.id);
      verr2_wrt_prior.push_back(
          (image.ProjectionCenter() - pose_prior.position).squaredNorm());
    }
    VLOG(2) << "Alignment error w.r.t. prior positions:\n"
            << "  - rmse:   " << std::sqrt(Mean(verr2_wrt_prior)) << '\n'
            << "  - median: " << std::sqrt(Median(verr2_wrt_prior)) << '\n';
  }

  return true;
}

template <typename CostFunctor>
ceres::CostFunction* CreatePositionPriorCostFunction(
    const PosePrior& pose_prior, const Sim3d& normalized_from_metric) {
  const Eigen::Vector3d normalized_position =
      normalized_from_metric * pose_prior.position;
  const Eigen::Matrix3d normalized_from_metric_scaled_rotation =
      normalized_from_metric.scale() *
      normalized_from_metric.rotation().toRotationMatrix();
  return CovarianceWeightedCostFunctor<CostFunctor>::Create(
      normalized_from_metric_scaled_rotation * pose_prior.position_covariance *
          normalized_from_metric_scaled_rotation.transpose(),
      normalized_position);
}

struct RestoreMetricGuard {
  Reconstruction& reconstruction;
  const std::optional<Sim3d>& normalized_from_metric;
  ~RestoreMetricGuard() {
    if (normalized_from_metric.has_value()) {
      reconstruction.Transform(Inverse(*normalized_from_metric));
    }
  }
};

class DefaultBundleAdjuster : public CeresBundleAdjuster {
 public:
  DefaultBundleAdjuster(const BundleAdjustmentOptions& options,
                        const BundleAdjustmentConfig& config,
                        Reconstruction& reconstruction,
                        const BundleAdjustmentPriors& priors)
      : CeresBundleAdjuster(options, config),
        reconstruction_(reconstruction),
        loss_function_(
            CreateCeresLossFunction(options_.ceres->loss_function_type,
                                    options_.ceres->loss_function_scale,
                                    options_.ceres->loss_function_weight)) {
    VLOG(2) << "Creating Ceres bundle adjuster";

    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem_ = std::make_shared<ceres::Problem>(problem_options);

    // Verify that reconstruction is internally consistent.
    THROW_CHECK(reconstruction_.IsValid());

    const std::vector<PosePrior> active_pose_priors =
        PreparePosePositionPriors(priors.pose_priors);
    if (!active_pose_priors.empty()) {
      // Center the reconstruction while constructing residuals to avoid
      // numerical instability at large metric coordinate offsets (e.g. UTM);
      // RestoreMetricGuard restores metric coordinates before returning.
      normalized_from_metric_ = reconstruction_.Normalize(/*fixed_scale=*/true);
    }
    const RestoreMetricGuard restore_guard{reconstruction_,
                                           normalized_from_metric_};

    // Set up problem.
    // Warning: AddPointsToProblem assumes that AddImageToProblem is called
    // first. Do not change order of instructions!
    for (const image_t image_id : config_.Images()) {
      AddImageToProblem(image_id);
    }
    for (const auto point3D_id : config_.VariablePoints()) {
      AddPointToProblem(point3D_id);
    }
    for (const auto point3D_id : config_.ConstantPoints()) {
      AddPointToProblem(point3D_id);
    }

    AddPosePositionPriorsToProblem(active_pose_priors);

    ParameterizeCameras(options_,
                        config_,
                        parameterized_camera_ids_,
                        reconstruction_,
                        *problem_);
    ParameterizeRigsAndFrames(options_,
                              config_,
                              parameterized_image_ids_,
                              reconstruction_,
                              *problem_);
    ParameterizePoints(options_,
                       config_,
                       point3D_num_observations_,
                       reconstruction_,
                       *problem_);
  }

  std::shared_ptr<BundleAdjustmentSummary> Solve() override {
    if (problem_->NumResiduals() == 0) {
      return std::make_shared<BundleAdjustmentSummary>();
    }

    if (normalized_from_metric_.has_value()) {
      reconstruction_.Transform(*normalized_from_metric_);
    }
    const RestoreMetricGuard restore_guard{reconstruction_,
                                           normalized_from_metric_};

    ceres::Solver::Summary ceres_summary =
        SolveWithGpuFallback(options_, config_, problem_.get());

    if (options_.print_summary || VLOG_IS_ON(1)) {
      PrintSolverSummary(ceres_summary, "Bundle adjustment report");
    }

    return CreateSummaryAndLogFailure(std::move(ceres_summary),
                                      "Bundle adjustment");
  }

  std::shared_ptr<ceres::Problem>& Problem() override { return problem_; }

  std::optional<Sim3d> NormalizedFromMetric() const override {
    return normalized_from_metric_;
  }

 private:
  void AddImageToProblem(const image_t image_id) {
    Image& image = reconstruction_.Image(image_id);

    if (image.IsRefInFrame()) {
      AddImageWithTrivialFrame(image);
    } else {
      AddImageWithNonTrivialFrame(image);
    }
  }

  void AddImageWithTrivialFrame(Image& image) {
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

      Point3D& point3D = reconstruction_.Point3D(point2D.point3D_id);
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

  void AddImageWithNonTrivialFrame(Image& image) {
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

      Point3D& point3D = reconstruction_.Point3D(point2D.point3D_id);
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

  void AddPointToProblem(const point3D_t point3D_id) {
    THROW_CHECK(!config_.IsIgnoredPoint(point3D_id));
    Point3D& point3D = reconstruction_.Point3D(point3D_id);

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

      Image& image = reconstruction_.Image(track_el.image_id);
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

  std::vector<PosePrior> PreparePosePositionPriors(
      const std::vector<PosePrior>& input_pose_priors) {
    if (!options_.pose_prior.use_position) {
      return {};
    }

    const bool has_variable_rig_from_world =
        options_.refine_rig_from_world &&
        !options_.constant_rig_from_world_rotation &&
        std::any_of(config_.Images().begin(),
                    config_.Images().end(),
                    [this](const image_t image_id) {
                      return !config_.HasConstantRigFromWorldPose(
                          reconstruction_.Image(image_id).FrameId());
                    });
    if (!has_variable_rig_from_world) {
      LOG_IF(WARNING, options_.constant_rig_from_world_rotation)
          << "Position priors require variable rig_from_world rotations; "
             "ignoring position priors.";
      return {};
    }

    std::vector<PosePrior> pose_priors = FilterPosePriorsInProblem(
        input_pose_priors,
        config_,
        options_.pose_prior.position_stddev,
        options_.pose_prior.overwrite_position_covariance);

    const auto variable_priors_end = std::stable_partition(
        pose_priors.begin(),
        pose_priors.end(),
        [this](const PosePrior& pose_prior) {
          return HasVariableCameraPose(
              reconstruction_.Image(pose_prior.corr_data_id.id),
              config_,
              options_);
        });
    if (variable_priors_end == pose_priors.begin()) {
      return {};
    }

    if (pose_priors.size() < 3 ||
        !AlignReconstruction(options_.pose_prior.position_alignment_ransac,
                             pose_priors,
                             reconstruction_)) {
      return {};
    }

    pose_priors.erase(variable_priors_end, pose_priors.end());
    return pose_priors;
  }

  void AddPosePositionPriorsToProblem(
      const std::vector<PosePrior>& active_pose_priors) {
    prior_position_loss_function_ = CreateCeresLossFunction(
        options_.ceres->prior_position_loss_function_type,
        options_.ceres->prior_position_loss_scale);

    for (const auto& pose_prior : active_pose_priors) {
      Image& image = reconstruction_.Image(pose_prior.corr_data_id.id);
      Frame& frame = *image.FramePtr();
      Rigid3d& rig_from_world = frame.RigFromWorld();

      if (image.IsRefInFrame()) {
        problem_->AddResidualBlock(CreatePositionPriorCostFunction<
                                       AbsolutePosePositionPriorCostFunctor>(
                                       pose_prior, *normalized_from_metric_),
                                   prior_position_loss_function_.get(),
                                   rig_from_world.params.data());
      } else {
        Rigid3d& cam_from_rig =
            frame.RigPtr()->SensorFromRig(image.CameraPtr()->SensorId());
        problem_->AddResidualBlock(CreatePositionPriorCostFunction<
                                       AbsoluteRigPosePositionPriorCostFunctor>(
                                       pose_prior, *normalized_from_metric_),
                                   prior_position_loss_function_.get(),
                                   cam_from_rig.params.data(),
                                   rig_from_world.params.data());
      }
      parameterized_image_ids_.insert(pose_prior.corr_data_id.id);
    }
  }

  Reconstruction& reconstruction_;
  std::optional<Sim3d> normalized_from_metric_;
  std::shared_ptr<ceres::Problem> problem_;
  std::unique_ptr<ceres::LossFunction> loss_function_;
  std::unique_ptr<ceres::LossFunction> prior_position_loss_function_;

  std::set<camera_t> parameterized_camera_ids_;
  std::set<image_t> parameterized_image_ids_;
  FlatHashMap<point3D_t, size_t> point3D_num_observations_;
};

}  // namespace

std::unique_ptr<CeresBundleAdjuster> CreateDefaultCeresBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    Reconstruction& reconstruction,
    const BundleAdjustmentPriors& priors) {
  return std::make_unique<DefaultBundleAdjuster>(
      options, config, reconstruction, priors);
}

std::unique_ptr<CeresBundleAdjuster> CreatePosePriorCeresBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const PosePriorBundleAdjustmentOptions& prior_options,
    const BundleAdjustmentConfig& config,
    std::vector<PosePrior> pose_priors,
    Reconstruction& reconstruction) {
  THROW_CHECK(prior_options.Check());
  BundleAdjustmentOptions merged_options = options;
  merged_options.pose_prior.use_position = true;
  merged_options.pose_prior.position_stddev =
      Eigen::Vector3d::Constant(prior_options.prior_position_fallback_stddev);
  merged_options.pose_prior.overwrite_position_covariance = false;
  merged_options.pose_prior.position_alignment_ransac =
      prior_options.alignment_ransac_options;
  THROW_CHECK_NOTNULL(merged_options.ceres)->prior_position_loss_function_type =
      prior_options.ceres->prior_position_loss_function_type;
  merged_options.ceres->prior_position_loss_scale =
      prior_options.ceres->prior_position_loss_scale;
  BundleAdjustmentPriors priors;
  priors.pose_priors = std::move(pose_priors);
  return CreateDefaultCeresBundleAdjuster(
      merged_options, config, reconstruction, priors);
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

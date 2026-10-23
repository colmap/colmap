// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/global_positioning.h"

#include "colmap/estimators/cost_functions/motion_averaging.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/math/random.h"
#include "colmap/util/cuda.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <Eigen/Cholesky>

namespace colmap {
namespace {

Eigen::Vector3d RandVector3d(double low, double high) {
  return Eigen::Vector3d(RandomUniformReal(low, high),
                         RandomUniformReal(low, high),
                         RandomUniformReal(low, high));
}

std::optional<Eigen::Matrix3d> ComputeObservationCovariance(
    const CamRayWithJac& ray,
    const Eigen::Quaterniond& cam_from_world,
    const double pixel_stddev) {
  const Eigen::Matrix2d gram = ray.jacobian.transpose() * ray.jacobian;
  // The propagated covariance is rank deficient along the bearing, where
  // BATA's free scale absorbs point distance. Match the radial precision
  // to the mean tangent precision for numerical stability of the optimization.
  const double radial_variance = 2.0 / gram.inverse().trace();
  if (!(radial_variance > 0.0) || !std::isfinite(radial_variance)) {
    return std::nullopt;
  }
  const Eigen::Matrix3d camera_covariance =
      pixel_stddev * pixel_stddev *
      (ray.jacobian * ray.jacobian.transpose() +
       radial_variance * ray.ray * ray.ray.transpose());
  const Eigen::Matrix3d rotation = cam_from_world.toRotationMatrix();
  return rotation.transpose() * camera_covariance * rotation;
}

template <typename CostFunctor, typename... Args>
ceres::CostFunction* CreateBATACostFunction(
    const std::optional<Eigen::Matrix3d>& covariance, Args&&... args) {
  if (!covariance) {
    return CostFunctor::Create(std::forward<Args>(args)...);
  }
  return CovarianceWeightedCostFunctor<CostFunctor>::Create(
      *covariance, std::forward<Args>(args)...);
}

class DefaultGlobalPositioner final : public GlobalPositioner {
 public:
  DefaultGlobalPositioner(const GlobalPositionerOptions& options,
                          Reconstruction& reconstruction,
                          ObservationScaleMap& observation_scales,
                          std::shared_ptr<ceres::LossFunction> loss_function)
      : GlobalPositioner(options) {
    Prepare(reconstruction, observation_scales, std::move(loss_function));
    options_.solver_options.num_threads =
        GetEffectiveNumThreads(options_.solver_options.num_threads);
    options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  }
};

}  // namespace

bool GlobalPositionerOptions::Check() const {
  CHECK_OPTION(std::isfinite(uncalibrated_observation_weight));
  CHECK_OPTION_GT(uncalibrated_observation_weight, 0.0);
  if (experimental_observation_stddev.has_value()) {
    CHECK_OPTION_GT(*experimental_observation_stddev, 0.0);
  }
  return true;
}

GlobalPositioner::GlobalPositioner(const GlobalPositionerOptions& options)
    : options_(options) {
  THROW_CHECK(options_.Check());
  if (options_.random_seed >= 0) {
    SetPRNGSeed(static_cast<unsigned>(options_.random_seed));
  }
}

ceres::Solver::Summary GlobalPositioner::Solve() {
  LOG(INFO) << "Solving the global positioner problem";

  ceres::Solver::Summary summary;
  ceres::Solve(options_.solver_options, problem_.get(), &summary);
  Finalize(summary);
  return summary;
}

bool GlobalPositioner::Finalize(const ceres::Solver::Summary& summary) {
  if (VLOG_IS_ON(2)) {
    LOG(INFO) << summary.FullReport();
  } else {
    LOG(INFO) << summary.BriefReport();
  }
  ConvertBackResults(*reconstruction_);
  return summary.IsSolutionUsable();
}

void GlobalPositioner::Prepare(
    Reconstruction& reconstruction,
    ObservationScaleMap& observation_scales,
    std::shared_ptr<ceres::LossFunction> loss_function) {
  reconstruction_ = &reconstruction;
  observation_scales_ = &observation_scales;

  LOG(INFO) << "Setting up the global positioner problem";

  // Setup the problem.
  SetupProblem(std::move(loss_function));

  // Add the point to camera constraints to the problem.
  AddPointToCameraConstraints(reconstruction);

  if (options_.use_parameter_block_ordering) {
    AddCamerasAndPointsToParameterGroups(reconstruction);
  }

  // Parameterize the variables, set image poses / tracks / scales to be
  // constant if desired
  ParameterizeVariables(reconstruction);
}

void GlobalPositioner::SetupProblem(
    std::shared_ptr<ceres::LossFunction> loss_function) {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  if (loss_function != nullptr) {
    loss_function_ = std::move(loss_function);
  } else {
    loss_function_ = options_.CreateLossFunction();
  }

  // Clear temporary storage from previous runs.
  frame_centers_.clear();
  cams_in_rig_.clear();
}

void GlobalPositioner::AddPointToCameraConstraints(
    Reconstruction& reconstruction) {
  VLOG(2) << reconstruction.NumPoints3D()
          << " point to camera constraints were added to the position "
             "estimation problem.";

  // Reweight uncalibrated cameras.
  loss_function_ptcam_uncalibrated_ = std::make_shared<ceres::ScaledLoss>(
      loss_function_.get(),
      options_.uncalibrated_observation_weight,
      ceres::DO_NOT_TAKE_OWNERSHIP);
  loss_function_ptcam_calibrated_ = loss_function_;

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <
        static_cast<size_t>(options_.min_num_view_per_track)) {
      continue;
    }

    AddPoint3DToProblem(point3D_id, reconstruction);
  }
}

void GlobalPositioner::AddPoint3DToProblem(point3D_t point3D_id,
                                           Reconstruction& reconstruction) {
  Point3D& point3D = reconstruction.Point3D(point3D_id);

  // For each view in the track add the point to camera correspondences.
  for (const auto& observation : point3D.track.Elements()) {
    if (!reconstruction.ExistsImage(observation.image_id)) continue;

    Image& image = reconstruction.Image(observation.image_id);
    if (!image.HasPose()) continue;

    const Camera& camera = *image.CameraPtr();
    const Eigen::Vector2d& pixel = image.Point2D(observation.point2D_idx).xy;
    std::optional<Eigen::Vector3d> cam_ray;
    std::optional<Eigen::Matrix3d> covariance;
    if (options_.experimental_observation_stddev) {
      const auto ray = camera.CamRayFromImgWithJac(pixel);
      if (ray) {
        covariance = ComputeObservationCovariance(
            *ray,
            image.CamFromWorld().rotation(),
            *options_.experimental_observation_stddev);
        if (covariance) {
          cam_ray = ray->ray;
        }
      }
    } else {
      cam_ray = camera.CamRayFromImg(pixel);
    }
    if (!cam_ray.has_value()) {
      LOG(WARNING)
          << "Ignoring feature with invalid ray or covariance: point3D_id="
          << point3D_id << ", image_id=" << observation.image_id
          << ", feature_id=" << observation.point2D_idx;
      continue;
    }

    const Eigen::Vector3d cam_from_point3D_dir =
        image.CamFromWorld().rotation().inverse() * (*cam_ray);

    const auto entry = observation_scales_->find(observation);
    THROW_CHECK(entry != observation_scales_->end())
        << "missing observation scale: " << observation;
    double& scale = entry->second;
    THROW_CHECK(std::isfinite(scale) && scale >= 1e-5)
        << "observation scale must be finite and at least 1e-5: "
        << observation;
    frame_centers_.try_emplace(
        image.FrameId(), image.FramePtr()->RigFromWorld().TgtOriginInSrc());

    // For calibrated and uncalibrated cameras, use different loss
    // functions
    ceres::LossFunction* loss_function =
        (camera.HasPriorFocalLength())
            ? loss_function_ptcam_calibrated_.get()
            : loss_function_ptcam_uncalibrated_.get();

    // If the image is not part of a camera rig, use the standard BATA error
    if (image.IsRefInFrame()) {
      ceres::CostFunction* cost_function =
          CreateBATACostFunction<BATAPairwiseDirectionCostFunctor>(
              covariance, cam_from_point3D_dir);

      problem_->AddResidualBlock(cost_function,
                                 loss_function,
                                 frame_centers_[image.FrameId()].data(),
                                 point3D.xyz.data(),
                                 &scale);
    } else {
      // If the image is part of a camera rig, use the RigBATA error.

      const rig_t rig_id = image.FramePtr()->RigId();
      Rig& rig = reconstruction.Rig(rig_id);
      Rigid3d& cam_from_rig = rig.SensorFromRig(image.CameraPtr()->SensorId());

      if (!cam_from_rig.translation().hasNaN()) {
        const Eigen::Vector3d cam_from_rig_dir =
            image.CamFromWorld().rotation().inverse() *
            cam_from_rig.translation();

        ceres::CostFunction* cost_function = CreateBATACostFunction<
            RigBATAPairwiseDirectionConstantRigCostFunctor>(
            covariance, cam_from_point3D_dir, cam_from_rig_dir);

        problem_->AddResidualBlock(cost_function,
                                   loss_function,
                                   point3D.xyz.data(),
                                   frame_centers_[image.FrameId()].data(),
                                   &scale);
      } else {
        // NaN translation means the sensor's cam_from_rig must be
        // re-estimated, which requires refine_sensor_from_rig=true.
        THROW_CHECK(options_.refine_sensor_from_rig)
            << "sensor_from_rig has NaN translation but "
               "refine_sensor_from_rig=false (image_id="
            << observation.image_id << ")";
        const sensor_t sensor_id = image.CameraPtr()->SensorId();
        if (cams_in_rig_.find(sensor_id) == cams_in_rig_.end()) {
          // Will be initialized to random values in ParameterizeVariables().
          cams_in_rig_[sensor_id] = Eigen::Vector3d::Zero();
        }

        ceres::CostFunction* cost_function =
            CreateBATACostFunction<RigBATAPairwiseDirectionCostFunctor>(
                covariance,
                cam_from_point3D_dir,
                image.FramePtr()->RigFromWorld().rotation());

        problem_->AddResidualBlock(cost_function,
                                   loss_function,
                                   point3D.xyz.data(),
                                   frame_centers_[image.FrameId()].data(),
                                   cams_in_rig_[sensor_id].data(),
                                   &scale);
      }
    }

    problem_->SetParameterLowerBound(&scale, 0, 1e-5);
    // Fix all scales, or the first scale to remove the gauge ambiguity.
    if (!options_.optimize_scales ||
        (options_.fix_first_scale && problem_->NumResidualBlocks() == 1)) {
      problem_->SetParameterBlockConstant(&scale);
    }
  }
}

void GlobalPositioner::AddCamerasAndPointsToParameterGroups(
    Reconstruction& reconstruction) {
  // Create a custom ordering for Schur-based problems.
  options_.solver_options.linear_solver_ordering.reset(
      new ceres::ParameterBlockOrdering);
  ceres::ParameterBlockOrdering* parameter_ordering =
      options_.solver_options.linear_solver_ordering.get();

  // Add scale parameters to group 0 (large and independent)
  for (auto& [observation, scale] : *observation_scales_) {
    if (problem_->HasParameterBlock(&scale)) {
      parameter_ordering->AddElementToGroup(&scale, 0);
    }
  }

  // Add point parameters to group 1.
  int group_id = 1;
  if (reconstruction.NumPoints3D() > 0) {
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      if (problem_->HasParameterBlock(point3D.xyz.data()))
        parameter_ordering->AddElementToGroup(
            reconstruction.Point3D(point3D_id).xyz.data(), group_id);
    }
    group_id++;
  }

  for (auto& [frame_id, center] : frame_centers_) {
    if (problem_->HasParameterBlock(center.data())) {
      parameter_ordering->AddElementToGroup(center.data(), group_id);
    }
  }

  // Add the cam_in_rig to be estimated into the parameter group
  for (auto& [sensor_id, center] : cams_in_rig_) {
    if (problem_->HasParameterBlock(center.data())) {
      parameter_ordering->AddElementToGroup(center.data(), group_id);
    }
  }
}

void GlobalPositioner::ParameterizeVariables(Reconstruction& reconstruction) {
  // For the global positioning, do not set any camera to be constant for easier
  // convergence

  // Initialize cams_in_rig_ with random values if optimizing positions.
  if (options_.optimize_positions) {
    for (auto& [sensor_id, center] : cams_in_rig_) {
      if (problem_->HasParameterBlock(center.data())) {
        center = RandVector3d(-1, 1);
      }
    }
  }

  // If not optimizing positions, set frame centers to be constant.
  if (!options_.optimize_positions) {
    for (auto& [_, center] : frame_centers_) {
      if (problem_->HasParameterBlock(center.data())) {
        problem_->SetParameterBlockConstant(center.data());
      }
    }
  }

  // If do not optimize the rotations, set the camera rotations to be constant
  if (!options_.optimize_points) {
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      if (problem_->HasParameterBlock(point3D.xyz.data())) {
        problem_->SetParameterBlockConstant(
            reconstruction.Point3D(point3D_id).xyz.data());
      }
    }
  }

#ifdef COLMAP_CUDA_ENABLED
  bool cuda_solver_enabled = false;

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 2)) && \
    !defined(CERES_NO_CUDA)
  if (options_.use_gpu &&
      reconstruction.NumImages() >=
          static_cast<size_t>(options_.min_num_images_gpu_solver)) {
    cuda_solver_enabled = true;
    options_.solver_options.dense_linear_algebra_library_type = ceres::CUDA;
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but Ceres was "
           "compiled without CUDA support. Falling back to CPU-based dense "
           "solvers.";
  }
#endif

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 3)) && \
    !defined(CERES_NO_CUDSS)
  if (options_.use_gpu &&
      reconstruction.NumImages() >=
          static_cast<size_t>(options_.min_num_images_gpu_solver)) {
    cuda_solver_enabled = true;
    options_.solver_options.sparse_linear_algebra_library_type =
        ceres::CUDA_SPARSE;
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but Ceres was "
           "compiled without cuDSS support. Falling back to CPU-based sparse "
           "solvers.";
  }
#endif

  if (cuda_solver_enabled) {
    const std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
    THROW_CHECK_GT(gpu_indices.size(), 0);
    SetBestCudaDevice(gpu_indices[0]);
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but COLMAP was "
           "compiled without CUDA support. Falling back to CPU-based "
           "solvers.";
  }
#endif  // COLMAP_CUDA_ENABLED

  // Set up the options for the solver
  // Do not use iterative solvers, for its suboptimal performance.
  // TODO: Investigate whether the direct solver should be chosen
  // adaptively based on problem scale.
  options_.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
}

void GlobalPositioner::ConvertBackResults(Reconstruction& reconstruction) {
  // Convert optimized frame centers back to rig_from_world translations.
  for (const auto& [frame_id, center] : frame_centers_) {
    Rigid3d& rig_from_world = reconstruction.Frame(frame_id).RigFromWorld();
    rig_from_world.translation() = rig_from_world.rotation() * -center;
  }

  for (const auto& [sensor_id, center] : cams_in_rig_) {
    // Find the rig containing this sensor.
    for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
      if (!rig.HasSensor(sensor_id)) {
        continue;
      }
      Rigid3d& sensor_from_rig =
          reconstruction.Rig(rig_id).SensorFromRig(sensor_id);
      sensor_from_rig.translation() = sensor_from_rig.rotation() * -center;
      break;
    }
  }
}

ceres::Problem& GlobalPositioner::Problem() { return *problem_; }

const ceres::Solver::Options& GlobalPositioner::SolverOptions() const {
  return options_.solver_options;
}

double* GlobalPositioner::FrameCenterParameterBlock(const frame_t frame_id) {
  const auto center = frame_centers_.find(frame_id);
  if (center == frame_centers_.end() ||
      !problem_->HasParameterBlock(center->second.data())) {
    return nullptr;
  }
  return center->second.data();
}

void GlobalPositioner::ExtendParameterBlockOrdering(
    const std::vector<std::pair<double*, int>>& parameter_groups) {
  const auto& ordering = options_.solver_options.linear_solver_ordering;
  THROW_CHECK_NOTNULL(ordering.get());
  for (const auto& [parameter, group] : parameter_groups) {
    if (group < 0 || !problem_->HasParameterBlock(parameter)) {
      throw std::invalid_argument("invalid parameter block group assignment");
    }
    ordering->AddElementToGroup(parameter, group);
  }

  struct ScalarUsage {
    int num_residuals = 0;
    ceres::ResidualBlockId residual = nullptr;
  };
  FlatHashMap<double*, ScalarUsage> scalar_usages;
  std::vector<double*> parameter_blocks;
  problem_->GetParameterBlocks(&parameter_blocks);
  for (double* parameter_block : parameter_blocks) {
    if (ordering->IsMember(parameter_block)) continue;
    ordering->AddElementToGroup(parameter_block, 3);
    if (problem_->ParameterBlockSize(parameter_block) == 1) {
      scalar_usages.emplace(parameter_block, ScalarUsage{});
    }
  }
  if (!scalar_usages.empty()) {
    // Count scalar uses in one pass to avoid repeated full-problem scans.
    std::vector<ceres::ResidualBlockId> residuals;
    std::vector<double*> residual_parameters;
    problem_->GetResidualBlocks(&residuals);
    for (const ceres::ResidualBlockId residual : residuals) {
      problem_->GetParameterBlocksForResidualBlock(residual,
                                                   &residual_parameters);
      for (double* parameter : residual_parameters) {
        const auto it = scalar_usages.find(parameter);
        if (it == scalar_usages.end()) continue;
        ++it->second.num_residuals;
        it->second.residual = residual;
      }
    }
    for (double* parameter : parameter_blocks) {
      const auto it = scalar_usages.find(parameter);
      if (it == scalar_usages.end() || it->second.num_residuals != 1) continue;
      problem_->GetParameterBlocksForResidualBlock(it->second.residual,
                                                   &residual_parameters);
      if (std::none_of(
              residual_parameters.begin(),
              residual_parameters.end(),
              [&](double* other) { return ordering->GroupId(other) == 0; })) {
        ordering->AddElementToGroup(parameter, 0);
      }
    }
  }
}

std::unique_ptr<GlobalPositioner> GlobalPositioner::CreateDefault(
    const GlobalPositionerOptions& options,
    Reconstruction& reconstruction,
    ObservationScaleMap& observation_scales,
    std::shared_ptr<ceres::LossFunction> loss_function) {
  return std::make_unique<DefaultGlobalPositioner>(
      options, reconstruction, observation_scales, std::move(loss_function));
}

void RandomizeReconstructionPositions(const GlobalPositionerOptions& options,
                                      const PoseGraph& pose_graph,
                                      Reconstruction& reconstruction) {
  if (options.random_seed >= 0) {
    SetPRNGSeed(static_cast<unsigned>(options.random_seed));
  }
  FlatHashSet<frame_t> constrained_positions;
  constrained_positions.reserve(reconstruction.NumFrames());
  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    constrained_positions.insert(reconstruction.Image(image_id1).FrameId());
    constrained_positions.insert(reconstruction.Image(image_id2).FrameId());
  }

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <
        static_cast<size_t>(options.min_num_view_per_track)) {
      continue;
    }
    for (const auto& observation : point3D.track.Elements()) {
      THROW_CHECK(reconstruction.ExistsImage(observation.image_id));
      const Image& image = reconstruction.Image(observation.image_id);
      if (!image.HasPose()) continue;
      constrained_positions.insert(image.FrameId());
    }
  }

  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (constrained_positions.find(frame_id) == constrained_positions.end()) {
      continue;
    }
    auto& pose = reconstruction.Frame(frame_id).RigFromWorld();
    if (options.generate_random_positions && options.optimize_positions) {
      const Eigen::Vector3d center = 100.0 * RandVector3d(-1, 1);
      pose.translation() = -(pose.rotation() * center);
    }
  }

  VLOG(2) << "Constrained positions: " << constrained_positions.size();

  const bool random_initialization =
      options.optimize_points && options.generate_random_points;

  // Only set the points to be random if they are needed to be optimized
  if (random_initialization) {
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      if (point3D.track.Length() <
          static_cast<size_t>(options.min_num_view_per_track)) {
        continue;
      }
      reconstruction.Point3D(point3D_id).xyz = 100.0 * RandVector3d(-1, 1);
    }
  }
}

ObservationScaleMap GenerateObservationScales(
    const Reconstruction& reconstruction, bool from_geometry) {
  size_t total_observations = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    total_observations += point3D.track.Length();
  }
  ObservationScaleMap scales;
  scales.reserve(total_observations);
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    for (const auto& observation : point3D.track.Elements()) {
      double scale = 1.0;
      if (from_geometry) {
        if (!reconstruction.ExistsImage(observation.image_id)) continue;
        const Image& image = reconstruction.Image(observation.image_id);
        if (!image.HasPose()) continue;

        const Camera& camera = *image.CameraPtr();
        const Eigen::Vector2d& pixel =
            image.Point2D(observation.point2D_idx).xy;
        const auto cam_ray = camera.CamRayFromImg(pixel);
        if (!cam_ray.has_value()) continue;
        const Eigen::Vector3d cam_from_point3D_dir =
            image.CamFromWorld().rotation().inverse() * (*cam_ray);
        Eigen::Vector3d cam_from_point3D_translation =
            point3D.xyz - image.FramePtr()->RigFromWorld().TgtOriginInSrc();
        if (!image.IsRefInFrame()) {
          const Rigid3d& cam_from_rig =
              image.FramePtr()->RigPtr()->SensorFromRig(camera.SensorId());
          if (!cam_from_rig.translation().hasNaN()) {
            // Match the fixed-rig residual's camera-to-point displacement.
            cam_from_point3D_translation +=
                image.CamFromWorld().rotation().inverse() *
                cam_from_rig.translation();
          }
        }
        scale =
            std::max(1e-5,
                     cam_from_point3D_dir.dot(cam_from_point3D_translation) /
                         cam_from_point3D_translation.squaredNorm());
      }
      scales.emplace(observation, scale);
    }
  }
  return scales;
}

bool RunGlobalPositioning(const GlobalPositionerOptions& options,
                          const PoseGraph& pose_graph,
                          Reconstruction& reconstruction,
                          ObservationScaleMap* observation_scales) {
  if (reconstruction.NumImages() == 0 || reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Failed to run global positioning for empty incomplete "
                  "reconstruction: "
               << reconstruction;
    return false;
  }
  ObservationScaleMap local_scales;
  auto positioner_options = options;
  if (observation_scales == nullptr) {
    THROW_CHECK(options.Check());
    RandomizeReconstructionPositions(options, pose_graph, reconstruction);
    local_scales = GenerateObservationScales(
        reconstruction, options.initialize_scales_from_geometry);
    observation_scales = &local_scales;
    // Continue the random sequence already started by position initialization.
    positioner_options.random_seed = -1;
  }
  auto positioner = GlobalPositioner::CreateDefault(
      positioner_options, reconstruction, *observation_scales);
  return positioner->Solve().IsSolutionUsable();
}

}  // namespace colmap

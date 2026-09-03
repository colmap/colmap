#include "colmap/estimators/global_positioning.h"

#include "colmap/estimators/cost_functions/motion_averaging.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/math/random.h"
#include "colmap/util/cuda.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <utility>

namespace colmap {
namespace {

Eigen::Vector3d RandVector3d(double low, double high) {
  return Eigen::Vector3d(RandomUniformReal(low, high),
                         RandomUniformReal(low, high),
                         RandomUniformReal(low, high));
}

template <typename CostFunctor, typename... Args>
ceres::CostFunction* CreateBATACostFunction(const Eigen::Matrix3d* covariance,
                                            Args&&... args) {
  if (covariance == nullptr) {
    return CostFunctor::Create(std::forward<Args>(args)...);
  }
  return CovarianceWeightedCostFunctor<CostFunctor>::Create(
      *covariance, std::forward<Args>(args)...);
}

class DefaultGlobalPositioner final : public GlobalPositioner {
 public:
  DefaultGlobalPositioner(
      const GlobalPositionerOptions& options,
      const PoseGraph& pose_graph,
      Reconstruction& reconstruction,
      const ObservationCovarianceMap& observation_covariances,
      std::shared_ptr<ceres::LossFunction> loss_function)
      : GlobalPositioner(options) {
    Prepare(pose_graph,
            reconstruction,
            observation_covariances,
            std::move(loss_function));
    options_.solver_options.num_threads =
        GetEffectiveNumThreads(options_.solver_options.num_threads);
    options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  }
};

}  // namespace

GlobalPositioner::GlobalPositioner(const GlobalPositionerOptions& options)
    : options_(options) {
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
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction,
    const ObservationCovarianceMap& observation_covariances,
    std::shared_ptr<ceres::LossFunction> loss_function) {
  if (reconstruction.NumImages() == 0) {
    LOG(ERROR) << "Number of images = " << reconstruction.NumImages();
    throw std::runtime_error("global positioning requires images");
  }
  if (reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Number of tracks = " << reconstruction.NumPoints3D();
    throw std::runtime_error("global positioning requires 3D points");
  }
  reconstruction_ = &reconstruction;

  LOG(INFO) << "Setting up the global positioner problem";

  // Setup the problem.
  SetupProblem(std::move(loss_function));

  // Initialize camera translations to be random.
  // Also, convert the camera pose translation to be the camera center.
  InitializeRandomPositions(pose_graph, reconstruction);

  // Add the point to camera constraints to the problem.
  AddPointToCameraConstraints(reconstruction, observation_covariances);

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

  // Allocate enough memory for the scales. One for each residual.
  // Due to possibly invalid tracks, the actual number of residuals may be
  // smaller.
  scales_.clear();
  size_t total_observations = 0;
  for (const auto& [point3D_id, point3D] : reconstruction_->Points3D()) {
    total_observations += point3D.track.Length();
  }
  scales_.reserve(total_observations);
}

void GlobalPositioner::InitializeRandomPositions(
    const PoseGraph& pose_graph, Reconstruction& reconstruction) {
  FlatHashSet<frame_t> constrained_positions;
  constrained_positions.reserve(reconstruction.NumFrames());
  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    constrained_positions.insert(reconstruction.Image(image_id1).FrameId());
    constrained_positions.insert(reconstruction.Image(image_id2).FrameId());
  }

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <
        static_cast<size_t>(options_.min_num_view_per_track)) {
      continue;
    }
    for (const auto& observation : point3D.track.Elements()) {
      THROW_CHECK(reconstruction.ExistsImage(observation.image_id));
      const Image& image = reconstruction.Image(observation.image_id);
      if (!image.HasPose()) continue;
      constrained_positions.insert(image.FrameId());
    }
  }

  // Initialize frame centers in temporary storage.
  // The reconstruction poses remain in cam_from_world convention.
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (constrained_positions.find(frame_id) == constrained_positions.end()) {
      continue;
    }
    if (options_.generate_random_positions && options_.optimize_positions) {
      frame_centers_[frame_id] = 100.0 * RandVector3d(-1, 1);
    } else {
      frame_centers_[frame_id] = frame.RigFromWorld().TgtOriginInSrc();
    }
  }

  VLOG(2) << "Constrained positions: " << constrained_positions.size();
}

void GlobalPositioner::AddPointToCameraConstraints(
    Reconstruction& reconstruction,
    const ObservationCovarianceMap& observation_covariances) {
  VLOG(2) << reconstruction.NumPoints3D()
          << " point to camera constraints were added to the position "
             "estimation problem.";

  if (options_.downweight_uncalibrated_observations) {
    loss_function_ptcam_uncalibrated_ = std::make_shared<ceres::ScaledLoss>(
        loss_function_.get(), 0.5, ceres::DO_NOT_TAKE_OWNERSHIP);
  } else {
    loss_function_ptcam_uncalibrated_ = loss_function_;
  }
  loss_function_ptcam_calibrated_ = loss_function_;

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <
        static_cast<size_t>(options_.min_num_view_per_track)) {
      continue;
    }

    AddPoint3DToProblem(point3D_id, reconstruction, observation_covariances);
  }
}

void GlobalPositioner::AddPoint3DToProblem(
    point3D_t point3D_id,
    Reconstruction& reconstruction,
    const ObservationCovarianceMap& observation_covariances) {
  const bool random_initialization =
      options_.optimize_points && options_.generate_random_points;
  Point3D& point3D = reconstruction.Point3D(point3D_id);

  // Only set the points to be random if they are needed to be optimized
  if (random_initialization) {
    point3D.xyz = 100.0 * RandVector3d(-1, 1);
  }

  // For each view in the track add the point to camera correspondences.
  const std::vector<TrackElement>& observations = point3D.track.Elements();
  for (std::size_t track_element_index = 0;
       track_element_index < observations.size();
       ++track_element_index) {
    const TrackElement& observation = observations[track_element_index];
    if (!reconstruction.ExistsImage(observation.image_id)) continue;

    Image& image = reconstruction.Image(observation.image_id);
    if (!image.HasPose()) continue;

    const std::optional<Eigen::Vector3d> cam_ray =
        image.CameraPtr()->CamRayFromImg(
            image.Point2D(observation.point2D_idx).xy);
    if (!cam_ray.has_value()) {
      LOG(WARNING)
          << "Ignoring feature because it failed to project: point3D_id="
          << point3D_id << ", image_id=" << observation.image_id
          << ", feature_id=" << observation.point2D_idx;
      continue;
    }
    const Eigen::Vector3d cam_from_point3D_dir =
        image.CamFromWorld().rotation().inverse() * (*cam_ray);
    double& scale = scales_.emplace_back(1);
    const Eigen::Matrix3d* covariance = nullptr;
    if (!observation_covariances.empty()) {
      const Point3DTrackElementKey key{
          point3D_id, static_cast<uint64_t>(track_element_index)};
      const auto covariance_it = observation_covariances.find(key);
      if (covariance_it == observation_covariances.end()) {
        throw std::invalid_argument(
            "observation covariance map is missing a reconstruction "
            "observation");
      }
      if (!covariance_it->second.allFinite()) {
        throw std::invalid_argument("observation covariance must be finite");
      }
      covariance = &covariance_it->second;
    }

    if (!options_.generate_scales) {
      const Eigen::Vector3d cam_from_point3D_translation =
          point3D.xyz - frame_centers_[image.FrameId()];
      scale = std::max(1e-5,
                       cam_from_point3D_dir.dot(cam_from_point3D_translation) /
                           cam_from_point3D_translation.squaredNorm());
    }

    // For calibrated and uncalibrated cameras, use different loss
    // functions
    // Down weight the uncalibrated cameras
    Camera& camera = reconstruction.Camera(image.CameraId());
    ceres::LossFunction* loss_function =
        (camera.has_prior_focal_length)
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

  // If do not optimize the scales, set the scales to be constant
  if (!options_.optimize_scales) {
    for (double& scale : scales_) {
      if (problem_->HasParameterBlock(&scale)) {
        problem_->SetParameterBlockConstant(&scale);
      }
    }
  } else if (options_.fix_observation_scale_gauge) {
    // Set the first scale to be constant to remove the gauge ambiguity.
    for (double& scale : scales_) {
      if (problem_->HasParameterBlock(&scale)) {
        problem_->SetParameterBlockConstant(&scale);
        break;
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

void GlobalPositioner::SetParameterBlockOrdering() {
  auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();

  for (const auto& [point3D_id, point3D] : reconstruction_->Points3D()) {
    if (problem_->HasParameterBlock(point3D.xyz.data())) {
      if (!ordering->AddElementToGroup(
              reconstruction_->Point3D(point3D_id).xyz.data(), 1)) {
        throw std::logic_error("duplicate known parameter block address");
      }
    }
  }
  for (auto& [_, center] : frame_centers_) {
    if (problem_->HasParameterBlock(center.data())) {
      if (!ordering->AddElementToGroup(center.data(), 2)) {
        throw std::logic_error("duplicate known parameter block address");
      }
    }
  }
  for (auto& [sensor_id, center] : cams_in_rig_) {
    if (problem_->HasParameterBlock(center.data())) {
      if (!ordering->AddElementToGroup(center.data(), 2)) {
        throw std::logic_error("duplicate known parameter block address");
      }
    }
  }

  std::vector<double*> parameter_blocks;
  problem_->GetParameterBlocks(&parameter_blocks);
  for (double* parameter_block : parameter_blocks) {
    if (problem_->ParameterBlockSize(parameter_block) == 1 &&
        !ordering->AddElementToGroup(parameter_block, 0)) {
      throw std::logic_error("duplicate parameter block address");
    }
  }
  if (ordering->NumElements() != problem_->NumParameterBlocks() ||
      ordering->NumElements() != static_cast<int>(parameter_blocks.size())) {
    throw std::logic_error(
        "parameter block ordering does not cover the complete problem "
        "exactly once");
  }
  options_.solver_options.linear_solver_ordering = std::move(ordering);
}

std::unique_ptr<GlobalPositioner> GlobalPositioner::CreateDefault(
    const GlobalPositionerOptions& options,
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction,
    const ObservationCovarianceMap& observation_covariances,
    std::shared_ptr<ceres::LossFunction> loss_function) {
  return std::make_unique<DefaultGlobalPositioner>(options,
                                                   pose_graph,
                                                   reconstruction,
                                                   observation_covariances,
                                                   std::move(loss_function));
}

bool RunGlobalPositioning(const GlobalPositionerOptions& options,
                          const PoseGraph& pose_graph,
                          Reconstruction& reconstruction) {
  if (reconstruction.NumImages() == 0 || reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Failed to run global positioning for empty incomplete reconstruction: " << reconstruction;
    return false;
  }
  auto positioner =
      GlobalPositioner::CreateDefault(options, pose_graph, reconstruction);
  if (options.use_parameter_block_ordering) {
    positioner->SetParameterBlockOrdering();
  }
  return positioner->Solve().IsSolutionUsable();
}

}  // namespace colmap

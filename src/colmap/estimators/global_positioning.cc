#include "colmap/estimators/global_positioning.h"

#include "colmap/estimators/alignment.h"
#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/motion_averaging.h"
#include "colmap/estimators/solvers/similarity_transform.h"
#include "colmap/math/random.h"
#include "colmap/util/cuda.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/misc.h"
#include "colmap/util/string.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <memory>

namespace colmap {
namespace {

Eigen::Vector3d RandVector3d(double low, double high) {
  return Eigen::Vector3d(RandomUniformReal(low, high),
                         RandomUniformReal(low, high),
                         RandomUniformReal(low, high));
}

// One sensor-center prior transformed to a rig-center candidate for a single
// frame:
//   sensor_center_in_rig = Inverse(sensor_from_rig).translation
//   rig_center_in_world = sensor_center_in_world
//                        - world_from_rig.rotation * sensor_center_in_rig
// A reference sensor with identity extrinsics reduces to the sensor center
// (Rigid3d::TgtOriginInSrc() on an identity sensor_from_rig is zero).
struct RigCenterCandidate {
  Eigen::Vector3d center;
  bool has_covariance;
  Eigen::Matrix3d covariance;
};

bool ResolveRigCenterCandidate(const Reconstruction& reconstruction,
                               const PosePrior& prior,
                               RigCenterCandidate* candidate) {
  if (prior.corr_data_id.sensor_id.type != SensorType::CAMERA ||
      !prior.HasPosition()) {
    return false;
  }
  const image_t image_id = static_cast<image_t>(prior.corr_data_id.id);
  if (!reconstruction.ExistsImage(image_id)) {
    return false;
  }
  const Image& image = reconstruction.Image(image_id);
  if (!image.HasPose()) {
    return false;
  }
  const Frame& frame = *image.FramePtr();
  const Rig& rig = reconstruction.Rig(frame.RigId());
  const sensor_t sensor_id = image.CameraPtr()->SensorId();
  if (!rig.HasSensor(sensor_id)) {
    return false;
  }
  // The reference sensor has no explicit SensorFromRig (fixed to identity by
  // convention); every other sensor must have a known (non-NaN) extrinsic to
  // contribute a rig-center candidate.
  Rigid3d sensor_from_rig;
  if (!rig.IsRefSensor(sensor_id)) {
    if (!rig.HasSensorFromRig(sensor_id)) {
      return false;
    }
    sensor_from_rig = rig.SensorFromRig(sensor_id);
  }
  const Eigen::Vector3d sensor_center_in_rig = sensor_from_rig.TgtOriginInSrc();
  const Eigen::Vector3d world_from_rig_rotation_applied =
      frame.RigFromWorld().rotation().inverse() * sensor_center_in_rig;
  candidate->center = prior.position - world_from_rig_rotation_applied;
  candidate->has_covariance = prior.HasPositionCov();
  if (candidate->has_covariance) {
    // Both the measured sensor center and derived rig center are expressed in
    // the same world basis. The fixed rig offset changes the mean but has an
    // identity Jacobian with respect to the measured center.
    candidate->covariance = prior.position_covariance;
  } else {
    candidate->covariance.setZero();
  }
  return true;
}

// A frame has one position constraint. Multiple camera priors for the same rig
// frame are ambiguous without an explicit multi-sensor covariance model.
FlatHashMap<frame_t, Eigen::Vector3d> BuildPosePriorRigCenterSeeds(
    const Reconstruction& reconstruction,
    const std::vector<PosePrior>& pose_priors,
    int* num_usable_priors,
    FlatHashMap<frame_t, Eigen::Matrix3d>* out_covariances = nullptr) {
  FlatHashMap<frame_t, RigCenterCandidate> candidates_per_frame;
  for (const PosePrior& prior : pose_priors) {
    RigCenterCandidate candidate;
    if (!ResolveRigCenterCandidate(reconstruction, prior, &candidate)) {
      continue;
    }
    const image_t image_id = static_cast<image_t>(prior.corr_data_id.id);
    const frame_t frame_id = reconstruction.Image(image_id).FrameId();
    THROW_CHECK(candidates_per_frame.emplace(frame_id, candidate).second)
        << "Multiple position priors map to rig frame " << frame_id;
    ++(*num_usable_priors);
  }

  FlatHashMap<frame_t, Eigen::Vector3d> seeds;
  seeds.reserve(candidates_per_frame.size());
  for (const auto& [frame_id, candidate] : candidates_per_frame) {
    seeds[frame_id] = candidate.center;
    if (out_covariances != nullptr && candidate.has_covariance) {
      (*out_covariances)[frame_id] = candidate.covariance;
    }
  }
  return seeds;
}

}  // namespace

GlobalPositioner::GlobalPositioner(const GlobalPositionerOptions& options)
    : options_(options) {
  if (options_.random_seed >= 0) {
    SetPRNGSeed(static_cast<unsigned>(options_.random_seed));
  }
}

bool GlobalPositioner::Solve(const PoseGraph& pose_graph,
                             Reconstruction& reconstruction,
                             const std::vector<PosePrior>& pose_priors,
                             PosePriorPositionSummary* summary) {
  if (reconstruction.NumImages() == 0) {
    LOG(ERROR) << "Number of images = " << reconstruction.NumImages();
    return false;
  }
  if (reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Number of tracks = " << reconstruction.NumPoints3D();
    return false;
  }

  if (options_.pose_prior_position_mode != PosePriorPositionMode::off) {
    LOG(INFO) << StringPrintf(
        "Pose prior position trust: mode=%s, standardized_radius=%.6f "
        "(fixed policy)",
        std::string(
            PosePriorPositionModeToString(options_.pose_prior_position_mode))
            .c_str(),
        kPosePriorPositionRobustRadius);
  }

  LOG(INFO) << "Setting up the global positioner problem";

  // Setup the problem.
  SetupProblem(pose_graph, reconstruction);

  PosePriorPositionSummary local_summary;
  local_summary.requested = options_.pose_prior_position_mode;
  PosePriorPositionSummary* const summary_ptr =
      summary != nullptr ? summary : &local_summary;
  *summary_ptr = PosePriorPositionSummary();
  summary_ptr->requested = options_.pose_prior_position_mode;

  // Initialize camera translations to be random (optionally seeded from pose
  // priors). Also, convert the camera pose translation to be the camera
  // center.
  InitializeRandomPositions(
      pose_graph, reconstruction, pose_priors, summary_ptr);

  // Add the point to camera constraints to the problem.
  AddPointToCameraConstraints(reconstruction);

  if (options_.use_parameter_block_ordering) {
    AddCamerasAndPointsToParameterGroups(reconstruction);
  }

  // Parameterize the variables, set image poses / tracks / scales to be
  // constant if desired
  ParameterizeVariables(reconstruction);

  LOG(INFO) << "Solving the global positioner problem";

  ceres::Solver::Summary ceres_summary;
  options_.solver_options.num_threads =
      GetEffectiveNumThreads(options_.solver_options.num_threads);
  options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  ceres::Solve(options_.solver_options, problem_.get(), &ceres_summary);

  if (VLOG_IS_ON(2)) {
    LOG(INFO) << ceres_summary.FullReport();
  } else {
    LOG(INFO) << ceres_summary.BriefReport();
  }

  Sim3d gauge_from_solver;
  if (options_.pose_prior_position_mode == PosePriorPositionMode::optimize &&
      ceres_summary.IsSolutionUsable()) {
    EngagePositionPriorOptimization(
        reconstruction, pose_priors, summary_ptr, &gauge_from_solver);
  }

  // frame_centers_/cams_in_rig_ remain in the arbitrary visual/solver scale;
  // ConvertBackResults writes them into the reconstruction's poses first...
  ConvertBackResults(reconstruction);
  // ...then, only if `optimize` actually engaged, the whole reconstruction
  // (poses and points) is transformed once into the metric pose-prior frame.
  if (summary_ptr->engaged &&
      options_.pose_prior_position_mode == PosePriorPositionMode::optimize) {
    reconstruction.Transform(gauge_from_solver);
  }
  return ceres_summary.IsSolutionUsable();
}

void GlobalPositioner::SetupProblem(const PoseGraph& pose_graph,
                                    const Reconstruction& reconstruction) {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  loss_function_ = options_.CreateLossFunction();

  // Clear temporary storage from previous runs.
  frame_centers_.clear();
  cams_in_rig_.clear();

  // Allocate enough memory for the scales. One for each residual.
  // Due to possibly invalid tracks, the actual number of residuals may be
  // smaller.
  scales_.clear();
  size_t total_observations = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    total_observations += point3D.track.Length();
  }
  scales_.reserve(total_observations);
}

void GlobalPositioner::InitializeRandomPositions(
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction,
    const std::vector<PosePrior>& pose_priors,
    PosePriorPositionSummary* summary) {
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
    Reconstruction& reconstruction) {
  VLOG(2) << reconstruction.NumPoints3D()
          << " point to camera constraints were added to the position "
             "estimation problem.";

  // Down-weight uncalibrated cameras.
  loss_function_ptcam_uncalibrated_ = std::make_shared<ceres::ScaledLoss>(
      loss_function_.get(), 0.5, ceres::DO_NOT_TAKE_OWNERSHIP);
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
  const bool random_initialization =
      options_.optimize_points && options_.generate_random_points;

  Point3D& point3D = reconstruction.Point3D(point3D_id);

  // Only set the points to be random if they are needed to be optimized
  if (random_initialization) {
    point3D.xyz = 100.0 * RandVector3d(-1, 1);
  }

  // For each view in the track add the point to camera correspondences.
  for (const auto& observation : point3D.track.Elements()) {
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

    CHECK_GE(scales_.capacity(), scales_.size())
        << "Not enough capacity was reserved for the scales.";
    double& scale = scales_.emplace_back(1);

    if (!options_.generate_scales && random_initialization) {
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
          BATAPairwiseDirectionCostFunctor::Create(cam_from_point3D_dir);

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

        ceres::CostFunction* cost_function =
            RigBATAPairwiseDirectionConstantRigCostFunctor::Create(
                cam_from_point3D_dir, cam_from_rig_dir);

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
            RigBATAPairwiseDirectionCostFunctor::Create(
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

void GlobalPositioner::AddCamerasAndPointsToParameterGroups(
    Reconstruction& reconstruction) {
  // Create a custom ordering for Schur-based problems.
  options_.solver_options.linear_solver_ordering.reset(
      new ceres::ParameterBlockOrdering);
  ceres::ParameterBlockOrdering* parameter_ordering =
      options_.solver_options.linear_solver_ordering.get();

  // Add scale parameters to group 0 (large and independent)
  for (double& scale : scales_) {
    parameter_ordering->AddElementToGroup(&scale, 0);
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
    for (auto& [frame_id, center] : frame_centers_) {
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
  }
  // Set the first scale to be constant to remove the gauge ambiguity.
  for (double& scale : scales_) {
    if (problem_->HasParameterBlock(&scale)) {
      problem_->SetParameterBlockConstant(&scale);
      break;
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

namespace {
double PriorResidualRMSE(const Sim3d& gauge_from_solver,
                         const std::vector<Eigen::Vector3d>& src,
                         const std::vector<Eigen::Vector3d>& tgt,
                         const std::vector<char>& inlier_mask) {
  double sum_sq = 0.0;
  int count = 0;
  for (size_t i = 0; i < src.size(); ++i) {
    if (!inlier_mask[i]) {
      continue;
    }
    sum_sq += (gauge_from_solver * src[i] - tgt[i]).squaredNorm();
    ++count;
  }
  return count > 0 ? std::sqrt(sum_sq / count) : 0.0;
}

}  // namespace

void GlobalPositioner::EngagePositionPriorOptimization(
    Reconstruction& reconstruction,
    const std::vector<PosePrior>& pose_priors,
    PosePriorPositionSummary* summary,
    Sim3d* gauge_from_solver_out) {
  FlatHashMap<frame_t, Eigen::Matrix3d> covariances;
  int num_usable_priors = 0;
  const FlatHashMap<frame_t, Eigen::Vector3d> seeds =
      BuildPosePriorRigCenterSeeds(
          reconstruction, pose_priors, &num_usable_priors, &covariances);
  summary->num_usable_priors = num_usable_priors;
  if (covariances.size() != seeds.size()) {
    LOG(INFO) << "Pose prior optimize: requested=true, engaged=false "
                 "(every registered position prior requires valid covariance)";
    return;
  }

  // Sort frame correspondences before RANSAC for deterministic fixed-seed
  // runs, and require every candidate frame to actually be a free parameter
  // block from the visual solve above.
  std::vector<frame_t> frame_ids;
  frame_ids.reserve(seeds.size());
  for (const auto& [frame_id, center] : seeds) {
    if (frame_centers_.find(frame_id) != frame_centers_.end()) {
      frame_ids.push_back(frame_id);
    }
  }
  std::sort(frame_ids.begin(), frame_ids.end());
  summary->num_covered_frames = static_cast<int>(frame_ids.size());

  constexpr int kMinCorrespondences = 3;
  if (static_cast<int>(frame_ids.size()) < kMinCorrespondences) {
    LOG(INFO) << "Pose prior optimize: requested=true, engaged=false ("
              << frame_ids.size() << " prior-covered frames, need at least "
              << kMinCorrespondences << ")";
    return;
  }

  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> tgt;
  std::vector<Eigen::Matrix3d> tgt_covariances;
  src.reserve(frame_ids.size());
  tgt.reserve(frame_ids.size());
  tgt_covariances.reserve(frame_ids.size());
  for (const frame_t frame_id : frame_ids) {
    src.push_back(frame_centers_.at(frame_id));
    tgt.push_back(seeds.at(frame_id));
    tgt_covariances.push_back(covariances.at(frame_id));
  }

  const WeightedPositionAlignmentResult alignment =
      AlignWeightedPositionCorrespondences(
          src, tgt, tgt_covariances, options_.random_seed);
  const Sim3d& gauge_from_solver = alignment.tgt_from_src;

  const int num_inliers =
      alignment.success
          ? static_cast<int>(std::count(
                alignment.inlier_mask.begin(), alignment.inlier_mask.end(), 1))
          : 0;
  summary->num_inliers = num_inliers;
  if (!alignment.success || num_inliers < kMinCorrespondences) {
    LOG(INFO) << "Pose prior optimize: requested=true, engaged=false (RANSAC "
                 "gauge fit failed or found only "
              << num_inliers << " inliers)";
    return;
  }

  summary->initial_prior_rmse =
      PriorResidualRMSE(gauge_from_solver, src, tgt, alignment.inlier_mask);

  // New free parameter blocks for the jointly-optimized gauge.
  Eigen::Vector4d gauge_rotation = gauge_from_solver.rotation().coeffs();
  Eigen::Vector3d gauge_translation = gauge_from_solver.translation();
  double gauge_scale = gauge_from_solver.scale();
  problem_->AddParameterBlock(gauge_rotation.data(), 4);
  SetManifold(
      problem_.get(), gauge_rotation.data(), CreateEigenQuaternionManifold());
  problem_->AddParameterBlock(gauge_translation.data(), 3);
  problem_->AddParameterBlock(&gauge_scale, 1);
  problem_->SetParameterLowerBound(&gauge_scale, 0, 1e-5);

  auto prior_loss_function =
      std::make_unique<ceres::CauchyLoss>(kPosePriorPositionRobustRadius);

  for (size_t i = 0; i < frame_ids.size(); ++i) {
    if (!alignment.inlier_mask[i]) {
      continue;
    }
    const frame_t frame_id = frame_ids[i];
    const Eigen::Matrix3d& cov = covariances.at(frame_id);
    ceres::CostFunction* cost_function =
        CovarianceWeightedCostFunctor<PositionPriorViaSim3CostFunctor>::Create(
            cov, tgt[i]);
    problem_->AddResidualBlock(cost_function,
                               prior_loss_function.get(),
                               frame_centers_.at(frame_id).data(),
                               gauge_rotation.data(),
                               gauge_translation.data(),
                               &gauge_scale);
  }

  const auto frame_centers_before_second_solve = frame_centers_;
  const auto cams_in_rig_before_second_solve = cams_in_rig_;
  const std::vector<double> scales_before_second_solve = scales_;
  NodeHashMap<point3D_t, Eigen::Vector3d> points_before_second_solve;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (problem_->HasParameterBlock(point3D.xyz.data())) {
      points_before_second_solve.emplace(point3D_id, point3D.xyz);
    }
  }

  LOG(INFO) << "Solving the pose-prior-constrained global positioner problem";
  ceres::Solver::Summary second_ceres_summary;
  ceres::Solver::Options second_solver_options = options_.solver_options;
  // The first solve's custom ordering predates the three newly added gauge
  // blocks. Let Ceres derive a complete ordering for the augmented problem.
  second_solver_options.linear_solver_ordering.reset();
  ceres::Solve(second_solver_options, problem_.get(), &second_ceres_summary);
  if (VLOG_IS_ON(2)) {
    LOG(INFO) << second_ceres_summary.FullReport();
  } else {
    LOG(INFO) << second_ceres_summary.BriefReport();
  }
  if (!second_ceres_summary.IsSolutionUsable()) {
    frame_centers_ = frame_centers_before_second_solve;
    cams_in_rig_ = cams_in_rig_before_second_solve;
    scales_ = scales_before_second_solve;
    for (const auto& [point3D_id, xyz] : points_before_second_solve) {
      reconstruction.Point3D(point3D_id).xyz = xyz;
    }
    LOG(WARNING) << "Pose prior optimize: requested=true, engaged=false "
                    "(the constrained solve was unusable; restored the "
                    "visual solution)";
    return;
  }

  // frame_centers_ entries are live Ceres parameter blocks: re-read them
  // (and the gauge) after the second solve rather than reusing the
  // pre-solve `src` snapshot.
  const Sim3d final_gauge_from_solver(gauge_scale,
                                      Eigen::Quaterniond(gauge_rotation(3),
                                                         gauge_rotation(0),
                                                         gauge_rotation(1),
                                                         gauge_rotation(2)),
                                      gauge_translation);
  std::vector<Eigen::Vector3d> refined_src;
  refined_src.reserve(frame_ids.size());
  for (const frame_t frame_id : frame_ids) {
    refined_src.push_back(frame_centers_.at(frame_id));
  }
  summary->final_prior_rmse = PriorResidualRMSE(
      final_gauge_from_solver, refined_src, tgt, alignment.inlier_mask);
  summary->engaged = true;
  *gauge_from_solver_out = final_gauge_from_solver;

  LOG(INFO) << StringPrintf(
      "Pose prior optimize: requested=true, engaged=true, usable "
      "priors=%d, inliers=%d, initial RMSE=%.4f, final RMSE=%.4f",
      summary->num_usable_priors,
      summary->num_inliers,
      summary->initial_prior_rmse,
      summary->final_prior_rmse);
}

bool RunGlobalPositioning(const GlobalPositionerOptions& options,
                          const PoseGraph& pose_graph,
                          Reconstruction& reconstruction,
                          const std::vector<PosePrior>& pose_priors,
                          PosePriorPositionSummary* summary) {
  GlobalPositioner positioner(options);
  return positioner.Solve(pose_graph, reconstruction, pose_priors, summary);
}

}  // namespace colmap

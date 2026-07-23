#include "colmap/estimators/global_positioning.h"

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

#include <Eigen/Eigenvalues>

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
  data_t corr_data_id;
  Eigen::Vector3d center;
  double weight;
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
  candidate->corr_data_id = prior.corr_data_id;
  candidate->center = prior.position - world_from_rig_rotation_applied;
  // Inverse trace of the position covariance as a simple, deterministic
  // weight proxy; falls back to equal weighting when unusable.
  candidate->has_covariance = prior.HasPositionCov();
  if (candidate->has_covariance) {
    // Both the measured sensor center and derived rig center are expressed in
    // the same world basis. The fixed rig offset changes the mean but has an
    // identity Jacobian with respect to the measured center.
    candidate->covariance = prior.position_covariance;
    const double trace = candidate->covariance.trace();
    candidate->weight = trace > 0.0 ? 1.0 / trace : 1.0;
  } else {
    candidate->covariance.setZero();
    candidate->weight = 1.0;
  }
  return true;
}

bool CompareByDataId(const RigCenterCandidate& lhs,
                     const RigCenterCandidate& rhs) {
  if (lhs.corr_data_id.sensor_id.type != rhs.corr_data_id.sensor_id.type) {
    return lhs.corr_data_id.sensor_id.type < rhs.corr_data_id.sensor_id.type;
  }
  if (lhs.corr_data_id.sensor_id.id != rhs.corr_data_id.sensor_id.id) {
    return lhs.corr_data_id.sensor_id.id < rhs.corr_data_id.sensor_id.id;
  }
  return lhs.corr_data_id.id < rhs.corr_data_id.id;
}

// Builds one deterministic rig-center seed per frame covered by at least one
// usable position prior. Multiple sensor priors on the same frame are
// sorted by data_t, transformed to rig centers, and combined with a
// covariance-weighted mean (falling back to an arithmetic mean when no prior
// on that frame has a usable covariance). When `out_covariances` is not
// null, it is populated with one representative rig-frame covariance per
// seed (the first sorted candidate's covariance that has one, for `optimize`
// mode's whitening; not itself combined across multiple sensors).
FlatHashMap<frame_t, Eigen::Vector3d> BuildPosePriorRigCenterSeeds(
    const Reconstruction& reconstruction,
    const std::vector<PosePrior>& pose_priors,
    int* num_usable_priors,
    FlatHashMap<frame_t, Eigen::Matrix3d>* out_covariances = nullptr) {
  FlatHashMap<frame_t, std::vector<RigCenterCandidate>> candidates_per_frame;
  for (const PosePrior& prior : pose_priors) {
    RigCenterCandidate candidate;
    if (!ResolveRigCenterCandidate(reconstruction, prior, &candidate)) {
      continue;
    }
    const image_t image_id = static_cast<image_t>(prior.corr_data_id.id);
    const frame_t frame_id = reconstruction.Image(image_id).FrameId();
    candidates_per_frame[frame_id].push_back(candidate);
    ++(*num_usable_priors);
  }

  FlatHashMap<frame_t, Eigen::Vector3d> seeds;
  seeds.reserve(candidates_per_frame.size());
  for (auto& [frame_id, candidates] : candidates_per_frame) {
    std::sort(candidates.begin(), candidates.end(), CompareByDataId);
    const bool any_usable_covariance = std::any_of(
        candidates.begin(), candidates.end(), [](const RigCenterCandidate& c) {
          return c.weight != 1.0;
        });
    Eigen::Vector3d weighted_sum = Eigen::Vector3d::Zero();
    double weight_sum = 0.0;
    for (const RigCenterCandidate& candidate : candidates) {
      const double weight = any_usable_covariance ? candidate.weight : 1.0;
      weighted_sum += weight * candidate.center;
      weight_sum += weight;
    }
    seeds[frame_id] = weighted_sum / weight_sum;

    if (out_covariances != nullptr) {
      for (const RigCenterCandidate& candidate : candidates) {
        if (candidate.has_covariance) {
          (*out_covariances)[frame_id] = candidate.covariance;
          break;
        }
      }
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
    // The Sim3 RANSAC gate (EngagePositionPriorOptimization, below) defaults
    // to reusing the loss scale and fallback stddev rather than a third
    // tuning constant, unless pose_prior_position_ransac_max_error
    // explicitly overrides it; log the value that will actually be used.
    const bool has_explicit_ransac_gate =
        options_.pose_prior_position_ransac_max_error >= 0.0;
    const double ransac_gate =
        has_explicit_ransac_gate
            ? options_.pose_prior_position_ransac_max_error
            : options_.pose_prior_position_loss_scale *
                  options_.pose_prior_position_fallback_stddev;
    LOG(INFO) << StringPrintf(
        "Pose prior position trust: mode=%s, loss_scale=%.6f, "
        "fallback_stddev=%.6f, ransac_gate=%.6f (%s)",
        std::string(
            PosePriorPositionModeToString(options_.pose_prior_position_mode))
            .c_str(),
        options_.pose_prior_position_loss_scale,
        options_.pose_prior_position_fallback_stddev,
        ransac_gate,
        has_explicit_ransac_gate ? "explicit" : "derived");
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

  // Build rig-center seeds from pose priors when `initialize` is requested.
  // These seeds only affect the starting point of otherwise-free parameter
  // blocks: no prior residual is added, and normal post-positioning
  // normalization still runs, so this is a warm start, not a metric claim.
  FlatHashMap<frame_t, Eigen::Vector3d> prior_seeds;
  if (options_.pose_prior_position_mode == PosePriorPositionMode::initialize) {
    prior_seeds = BuildPosePriorRigCenterSeeds(
        reconstruction, pose_priors, &summary->num_usable_priors);
    if (!prior_seeds.empty()) {
      Eigen::Vector3d seed_mean = Eigen::Vector3d::Zero();
      for (const auto& [frame_id, center] : prior_seeds) {
        seed_mean += center;
      }
      seed_mean /= static_cast<double>(prior_seeds.size());
      for (auto& [frame_id, center] : prior_seeds) {
        center -= seed_mean;
      }
    }
  }

  // Initialize frame centers in temporary storage.
  // The reconstruction poses remain in cam_from_world convention.
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (constrained_positions.find(frame_id) == constrained_positions.end()) {
      continue;
    }
    const auto seed_it = prior_seeds.find(frame_id);
    if (seed_it != prior_seeds.end()) {
      frame_centers_[frame_id] = seed_it->second;
      ++summary->num_covered_frames;
    } else if (options_.generate_random_positions &&
               options_.optimize_positions) {
      frame_centers_[frame_id] = 100.0 * RandVector3d(-1, 1);
      if (options_.pose_prior_position_mode ==
          PosePriorPositionMode::initialize) {
        ++summary->num_fallback_frames;
      }
    } else {
      frame_centers_[frame_id] = frame.RigFromWorld().TgtOriginInSrc();
    }
  }

  if (options_.pose_prior_position_mode == PosePriorPositionMode::initialize) {
    summary->engaged = summary->num_covered_frames > 0;
    LOG(INFO) << StringPrintf(
        "Pose prior initialize: requested=true, engaged=%s, usable "
        "priors=%d, covered frames=%d, fallback frames=%d",
        summary->engaged ? "true" : "false",
        summary->num_usable_priors,
        summary->num_covered_frames,
        summary->num_fallback_frames);
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

bool IsUsableCovariance(const Eigen::Matrix3d& covariance) {
  if (!covariance.allFinite()) {
    return false;
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(
      0.5 * (covariance + covariance.transpose()));
  return eig.info() == Eigen::Success && eig.eigenvalues().allFinite() &&
         eig.eigenvalues().minCoeff() > 1e-12;
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
  src.reserve(frame_ids.size());
  tgt.reserve(frame_ids.size());
  for (const frame_t frame_id : frame_ids) {
    src.push_back(frame_centers_.at(frame_id));
    tgt.push_back(seeds.at(frame_id));
  }

  // The RANSAC inlier threshold defaults to reusing the same 95%-chi-square-
  // derived scale as the robust loss (via the fallback stddev), rather than
  // introducing a third, unrelated tuning constant, unless the caller
  // explicitly overrides it via pose_prior_position_ransac_max_error.
  RANSACOptions ransac_options;
  const bool has_explicit_ransac_gate =
      options_.pose_prior_position_ransac_max_error >= 0.0;
  const double error_scale = has_explicit_ransac_gate
                                 ? options_.pose_prior_position_ransac_max_error
                                 : options_.pose_prior_position_loss_scale *
                                       options_.pose_prior_position_fallback_stddev;
  ransac_options.max_error = error_scale;

  Sim3d gauge_from_solver;
  const auto report =
      EstimateSim3dRobust(src, tgt, ransac_options, gauge_from_solver);

  const int num_inliers =
      report.success ? static_cast<int>(report.support.num_inliers) : 0;
  summary->num_inliers = num_inliers;
  if (!report.success || num_inliers < kMinCorrespondences) {
    LOG(INFO) << "Pose prior optimize: requested=true, engaged=false (RANSAC "
                 "gauge fit failed or found only "
              << num_inliers << " inliers)";
    return;
  }

  summary->initial_prior_rmse =
      PriorResidualRMSE(gauge_from_solver, src, tgt, report.inlier_mask);

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

  auto prior_loss_function = std::make_unique<ceres::CauchyLoss>(
      options_.pose_prior_position_loss_scale);
  const Eigen::Matrix3d fallback_covariance =
      Eigen::Matrix3d::Identity() *
      (options_.pose_prior_position_fallback_stddev *
       options_.pose_prior_position_fallback_stddev);

  for (size_t i = 0; i < frame_ids.size(); ++i) {
    if (!report.inlier_mask[i]) {
      continue;
    }
    const frame_t frame_id = frame_ids[i];
    const auto cov_it = covariances.find(frame_id);
    const Eigen::Matrix3d& cov =
        cov_it != covariances.end() && IsUsableCovariance(cov_it->second)
            ? cov_it->second
            : fallback_covariance;
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
      final_gauge_from_solver, refined_src, tgt, report.inlier_mask);
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

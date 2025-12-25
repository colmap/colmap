#include "glomap/estimators/global_positioning.h"

#include "colmap/util/cuda.h"
#include "colmap/util/misc.h"

#include "glomap/estimators/cost_function.h"

namespace glomap {
namespace {

Eigen::Vector3d RandVector3d(std::mt19937& random_generator,
                             double low,
                             double high) {
  std::uniform_real_distribution<double> distribution(low, high);
  return Eigen::Vector3d(distribution(random_generator),
                         distribution(random_generator),
                         distribution(random_generator));
}

}  // namespace

GlobalPositioner::GlobalPositioner(const GlobalPositionerOptions& options)
    : options_(options) {
  random_generator_.seed(options_.seed);
}

bool GlobalPositioner::Solve(const ViewGraph& view_graph,
                             colmap::Reconstruction& reconstruction) {
  if (reconstruction.NumRigs() > 1) {
    LOG(ERROR) << "Number of camera rigs = " << reconstruction.NumRigs();
  }
  if (reconstruction.NumImages() == 0) {
    LOG(ERROR) << "Number of images = " << reconstruction.NumImages();
    return false;
  }
  if (view_graph.image_pairs.empty() &&
      options_.constraint_type != GlobalPositionerOptions::ONLY_POINTS) {
    LOG(ERROR) << "Number of image_pairs = " << view_graph.image_pairs.size();
    return false;
  }
  if (reconstruction.NumPoints3D() == 0 &&
      options_.constraint_type != GlobalPositionerOptions::ONLY_CAMERAS) {
    LOG(ERROR) << "Number of tracks = " << reconstruction.NumPoints3D();
    return false;
  }

  LOG(INFO) << "Setting up the global positioner problem";

  // Setup the problem.
  SetupProblem(view_graph, reconstruction);

  // Initialize camera translations to be random.
  // Also, convert the camera pose translation to be the camera center.
  InitializeRandomPositions(view_graph, reconstruction);

  // Add the camera to camera constraints to the problem.
  // TODO: support the relative constraints with trivial frames to a non trivial
  // frame
  if (options_.constraint_type != GlobalPositionerOptions::ONLY_POINTS) {
    AddCameraToCameraConstraints(view_graph, reconstruction);
  }

  // Add the point to camera constraints to the problem.
  if (options_.constraint_type != GlobalPositionerOptions::ONLY_CAMERAS) {
    AddPointToCameraConstraints(reconstruction);
  }

  AddCamerasAndPointsToParameterGroups(reconstruction);

  // Parameterize the variables, set image poses / tracks / scales to be
  // constant if desired
  ParameterizeVariables(reconstruction);

  LOG(INFO) << "Solving the global positioner problem";

  ceres::Solver::Summary summary;
  options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  ceres::Solve(options_.solver_options, problem_.get(), &summary);

  if (VLOG_IS_ON(2)) {
    LOG(INFO) << summary.FullReport();
  } else {
    LOG(INFO) << summary.BriefReport();
  }

  ConvertBackResults(reconstruction);
  return summary.IsSolutionUsable();
}

void GlobalPositioner::SetupProblem(
    const ViewGraph& view_graph, const colmap::Reconstruction& reconstruction) {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  loss_function_ = options_.CreateLossFunction();

  // Allocate enough memory for the scales. One for each residual.
  // Due to possibly invalid image pairs or tracks, the actual number of
  // residuals may be smaller.
  scales_.clear();
  size_t total_observations = 0;
  for (const auto& [track_id, track] : reconstruction.Points3D()) {
    total_observations += track.track.Length();
  }
  scales_.reserve(view_graph.image_pairs.size() + total_observations);

  // Initialize the rig scales to be 1.0.
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    rig_scales_.emplace(rig_id, 1.0);
  }
}

void GlobalPositioner::InitializeRandomPositions(
    const ViewGraph& view_graph, colmap::Reconstruction& reconstruction) {
  std::unordered_set<frame_t> constrained_positions;
  constrained_positions.reserve(reconstruction.NumFrames());
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (image_pair.is_valid == false) continue;
    constrained_positions.insert(
        reconstruction.Image(image_pair.image_id1).FrameId());
    constrained_positions.insert(
        reconstruction.Image(image_pair.image_id2).FrameId());
  }

  for (const auto& [track_id, track] : reconstruction.Points3D()) {
    if (track.track.Length() < options_.min_num_view_per_track) continue;
    for (const auto& observation : track.track.Elements()) {
      if (!reconstruction.ExistsImage(observation.image_id)) continue;
      Image& image = reconstruction.Image(observation.image_id);
      if (!image.HasPose()) continue;
      constrained_positions.insert(image.FrameId());
    }
  }

  if (!options_.generate_random_positions || !options_.optimize_positions) {
    for (const auto& [frame_id, frame] : reconstruction.Frames()) {
      if (constrained_positions.find(frame_id) != constrained_positions.end()) {
        // Will be converted back to rig_from_world after the optimization.
        reconstruction.Frame(frame_id).RigFromWorld().translation =
            frame.RigFromWorld().TgtOriginInSrc();
      }
    }
    return;
  }

  // Generate random positions for the cameras centers.
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    // Only set the cameras to be random if they are needed to be optimized
    if (constrained_positions.find(frame_id) != constrained_positions.end()) {
      reconstruction.Frame(frame_id).RigFromWorld().translation =
          100.0 * RandVector3d(random_generator_, -1, 1);
    } else {
      // Will be converted back to rig_from_world after the optimization.
      reconstruction.Frame(frame_id).RigFromWorld().translation =
          frame.RigFromWorld().TgtOriginInSrc();
    }
  }

  VLOG(2) << "Constrained positions: " << constrained_positions.size();
}

void GlobalPositioner::AddCameraToCameraConstraints(
    const ViewGraph& view_graph, colmap::Reconstruction& reconstruction) {
  // For cam to cam constraint, only support the trivial frames now
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (!image.HasPose()) continue;
    if (!image.IsRefInFrame()) {
      LOG(ERROR) << "Now, only trivial frames are supported for the camera to "
                    "camera constraints";
    }
  }

  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (image_pair.is_valid == false) {
      continue;
    }

    if (!reconstruction.ExistsImage(image_pair.image_id1) ||
        !reconstruction.ExistsImage(image_pair.image_id2)) {
      continue;
    }

    Image& image1 = reconstruction.Image(image_pair.image_id1);
    Image& image2 = reconstruction.Image(image_pair.image_id2);

    CHECK_GE(scales_.capacity(), scales_.size())
        << "Not enough capacity was reserved for the scales.";
    double& scale = scales_.emplace_back(1);

    const Eigen::Vector3d translation =
        image2.CamFromWorld().rotation.inverse() *
        -image_pair.cam2_from_cam1.translation;
    ceres::CostFunction* cost_function =
        BATAPairwiseDirectionCostFunctor::Create(translation);
    problem_->AddResidualBlock(
        cost_function,
        loss_function_.get(),
        // Note that the translations are converted to the rig centers
        // during the pre-processing.
        image1.FramePtr()->RigFromWorld().translation.data(),
        image2.FramePtr()->RigFromWorld().translation.data(),
        &scale);

    problem_->SetParameterLowerBound(&scale, 0, 1e-5);
  }

  VLOG(2) << problem_->NumResidualBlocks()
          << " camera to camera constraints were added to the position "
             "estimation problem.";
}

void GlobalPositioner::AddPointToCameraConstraints(
    colmap::Reconstruction& reconstruction) {
  // The number of camera-to-camera constraints coming from the relative poses

  const size_t num_cam_to_cam = problem_->NumResidualBlocks();
  // Find the tracks that are relevant to the current set of cameras
  const size_t num_pt_to_cam = reconstruction.NumPoints3D();

  VLOG(2) << num_pt_to_cam
          << " point to camera constriants were added to the position "
             "estimation problem.";

  if (num_pt_to_cam == 0) return;

  double weight_scale_pt = 1.0;
  // Set the relative weight of the point to camera constraints based on
  // the number of camera to camera constraints.
  if (num_cam_to_cam > 0 &&
      options_.constraint_type ==
          GlobalPositionerOptions::POINTS_AND_CAMERAS_BALANCED) {
    weight_scale_pt = options_.constraint_reweight_scale *
                      static_cast<double>(num_cam_to_cam) /
                      static_cast<double>(num_pt_to_cam);
  }
  VLOG(2) << "Point to camera weight scaled: " << weight_scale_pt;

  if (loss_function_ptcam_uncalibrated_ == nullptr) {
    loss_function_ptcam_uncalibrated_ =
        std::make_shared<ceres::ScaledLoss>(loss_function_.get(),
                                            0.5 * weight_scale_pt,
                                            ceres::DO_NOT_TAKE_OWNERSHIP);
  }

  if (options_.constraint_type ==
      GlobalPositionerOptions::POINTS_AND_CAMERAS_BALANCED) {
    loss_function_ptcam_calibrated_ = std::make_shared<ceres::ScaledLoss>(
        loss_function_.get(), weight_scale_pt, ceres::DO_NOT_TAKE_OWNERSHIP);
  } else {
    loss_function_ptcam_calibrated_ = loss_function_;
  }

  for (const auto& [track_id, track] : reconstruction.Points3D()) {
    if (track.track.Length() < options_.min_num_view_per_track) continue;

    AddTrackToProblem(track_id, reconstruction);
  }
}

void GlobalPositioner::AddTrackToProblem(
    point3D_t track_id, colmap::Reconstruction& reconstruction) {
  const bool random_initialization =
      options_.optimize_points && options_.generate_random_points;

  Point3D& track = reconstruction.Point3D(track_id);

  // Only set the points to be random if they are needed to be optimized
  if (random_initialization) {
    track.xyz = 100.0 * RandVector3d(random_generator_, -1, 1);
  }

  // For each view in the track add the point to camera correspondences.
  for (const auto& observation : track.track.Elements()) {
    if (!reconstruction.ExistsImage(observation.image_id)) continue;

    Image& image = reconstruction.Image(observation.image_id);
    if (!image.HasPose()) continue;

    const std::optional<Eigen::Vector2d> cam_point =
        image.CameraPtr()->CamFromImg(
            image.Point2D(observation.point2D_idx).xy);
    if (!cam_point.has_value()) {
      LOG(WARNING) << "Ignoring feature because it failed to project: track_id="
                   << track_id << ", image_id=" << observation.image_id
                   << ", feature_id=" << observation.point2D_idx;
      continue;
    }

    const Eigen::Vector3d cam_from_point3D_dir =
        image.CamFromWorld().rotation.inverse() *
        cam_point->homogeneous().normalized();

    CHECK_GE(scales_.capacity(), scales_.size())
        << "Not enough capacity was reserved for the scales.";
    double& scale = scales_.emplace_back(1);

    if (!options_.generate_scales && random_initialization) {
      const Eigen::Vector3d cam_from_point3D_translation =
          track.xyz - image.CamFromWorld().translation;
      scale = std::max(1e-5,
                       cam_from_point3D_dir.dot(cam_from_point3D_translation) /
                           cam_from_point3D_translation.squaredNorm());
    }

    // For calibrated and uncalibrated cameras, use different loss
    // functions
    // Down weight the uncalibrated cameras
    colmap::Camera& camera = reconstruction.Camera(image.CameraId());
    ceres::LossFunction* loss_function =
        (camera.has_prior_focal_length)
            ? loss_function_ptcam_calibrated_.get()
            : loss_function_ptcam_uncalibrated_.get();

    // If the image is not part of a camera rig, use the standard BATA error
    if (image.IsRefInFrame()) {
      ceres::CostFunction* cost_function =
          BATAPairwiseDirectionCostFunctor::Create(cam_from_point3D_dir);

      problem_->AddResidualBlock(
          cost_function,
          loss_function,
          // Note that the translations are converted to the rig centers
          // during the pre-processing.
          image.FramePtr()->RigFromWorld().translation.data(),
          track.xyz.data(),
          &scale);
    } else {
      // If the image is part of a camera rig, use the RigBATA error.

      const rig_t rig_id = image.FramePtr()->RigId();
      Rig& rig = reconstruction.Rig(rig_id);
      Rigid3d& cam_from_rig = rig.SensorFromRig(image.CameraPtr()->SensorId());

      if (!cam_from_rig.translation.hasNaN()) {
        const Eigen::Vector3d cam_from_rig_dir =
            image.CamFromWorld().rotation.inverse() * cam_from_rig.translation;

        ceres::CostFunction* cost_function =
            RigBATAPairwiseDirectionCostFunctor::Create(cam_from_point3D_dir,
                                                        cam_from_rig_dir);

        problem_->AddResidualBlock(
            cost_function,
            loss_function,
            track.xyz.data(),
            // Note that the translations are converted to the rig centers
            // during the pre-processing.
            image.FramePtr()->RigFromWorld().translation.data(),
            &scale,
            &rig_scales_[rig_id]);
      } else {
        // If the cam_from_rig contains nan values, it means that it needs to be
        // re-estimated In this case, use the rigged cost NOTE: the scale for
        // the rig is not needed, as it would natrually be consistent with the
        // global one
        ceres::CostFunction* cost_function =
            RigUnknownBATAPairwiseDirectionCostFunctor::Create(
                cam_from_point3D_dir,
                image.FramePtr()->RigFromWorld().rotation);

        problem_->AddResidualBlock(
            cost_function,
            loss_function,
            track.xyz.data(),
            // Note that the translations are converted to the rig centers
            // during the pre-processing.
            image.FramePtr()->RigFromWorld().translation.data(),
            cam_from_rig.translation.data(),
            &scale);
      }
    }

    problem_->SetParameterLowerBound(&scale, 0, 1e-5);
  }
}

void GlobalPositioner::AddCamerasAndPointsToParameterGroups(
    colmap::Reconstruction& reconstruction) {
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
    for (const auto& [track_id, track] : reconstruction.Points3D()) {
      if (problem_->HasParameterBlock(
              reconstruction.Point3D(track_id).xyz.data()))
        parameter_ordering->AddElementToGroup(
            reconstruction.Point3D(track_id).xyz.data(), group_id);
    }
    group_id++;
  }

  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (!frame.HasPose()) continue;
    if (problem_->HasParameterBlock(
            reconstruction.Frame(frame_id).RigFromWorld().translation.data())) {
      parameter_ordering->AddElementToGroup(
          reconstruction.Frame(frame_id).RigFromWorld().translation.data(),
          group_id);
    }
  }

  // Add the cam_from_rigs to be estimated into the parameter group
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor_id.type == SensorType::CAMERA) {
        Eigen::Vector3d& translation =
            reconstruction.Rig(rig_id).SensorFromRig(sensor_id).translation;
        if (problem_->HasParameterBlock(translation.data())) {
          parameter_ordering->AddElementToGroup(translation.data(), group_id);
        }
      }
    }
  }

  group_id++;

  // Also add the scales to the group
  for (auto& [rig_id, scale] : rig_scales_) {
    if (problem_->HasParameterBlock(&scale))
      parameter_ordering->AddElementToGroup(&scale, group_id);
  }
}

void GlobalPositioner::ParameterizeVariables(
    colmap::Reconstruction& reconstruction) {
  // For the global positioning, do not set any camera to be constant for easier
  // convergence

  // First, for cam_from_rig that needs to be estimated, we need to initialize
  // the center
  if (options_.optimize_positions) {
    for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
      for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
        if (sensor_id.type == SensorType::CAMERA) {
          Eigen::Vector3d& translation =
              reconstruction.Rig(rig_id).SensorFromRig(sensor_id).translation;
          if (problem_->HasParameterBlock(translation.data())) {
            translation = RandVector3d(random_generator_, -1, 1);
          }
        }
      }
    }
  }

  // If do not optimize the positions, set the camera positions to be constant
  if (!options_.optimize_positions) {
    for (const auto& [frame_id, frame] : reconstruction.Frames()) {
      if (!frame.HasPose()) continue;
      if (problem_->HasParameterBlock(
              reconstruction.Frame(frame_id).RigFromWorld().translation.data()))
        problem_->SetParameterBlockConstant(
            reconstruction.Frame(frame_id).RigFromWorld().translation.data());
    }
  }

  // If do not optimize the rotations, set the camera rotations to be constant
  if (!options_.optimize_points) {
    for (const auto& [track_id, track] : reconstruction.Points3D()) {
      if (problem_->HasParameterBlock(
              reconstruction.Point3D(track_id).xyz.data())) {
        problem_->SetParameterBlockConstant(
            reconstruction.Point3D(track_id).xyz.data());
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
  // Set the first rig scale to be constant to remove the gauge ambiguity.
  for (double& scale : scales_) {
    if (problem_->HasParameterBlock(&scale)) {
      problem_->SetParameterBlockConstant(&scale);
      break;
    }
  }
  // Set the rig scales to be constant
  // TODO: add a flag to allow the scales to be optimized (if they are not in
  // metric scale)
  for (auto& [rig_id, scale] : rig_scales_) {
    if (problem_->HasParameterBlock(&scale)) {
      problem_->SetParameterBlockConstant(&scale);
    }
  }

#ifdef GLOMAP_CUDA_ENABLED
  const size_t num_images = reconstruction.NumFrames();
  bool cuda_solver_enabled = false;

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 2)) && \
    !defined(CERES_NO_CUDA)
  if (options_.use_gpu &&
      static_cast<int>(num_images) >= options_.min_num_images_gpu_solver) {
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
      static_cast<int>(num_images) >= options_.min_num_images_gpu_solver) {
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
    const std::vector<int> gpu_indices =
        colmap::CSVToVector<int>(options_.gpu_index);
    THROW_CHECK_GT(gpu_indices.size(), 0);
    colmap::SetBestCudaDevice(gpu_indices[0]);
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but COLMAP was "
           "compiled without CUDA support. Falling back to CPU-based "
           "solvers.";
  }
#endif  // GLOMAP_CUDA_ENABLED

  // Set up the options for the solver
  // Do not use iterative solvers, for its suboptimal performance.
  if (reconstruction.NumPoints3D() > 0) {
    options_.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    options_.solver_options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;
  } else {
    options_.solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options_.solver_options.preconditioner_type = ceres::JACOBI;
  }
}

void GlobalPositioner::ConvertBackResults(
    colmap::Reconstruction& reconstruction) {
  // Translations store the frame/camera centers. Convert them back
  // and apply the rig scales.

  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    reconstruction.Frame(frame_id).RigFromWorld().translation =
        reconstruction.Frame(frame_id).RigFromWorld().rotation *
        -reconstruction.Frame(frame_id).RigFromWorld().translation;
    reconstruction.Frame(frame_id).RigFromWorld().translation *=
        rig_scales_[frame.RigId()];
  }

  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    for (const auto& [sensor_id, cam_from_rig] : rig.NonRefSensors()) {
      if (cam_from_rig.has_value()) {
        Rigid3d& sensor_from_rig =
            reconstruction.Rig(rig_id).SensorFromRig(sensor_id);
        if (problem_->HasParameterBlock(sensor_from_rig.translation.data())) {
          sensor_from_rig.translation =
              sensor_from_rig.rotation * -sensor_from_rig.translation;
        } else {
          sensor_from_rig.translation *= rig_scales_[rig_id];
        }
      }
    }
  }
}

}  // namespace glomap

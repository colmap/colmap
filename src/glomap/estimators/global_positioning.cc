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

bool GlobalPositioner::Solve(
    const ViewGraph& view_graph,
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<camera_t, colmap::Camera>& cameras,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& tracks) {
  if (rigs.size() > 1) {
    LOG(ERROR) << "Number of camera rigs = " << rigs.size();
  }
  if (images.empty()) {
    LOG(ERROR) << "Number of images = " << images.size();
    return false;
  }
  if (view_graph.image_pairs.empty() &&
      options_.constraint_type != GlobalPositionerOptions::ONLY_POINTS) {
    LOG(ERROR) << "Number of image_pairs = " << view_graph.image_pairs.size();
    return false;
  }
  if (tracks.empty() &&
      options_.constraint_type != GlobalPositionerOptions::ONLY_CAMERAS) {
    LOG(ERROR) << "Number of tracks = " << tracks.size();
    return false;
  }

  LOG(INFO) << "Setting up the global positioner problem";

  // Setup the problem.
  SetupProblem(view_graph, rigs, tracks);

  // Initialize camera translations to be random.
  // Also, convert the camera pose translation to be the camera center.
  InitializeRandomPositions(view_graph, frames, images, tracks);

  // Add the camera to camera constraints to the problem.
  // TODO: support the relative constraints with trivial frames to a non trivial
  // frame
  if (options_.constraint_type != GlobalPositionerOptions::ONLY_POINTS) {
    AddCameraToCameraConstraints(view_graph, images);
  }

  // Add the point to camera constraints to the problem.
  if (options_.constraint_type != GlobalPositionerOptions::ONLY_CAMERAS) {
    AddPointToCameraConstraints(rigs, cameras, frames, images, tracks);
  }

  AddCamerasAndPointsToParameterGroups(rigs, frames, tracks);

  // Parameterize the variables, set image poses / tracks / scales to be
  // constant if desired
  ParameterizeVariables(rigs, frames, tracks);

  LOG(INFO) << "Solving the global positioner problem";

  ceres::Solver::Summary summary;
  options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  ceres::Solve(options_.solver_options, problem_.get(), &summary);

  if (VLOG_IS_ON(2)) {
    LOG(INFO) << summary.FullReport();
  } else {
    LOG(INFO) << summary.BriefReport();
  }

  ConvertBackResults(rigs, frames);
  return summary.IsSolutionUsable();
}

void GlobalPositioner::SetupProblem(
    const ViewGraph& view_graph,
    const std::unordered_map<rig_t, Rig>& rigs,
    const std::unordered_map<point3D_t, Point3D>& tracks) {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  loss_function_ = options_.CreateLossFunction();

  // Allocate enough memory for the scales. One for each residual.
  // Due to possibly invalid image pairs or tracks, the actual number of
  // residuals may be smaller.
  scales_.clear();
  scales_.reserve(
      view_graph.image_pairs.size() +
      std::accumulate(tracks.begin(),
                      tracks.end(),
                      0,
                      [](int sum, const std::pair<point3D_t, Point3D>& track) {
                        return sum + track.second.track.Length();
                      }));

  // Initialize the rig scales to be 1.0.
  for (const auto& [rig_id, rig] : rigs) {
    rig_scales_.emplace(rig_id, 1.0);
  }
}

void GlobalPositioner::InitializeRandomPositions(
    const ViewGraph& view_graph,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& tracks) {
  std::unordered_set<image_t> constrained_positions;
  constrained_positions.reserve(frames.size());
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (image_pair.is_valid == false) continue;
    constrained_positions.insert(images[image_pair.image_id1].FrameId());
    constrained_positions.insert(images[image_pair.image_id2].FrameId());
  }

  for (const auto& [track_id, track] : tracks) {
    if (track.track.Length() < options_.min_num_view_per_track) continue;
    for (const auto& observation : tracks[track_id].track.Elements()) {
      if (images.find(observation.image_id) == images.end()) continue;
      Image& image = images[observation.image_id];
      if (!image.HasPose()) continue;
      constrained_positions.insert(images[observation.image_id].FrameId());
    }
  }

  if (!options_.generate_random_positions || !options_.optimize_positions) {
    for (auto& [frame_id, frame] : frames) {
      if (constrained_positions.find(frame_id) != constrained_positions.end()) {
        // Will be converted back to rig_from_world after the optimization.
        frame.RigFromWorld().translation =
            frame.RigFromWorld().TgtOriginInSrc();
      }
    }
    return;
  }

  // Generate random positions for the cameras centers.
  for (auto& [frame_id, frame] : frames) {
    // Only set the cameras to be random if they are needed to be optimized
    if (constrained_positions.find(frame_id) != constrained_positions.end()) {
      frame.RigFromWorld().translation =
          100.0 * RandVector3d(random_generator_, -1, 1);
    } else {
      // Will be converted back to rig_from_world after the optimization.
      frame.RigFromWorld().translation = frame.RigFromWorld().TgtOriginInSrc();
    }
  }

  VLOG(2) << "Constrained positions: " << constrained_positions.size();
}

void GlobalPositioner::AddCameraToCameraConstraints(
    const ViewGraph& view_graph, std::unordered_map<image_t, Image>& images) {
  // For cam to cam constraint, only support the trivial frames now
  for (const auto& [image_id, image] : images) {
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

    const auto image1_it = images.find(image_pair.image_id1);
    const auto image2_it = images.find(image_pair.image_id2);
    if (image1_it == images.end() || image2_it == images.end()) {
      continue;
    }

    CHECK_GE(scales_.capacity(), scales_.size())
        << "Not enough capacity was reserved for the scales.";
    double& scale = scales_.emplace_back(1);

    const Eigen::Vector3d translation =
        image2_it->second.CamFromWorld().rotation.inverse() *
        -image_pair.cam2_from_cam1.translation;
    ceres::CostFunction* cost_function =
        BATAPairwiseDirectionError::Create(translation);
    problem_->AddResidualBlock(
        cost_function,
        loss_function_.get(),
        // Note that the translations are converted to the rig centers
        // during the pre-processing.
        image1_it->second.FramePtr()->RigFromWorld().translation.data(),
        image2_it->second.FramePtr()->RigFromWorld().translation.data(),
        &scale);

    problem_->SetParameterLowerBound(&scale, 0, 1e-5);
  }

  VLOG(2) << problem_->NumResidualBlocks()
          << " camera to camera constraints were added to the position "
             "estimation problem.";
}

void GlobalPositioner::AddPointToCameraConstraints(
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<camera_t, colmap::Camera>& cameras,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& tracks) {
  // The number of camera-to-camera constraints coming from the relative poses

  const size_t num_cam_to_cam = problem_->NumResidualBlocks();
  // Find the tracks that are relevant to the current set of cameras
  const size_t num_pt_to_cam = tracks.size();

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

  for (auto& [track_id, track] : tracks) {
    if (track.track.Length() < options_.min_num_view_per_track) continue;

    AddTrackToProblem(track_id, rigs, cameras, frames, images, tracks);
  }
}

void GlobalPositioner::AddTrackToProblem(
    point3D_t track_id,
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<camera_t, colmap::Camera>& cameras,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& tracks) {
  const bool random_initialization =
      options_.optimize_points && options_.generate_random_points;

  Point3D& track = tracks[track_id];

  // Only set the points to be random if they are needed to be optimized
  if (random_initialization) {
    track.xyz = 100.0 * RandVector3d(random_generator_, -1, 1);
  }

  // For each view in the track add the point to camera correspondences.
  for (const auto& observation : track.track.Elements()) {
    if (images.find(observation.image_id) == images.end()) continue;

    Image& image = images[observation.image_id];
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
    ceres::LossFunction* loss_function =
        (cameras[image.CameraId()].has_prior_focal_length)
            ? loss_function_ptcam_calibrated_.get()
            : loss_function_ptcam_uncalibrated_.get();

    // If the image is not part of a camera rig, use the standard BATA error
    if (image.IsRefInFrame()) {
      ceres::CostFunction* cost_function =
          BATAPairwiseDirectionError::Create(cam_from_point3D_dir);

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
      Rigid3d& cam_from_rig =
          rigs.at(rig_id).SensorFromRig(image.CameraPtr()->SensorId());

      if (!cam_from_rig.translation.hasNaN()) {
        const Eigen::Vector3d cam_from_rig_dir =
            image.CamFromWorld().rotation.inverse() * cam_from_rig.translation;

        ceres::CostFunction* cost_function =
            RigBATAPairwiseDirectionError::Create(cam_from_point3D_dir,
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
            RigUnknownBATAPairwiseDirectionError::Create(
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
    // std::unordered_map<image_t, Image>& images,
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<point3D_t, Point3D>& tracks) {
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
  if (tracks.size() > 0) {
    for (auto& [track_id, track] : tracks) {
      if (problem_->HasParameterBlock(track.xyz.data()))
        parameter_ordering->AddElementToGroup(track.xyz.data(), group_id);
    }
    group_id++;
  }

  for (auto& [frame_id, frame] : frames) {
    if (!frame.HasPose()) continue;
    if (problem_->HasParameterBlock(frame.RigFromWorld().translation.data())) {
      parameter_ordering->AddElementToGroup(
          frame.RigFromWorld().translation.data(), group_id);
    }
  }

  // Add the cam_from_rigs to be estimated into the parameter group
  for (auto& [rig_id, rig] : rigs) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor_id.type == SensorType::CAMERA) {
        Eigen::Vector3d& translation = rig.SensorFromRig(sensor_id).translation;
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
    // std::unordered_map<image_t, Image>& images,
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<point3D_t, Point3D>& tracks) {
  // For the global positioning, do not set any camera to be constant for easier
  // convergence

  // First, for cam_from_rig that needs to be estimated, we need to initialize
  // the center
  if (options_.optimize_positions) {
    for (auto& [rig_id, rig] : rigs) {
      for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
        if (sensor_id.type == SensorType::CAMERA) {
          Eigen::Vector3d& translation =
              rig.SensorFromRig(sensor_id).translation;
          if (problem_->HasParameterBlock(translation.data())) {
            translation = RandVector3d(random_generator_, -1, 1);
          }
        }
      }
    }
  }

  // If do not optimize the positions, set the camera positions to be constant
  if (!options_.optimize_positions) {
    for (auto& [frame_id, frame] : frames) {
      if (!frame.HasPose()) continue;
      if (problem_->HasParameterBlock(frame.RigFromWorld().translation.data()))
        problem_->SetParameterBlockConstant(
            frame.RigFromWorld().translation.data());
    }
  }

  // If do not optimize the rotations, set the camera rotations to be constant
  if (!options_.optimize_points) {
    for (auto& [track_id, track] : tracks) {
      if (problem_->HasParameterBlock(track.xyz.data())) {
        problem_->SetParameterBlockConstant(track.xyz.data());
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
  const int num_images = frames.size();
  bool cuda_solver_enabled = false;

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 2)) && \
    !defined(CERES_NO_CUDA)
  if (options_.use_gpu && num_images >= options_.min_num_images_gpu_solver) {
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
  if (options_.use_gpu && num_images >= options_.min_num_images_gpu_solver) {
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
  if (tracks.size() > 0) {
    options_.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    options_.solver_options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;
  } else {
    options_.solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options_.solver_options.preconditioner_type = ceres::JACOBI;
  }
}

void GlobalPositioner::ConvertBackResults(
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<frame_t, Frame>& frames) {
  // Translations store the frame/camera centers. Convert them back
  // and apply the rig scales.

  for (auto& [frame_id, frame] : frames) {
    frame.RigFromWorld().translation =
        frame.RigFromWorld().rotation * -frame.RigFromWorld().translation;
    frame.RigFromWorld().translation *= rig_scales_[frame.RigId()];
  }

  for (auto& [rig_id, rig] : rigs) {
    for (auto& [sensor_id, cam_from_rig] : rig.NonRefSensors()) {
      if (cam_from_rig.has_value()) {
        if (problem_->HasParameterBlock(
                rig.SensorFromRig(sensor_id).translation.data())) {
          cam_from_rig->translation =
              cam_from_rig->rotation * -cam_from_rig->translation;
        } else {
          cam_from_rig->translation *= rig_scales_[rig_id];
        }
      }
    }
  }
}

}  // namespace glomap

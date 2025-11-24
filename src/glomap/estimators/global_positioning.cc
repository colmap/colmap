#include "glomap/estimators/global_positioning.h"

#include "colmap/util/cuda.h"
#include "colmap/util/misc.h"

#include "glomap/estimators/cost_function.h"
#include "glomap/math/rigid3d.h"

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
    std::unordered_map<track_t, Track>& tracks) {
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

  ConvertResults(rigs, frames);
  return summary.IsSolutionUsable();
}

void GlobalPositioner::SetupProblem(
    const ViewGraph& view_graph,
    const std::unordered_map<rig_t, Rig>& rigs,
    const std::unordered_map<track_t, Track>& tracks) {
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
                      [](int sum, const std::pair<track_t, Track>& track) {
                        return sum + track.second.observations.size();
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
    std::unordered_map<track_t, Track>& tracks) {
  std::unordered_set<image_t> constrained_positions;
  constrained_positions.reserve(frames.size());
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (image_pair.is_valid == false) continue;
    constrained_positions.insert(images[image_pair.image_id1].frame_id);
    constrained_positions.insert(images[image_pair.image_id2].frame_id);
  }

  for (const auto& [track_id, track] : tracks) {
    if (track.observations.size() < options_.min_num_view_per_track) continue;
    for (const auto& observation : tracks[track_id].observations) {
      if (images.find(observation.first) == images.end()) continue;
      Image& image = images[observation.first];
      if (!image.IsRegistered()) continue;
      constrained_positions.insert(images[observation.first].frame_id);
    }
  }

  if (!options_.generate_random_positions || !options_.optimize_positions) {
    for (auto& [frame_id, frame] : frames) {
      if (constrained_positions.find(frame_id) != constrained_positions.end())
        frame.RigFromWorld().translation = CenterFromPose(frame.RigFromWorld());
    }
    return;
  }

  // Generate random positions for the cameras centers.
  for (auto& [frame_id, frame] : frames) {
    // Only set the cameras to be random if they are needed to be optimized
    if (constrained_positions.find(frame_id) != constrained_positions.end())
      frame.RigFromWorld().translation =
          100.0 * RandVector3d(random_generator_, -1, 1);
    else
      frame.RigFromWorld().translation = CenterFromPose(frame.RigFromWorld());
  }

  VLOG(2) << "Constrained positions: " << constrained_positions.size();
}

void GlobalPositioner::AddCameraToCameraConstraints(
    const ViewGraph& view_graph, std::unordered_map<image_t, Image>& images) {
  // For cam to cam constraint, only support the trivial frames now
  for (const auto& [image_id, image] : images) {
    if (!image.IsRegistered()) continue;
    if (!image.HasTrivialFrame()) {
      LOG(ERROR) << "Now, only trivial frames are supported for the camera to "
                    "camera constraints";
    }
  }

  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (image_pair.is_valid == false) continue;

    const image_t image_id1 = image_pair.image_id1;
    const image_t image_id2 = image_pair.image_id2;
    if (images.find(image_id1) == images.end() ||
        images.find(image_id2) == images.end()) {
      continue;
    }

    CHECK_GE(scales_.capacity(), scales_.size())
        << "Not enough capacity was reserved for the scales.";
    double& scale = scales_.emplace_back(1);

    const Eigen::Vector3d translation =
        -(images[image_id2].CamFromWorld().rotation.inverse() *
          image_pair.cam2_from_cam1.translation);
    ceres::CostFunction* cost_function =
        BATAPairwiseDirectionError::Create(translation);
    problem_->AddResidualBlock(
        cost_function,
        loss_function_.get(),
        images[image_id1].frame_ptr->RigFromWorld().translation.data(),
        images[image_id2].frame_ptr->RigFromWorld().translation.data(),
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
    std::unordered_map<track_t, Track>& tracks) {
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
    if (track.observations.size() < options_.min_num_view_per_track) continue;

    // Only set the points to be random if they are needed to be optimized
    if (options_.optimize_points && options_.generate_random_points) {
      track.xyz = 100.0 * RandVector3d(random_generator_, -1, 1);
      track.is_initialized = true;
    }

    AddTrackToProblem(track_id, rigs, cameras, frames, images, tracks);
  }
}

void GlobalPositioner::AddTrackToProblem(
    track_t track_id,
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<camera_t, colmap::Camera>& cameras,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  // For each view in the track add the point to camera correspondences.
  for (const auto& observation : tracks[track_id].observations) {
    if (images.find(observation.first) == images.end()) continue;

    Image& image = images[observation.first];
    if (!image.IsRegistered()) continue;

    const Eigen::Vector3d& feature_undist =
        image.features_undist[observation.second];
    if (feature_undist.array().isNaN().any()) {
      LOG(WARNING)
          << "Ignoring feature because it failed to undistort: track_id="
          << track_id << ", image_id=" << observation.first
          << ", feature_id=" << observation.second;
      continue;
    }

    const Eigen::Vector3d translation =
        image.CamFromWorld().rotation.inverse() *
        image.features_undist[observation.second];

    double& scale = scales_.emplace_back(1);

    if (!options_.generate_scales && tracks[track_id].is_initialized) {
      const Eigen::Vector3d trans_calc =
          tracks[track_id].xyz - image.CamFromWorld().translation;
      scale = std::max(1e-5,
                       translation.dot(trans_calc) / trans_calc.squaredNorm());
    }

    CHECK_GE(scales_.capacity(), scales_.size())
        << "Not enough capacity was reserved for the scales.";

    // For calibrated and uncalibrated cameras, use different loss
    // functions
    // Down weight the uncalibrated cameras
    ceres::LossFunction* loss_function =
        (cameras[image.camera_id].has_prior_focal_length)
            ? loss_function_ptcam_calibrated_.get()
            : loss_function_ptcam_uncalibrated_.get();

    // If the image is not part of a camera rig, use the standard BATA error
    if (image.HasTrivialFrame()) {
      ceres::CostFunction* cost_function =
          BATAPairwiseDirectionError::Create(translation);

      problem_->AddResidualBlock(
          cost_function,
          loss_function,
          image.frame_ptr->RigFromWorld().translation.data(),
          tracks[track_id].xyz.data(),
          &scale);
      // If the image is part of a camera rig, use the RigBATA error
    } else {
      rig_t rig_id = image.frame_ptr->RigId();
      // Otherwise, use the camera rig translation from the frame
      Rigid3d& cam_from_rig = rigs.at(rig_id).SensorFromRig(
          sensor_t(SensorType::CAMERA, image.camera_id));

      Eigen::Vector3d cam_from_rig_translation = cam_from_rig.translation;

      if (!cam_from_rig_translation.hasNaN()) {
        const Eigen::Vector3d translation_rig =
            // image.cam_from_world.rotation.inverse() *
            // cam_from_rig.translation;
            image.CamFromWorld().rotation.inverse() * cam_from_rig_translation;

        ceres::CostFunction* cost_function =
            RigBATAPairwiseDirectionError::Create(translation, translation_rig);

        problem_->AddResidualBlock(
            cost_function,
            loss_function,
            image.frame_ptr->RigFromWorld().translation.data(),
            tracks[track_id].xyz.data(),
            &scale,
            &rig_scales_[rig_id]);
      } else {
        // If the cam_from_rig contains nan values, it means that it needs to be
        // re-estimated In this case, use the rigged cost NOTE: the scale for
        // the rig is not needed, as it would natrually be consistent with the
        // global one
        ceres::CostFunction* cost_function =
            RigUnknownBATAPairwiseDirectionError::Create(
                translation, image.frame_ptr->RigFromWorld().rotation);

        problem_->AddResidualBlock(
            cost_function,
            loss_function,
            tracks[track_id].xyz.data(),
            image.frame_ptr->RigFromWorld().translation.data(),
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
    std::unordered_map<track_t, Track>& tracks) {
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
    std::unordered_map<track_t, Track>& tracks) {
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

  int num_images = frames.size();
#ifdef GLOMAP_CUDA_ENABLED
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

void GlobalPositioner::ConvertResults(
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<frame_t, Frame>& frames) {
  // translation now stores the camera position, needs to convert back
  for (auto& [frame_id, frame] : frames) {
    frame.RigFromWorld().translation =
        -(frame.RigFromWorld().rotation * frame.RigFromWorld().translation);

    rig_t idx_rig = frame.RigId();
    frame.RigFromWorld().translation *= rig_scales_[idx_rig];
  }

  // Update the rig scales
  for (auto& [rig_id, rig] : rigs) {
    for (auto& [sensor_id, cam_from_rig] : rig.NonRefSensors()) {
      if (cam_from_rig.has_value()) {
        if (problem_->HasParameterBlock(
                rig.SensorFromRig(sensor_id).translation.data())) {
          cam_from_rig->translation =
              -(cam_from_rig->rotation * cam_from_rig->translation);
        } else {
          // If the camera is part of a rig, then scale the translation
          // by the rig scale
          cam_from_rig->translation *= rig_scales_[rig_id];
        }
      }
    }
  }
}

}  // namespace glomap

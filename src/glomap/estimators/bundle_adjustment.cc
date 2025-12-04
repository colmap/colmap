#include "bundle_adjustment.h"

#include "colmap/estimators/cost_functions.h"
#include "colmap/estimators/manifold.h"
#include "colmap/sensor/models.h"
#include "colmap/util/cuda.h"
#include "colmap/util/misc.h"

namespace glomap {

bool BundleAdjuster::Solve(
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<camera_t, colmap::Camera>& cameras,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& tracks) {
  // Check if the input data is valid
  if (images.empty()) {
    LOG(ERROR) << "Number of images = " << images.size();
    return false;
  }
  if (tracks.empty()) {
    LOG(ERROR) << "Number of tracks = " << tracks.size();
    return false;
  }

  // Reset the problem
  Reset();

  // Add the constraints that the point tracks impose on the problem
  AddPointToCameraConstraints(rigs, cameras, frames, images, tracks);

  // Add the cameras and points to the parameter groups for schur-based
  // optimization
  AddCamerasAndPointsToParameterGroups(rigs, cameras, frames, tracks);

  // Parameterize the variables
  ParameterizeVariables(rigs, cameras, frames, tracks);

  // Set the solver options.
  ceres::Solver::Summary summary;

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

  // Do not use the iterative solver, as it does not seem to be helpful
  options_.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  options_.solver_options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;

  options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  ceres::Solve(options_.solver_options, problem_.get(), &summary);
  if (VLOG_IS_ON(2))
    LOG(INFO) << summary.FullReport();
  else
    LOG(INFO) << summary.BriefReport();

  return summary.IsSolutionUsable();
}

void BundleAdjuster::Reset() {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  loss_function_ = options_.CreateLossFunction();
}

void BundleAdjuster::AddPointToCameraConstraints(
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<camera_t, colmap::Camera>& cameras,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& tracks) {
  for (auto& [track_id, track] : tracks) {
    if (track.track.Length() < options_.min_num_view_per_track) continue;

    for (const auto& observation : tracks[track_id].track.Elements()) {
      if (images.find(observation.image_id) == images.end()) continue;

      Image& image = images[observation.image_id];
      Frame* frame_ptr = image.frame_ptr;
      const image_t rig_id = image.frame_ptr->RigId();

      ceres::CostFunction* cost_function = nullptr;
      // if (image_id_to_camera_rig_index_.find(observation.first) ==
      //     image_id_to_camera_rig_index_.end()) {
      if (image.IsReferenceInFrame()) {
        cost_function =
            colmap::CreateCameraCostFunction<colmap::ReprojErrorCostFunctor>(
                cameras[image.camera_id].model_id,
                image.features[observation.point2D_idx]);
        problem_->AddResidualBlock(
            cost_function,
            loss_function_.get(),
            frame_ptr->RigFromWorld().rotation.coeffs().data(),
            frame_ptr->RigFromWorld().translation.data(),
            tracks[track_id].xyz.data(),
            cameras[image.camera_id].params.data());
      } else if (!options_.optimize_rig_poses) {
        const Rigid3d& cam_from_rig = rigs[rig_id].SensorFromRig(
            sensor_t(SensorType::CAMERA, image.camera_id));
        cost_function = colmap::CreateCameraCostFunction<
            colmap::RigReprojErrorConstantRigCostFunctor>(
            cameras[image.camera_id].model_id,
            image.features[observation.point2D_idx],
            cam_from_rig);
        problem_->AddResidualBlock(
            cost_function,
            loss_function_.get(),
            frame_ptr->RigFromWorld().rotation.coeffs().data(),
            frame_ptr->RigFromWorld().translation.data(),
            tracks[track_id].xyz.data(),
            cameras[image.camera_id].params.data());
      } else {
        // If the image is part of a camera rig, use the RigBATA error
        // Down weight the uncalibrated cameras
        Rigid3d& cam_from_rig = rigs[rig_id].SensorFromRig(
            sensor_t(SensorType::CAMERA, image.camera_id));
        cost_function =
            colmap::CreateCameraCostFunction<colmap::RigReprojErrorCostFunctor>(
                cameras[image.camera_id].model_id,
                image.features[observation.point2D_idx]);
        problem_->AddResidualBlock(
            cost_function,
            loss_function_.get(),
            cam_from_rig.rotation.coeffs().data(),
            cam_from_rig.translation.data(),
            frame_ptr->RigFromWorld().rotation.coeffs().data(),
            frame_ptr->RigFromWorld().translation.data(),
            tracks[track_id].xyz.data(),
            cameras[image.camera_id].params.data());
      }

      if (cost_function != nullptr) {
      } else {
        LOG(ERROR) << "Camera model not supported: "
                   << colmap::CameraModelIdToName(
                          cameras[image.camera_id].model_id);
      }
    }
  }
}

void BundleAdjuster::AddCamerasAndPointsToParameterGroups(
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<camera_t, colmap::Camera>& cameras,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<point3D_t, Point3D>& tracks) {
  if (tracks.size() == 0) return;

  // Create a custom ordering for Schur-based problems.
  options_.solver_options.linear_solver_ordering.reset(
      new ceres::ParameterBlockOrdering);
  ceres::ParameterBlockOrdering* parameter_ordering =
      options_.solver_options.linear_solver_ordering.get();
  // Add point parameters to group 0.
  for (auto& [track_id, track] : tracks) {
    if (problem_->HasParameterBlock(track.xyz.data()))
      parameter_ordering->AddElementToGroup(track.xyz.data(), 0);
  }

  // Add frame parameters to group 1.
  for (auto& [frame_id, frame] : frames) {
    if (!frame.HasPose()) continue;
    if (problem_->HasParameterBlock(frame.RigFromWorld().translation.data())) {
      parameter_ordering->AddElementToGroup(
          frame.RigFromWorld().translation.data(), 1);
      parameter_ordering->AddElementToGroup(
          frame.RigFromWorld().rotation.coeffs().data(), 1);
    }
  }

  // Add the cam_from_rigs to be estimated into the parameter group
  for (auto& [rig_id, rig] : rigs) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor_id.type == SensorType::CAMERA) {
        Eigen::Vector3d& translation = rig.SensorFromRig(sensor_id).translation;
        if (problem_->HasParameterBlock(translation.data())) {
          parameter_ordering->AddElementToGroup(translation.data(), 1);
        }
        Eigen::Quaterniond& rotation = rig.SensorFromRig(sensor_id).rotation;
        if (problem_->HasParameterBlock(rotation.coeffs().data())) {
          parameter_ordering->AddElementToGroup(rotation.coeffs().data(), 1);
        }
      }
    }
  }

  // Add camera parameters to group 1.
  for (auto& [camera_id, camera] : cameras) {
    if (problem_->HasParameterBlock(camera.params.data()))
      parameter_ordering->AddElementToGroup(camera.params.data(), 1);
  }
}

void BundleAdjuster::ParameterizeVariables(
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<camera_t, colmap::Camera>& cameras,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<point3D_t, Point3D>& tracks) {
  frame_t center;

  // Parameterize rotations, and set rotations and translations to be constant
  // if desired FUTURE: Consider fix the scale of the reconstruction
  int counter = 0;
  for (auto& [frame_id, frame] : frames) {
    if (!frame.HasPose()) continue;
    if (problem_->HasParameterBlock(
            frame.RigFromWorld().rotation.coeffs().data())) {
      colmap::SetQuaternionManifold(
          problem_.get(), frame.RigFromWorld().rotation.coeffs().data());

      if (!options_.optimize_rotations || counter == 0)
        problem_->SetParameterBlockConstant(
            frame.RigFromWorld().rotation.coeffs().data());
      if (!options_.optimize_translation || counter == 0)
        problem_->SetParameterBlockConstant(
            frame.RigFromWorld().translation.data());

      counter++;
    }
  }

  // Parameterize the camera parameters, or set them to be constant if desired
  if (options_.optimize_intrinsics && !options_.optimize_principal_point) {
    for (auto& [camera_id, camera] : cameras) {
      if (problem_->HasParameterBlock(camera.params.data())) {
        std::vector<int> principal_point_idxs;
        for (auto idx : camera.PrincipalPointIdxs()) {
          principal_point_idxs.push_back(idx);
        }
        colmap::SetSubsetManifold(camera.params.size(),
                                  principal_point_idxs,
                                  problem_.get(),
                                  camera.params.data());
      }
    }
  } else if (!options_.optimize_intrinsics &&
             !options_.optimize_principal_point) {
    for (auto& [camera_id, camera] : cameras) {
      if (problem_->HasParameterBlock(camera.params.data())) {
        problem_->SetParameterBlockConstant(camera.params.data());
      }
    }
  }

  // If we optimize the rig poses, then parameterize them
  if (options_.optimize_rig_poses) {
    for (auto& [rig_id, rig] : rigs) {
      for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
        if (sensor_id.type == SensorType::CAMERA) {
          Eigen::Quaterniond& rotation = rig.SensorFromRig(sensor_id).rotation;
          if (problem_->HasParameterBlock(rotation.coeffs().data())) {
            colmap::SetQuaternionManifold(problem_.get(),
                                          rotation.coeffs().data());
          }
        }
      }
    }
  }

  if (!options_.optimize_points) {
    for (auto& [track_id, track] : tracks) {
      if (problem_->HasParameterBlock(track.xyz.data())) {
        problem_->SetParameterBlockConstant(track.xyz.data());
      }
    }
  }
}

}  // namespace glomap

#include "bundle_adjustment.h"

#include "colmap/estimators/cost_functions.h"
#include "colmap/estimators/manifold.h"
#include "colmap/sensor/models.h"
#include "colmap/util/cuda.h"
#include "colmap/util/misc.h"

namespace glomap {

bool BundleAdjuster::Solve(colmap::Reconstruction& reconstruction) {
  // Check if the input data is valid
  if (reconstruction.NumImages() == 0) {
    LOG(ERROR) << "Number of images = " << reconstruction.NumImages();
    return false;
  }
  if (reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Number of tracks = " << reconstruction.NumPoints3D();
    return false;
  }

  // Reset the problem
  Reset();

  // Add the constraints that the point tracks impose on the problem
  AddPointToCameraConstraints(reconstruction);

  // Add the cameras and points to the parameter groups for schur-based
  // optimization
  AddCamerasAndPointsToParameterGroups(reconstruction);

  // Parameterize the variables
  ParameterizeVariables(reconstruction);

  // Set the solver options.
  ceres::Solver::Summary summary;

#ifdef GLOMAP_CUDA_ENABLED
  bool cuda_solver_enabled = false;

  const size_t num_images = reconstruction.NumImages();

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
    colmap::Reconstruction& reconstruction) {
  for (const auto& [track_id, track] : reconstruction.Points3D()) {
    if (track.track.Length() < options_.min_num_view_per_track) continue;

    for (const auto& observation : track.track.Elements()) {
      THROW_CHECK(reconstruction.ExistsImage(observation.image_id));

      Image& image = reconstruction.Image(observation.image_id);
      colmap::Frame* frame = image.FramePtr();
      const rig_t rig_id = frame->RigId();
      colmap::Camera& camera = reconstruction.Camera(image.CameraId());

      ceres::CostFunction* cost_function = nullptr;
      if (image.IsRefInFrame()) {
        cost_function =
            colmap::CreateCameraCostFunction<colmap::ReprojErrorCostFunctor>(
                camera.model_id, image.Point2D(observation.point2D_idx).xy);
        problem_->AddResidualBlock(
            cost_function,
            loss_function_.get(),
            frame->RigFromWorld().rotation.coeffs().data(),
            frame->RigFromWorld().translation.data(),
            reconstruction.Point3D(track_id).xyz.data(),
            camera.params.data());
      } else if (!options_.optimize_rig_poses) {
        Rig& rig = reconstruction.Rig(rig_id);
        const Rigid3d& cam_from_rig =
            rig.SensorFromRig(sensor_t(SensorType::CAMERA, image.CameraId()));
        cost_function = colmap::CreateCameraCostFunction<
            colmap::RigReprojErrorConstantRigCostFunctor>(
            camera.model_id,
            image.Point2D(observation.point2D_idx).xy,
            cam_from_rig);
        problem_->AddResidualBlock(
            cost_function,
            loss_function_.get(),
            frame->RigFromWorld().rotation.coeffs().data(),
            frame->RigFromWorld().translation.data(),
            reconstruction.Point3D(track_id).xyz.data(),
            camera.params.data());
      } else {
        // If the image is part of a camera rig, use the RigBATA error
        // Down weight the uncalibrated cameras
        Rig& rig = reconstruction.Rig(rig_id);
        Rigid3d& cam_from_rig =
            rig.SensorFromRig(sensor_t(SensorType::CAMERA, image.CameraId()));
        cost_function =
            colmap::CreateCameraCostFunction<colmap::RigReprojErrorCostFunctor>(
                camera.model_id, image.Point2D(observation.point2D_idx).xy);
        problem_->AddResidualBlock(
            cost_function,
            loss_function_.get(),
            cam_from_rig.rotation.coeffs().data(),
            cam_from_rig.translation.data(),
            frame->RigFromWorld().rotation.coeffs().data(),
            frame->RigFromWorld().translation.data(),
            reconstruction.Point3D(track_id).xyz.data(),
            camera.params.data());
      }

      if (cost_function != nullptr) {
      } else {
        LOG(ERROR) << "Camera model not supported: "
                   << colmap::CameraModelIdToName(camera.model_id);
      }
    }
  }
}

void BundleAdjuster::AddCamerasAndPointsToParameterGroups(
    colmap::Reconstruction& reconstruction) {
  if (reconstruction.NumPoints3D() == 0) return;

  // Create a custom ordering for Schur-based problems.
  options_.solver_options.linear_solver_ordering.reset(
      new ceres::ParameterBlockOrdering);
  ceres::ParameterBlockOrdering* parameter_ordering =
      options_.solver_options.linear_solver_ordering.get();
  // Add point parameters to group 0.
  for (const auto& [track_id, track] : reconstruction.Points3D()) {
    if (problem_->HasParameterBlock(
            reconstruction.Point3D(track_id).xyz.data()))
      parameter_ordering->AddElementToGroup(
          reconstruction.Point3D(track_id).xyz.data(), 0);
  }

  // Add frame parameters to group 1.
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (!frame.HasPose()) continue;
    if (problem_->HasParameterBlock(
            reconstruction.Frame(frame_id).RigFromWorld().translation.data())) {
      parameter_ordering->AddElementToGroup(
          reconstruction.Frame(frame_id).RigFromWorld().translation.data(), 1);
      parameter_ordering->AddElementToGroup(reconstruction.Frame(frame_id)
                                                .RigFromWorld()
                                                .rotation.coeffs()
                                                .data(),
                                            1);
    }
  }

  // Add the cam_from_rigs to be estimated into the parameter group
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor_id.type == SensorType::CAMERA) {
        Eigen::Vector3d& translation =
            reconstruction.Rig(rig_id).SensorFromRig(sensor_id).translation;
        if (problem_->HasParameterBlock(translation.data())) {
          parameter_ordering->AddElementToGroup(translation.data(), 1);
        }
        Eigen::Quaterniond& rotation =
            reconstruction.Rig(rig_id).SensorFromRig(sensor_id).rotation;
        if (problem_->HasParameterBlock(rotation.coeffs().data())) {
          parameter_ordering->AddElementToGroup(rotation.coeffs().data(), 1);
        }
      }
    }
  }

  // Add camera parameters to group 1.
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    if (problem_->HasParameterBlock(
            reconstruction.Camera(camera_id).params.data()))
      parameter_ordering->AddElementToGroup(
          reconstruction.Camera(camera_id).params.data(), 1);
  }
}

void BundleAdjuster::ParameterizeVariables(
    colmap::Reconstruction& reconstruction) {
  // Parameterize rotations, and set rotations and translations to be constant
  // if desired FUTURE: Consider fix the scale of the reconstruction
  int counter = 0;
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (!frame.HasPose()) continue;
    if (problem_->HasParameterBlock(reconstruction.Frame(frame_id)
                                        .RigFromWorld()
                                        .rotation.coeffs()
                                        .data())) {
      colmap::SetQuaternionManifold(problem_.get(),
                                    reconstruction.Frame(frame_id)
                                        .RigFromWorld()
                                        .rotation.coeffs()
                                        .data());

      if (!options_.optimize_rotations || counter == 0)
        problem_->SetParameterBlockConstant(reconstruction.Frame(frame_id)
                                                .RigFromWorld()
                                                .rotation.coeffs()
                                                .data());
      if (!options_.optimize_translation || counter == 0)
        problem_->SetParameterBlockConstant(
            reconstruction.Frame(frame_id).RigFromWorld().translation.data());

      counter++;
    }
  }

  // Parameterize the camera parameters, or set them to be constant if desired
  if (options_.optimize_intrinsics && !options_.optimize_principal_point) {
    for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
      if (problem_->HasParameterBlock(
              reconstruction.Camera(camera_id).params.data())) {
        std::vector<int> principal_point_idxs;
        for (auto idx : camera.PrincipalPointIdxs()) {
          principal_point_idxs.push_back(idx);
        }
        colmap::SetSubsetManifold(
            camera.params.size(),
            principal_point_idxs,
            problem_.get(),
            reconstruction.Camera(camera_id).params.data());
      }
    }
  } else if (!options_.optimize_intrinsics &&
             !options_.optimize_principal_point) {
    for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
      if (problem_->HasParameterBlock(
              reconstruction.Camera(camera_id).params.data())) {
        problem_->SetParameterBlockConstant(
            reconstruction.Camera(camera_id).params.data());
      }
    }
  }

  // If we optimize the rig poses, then parameterize them
  if (options_.optimize_rig_poses) {
    for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
      for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
        if (sensor_id.type == SensorType::CAMERA) {
          Eigen::Quaterniond& rotation =
              reconstruction.Rig(rig_id).SensorFromRig(sensor_id).rotation;
          if (problem_->HasParameterBlock(rotation.coeffs().data())) {
            colmap::SetQuaternionManifold(problem_.get(),
                                          rotation.coeffs().data());
          }
        }
      }
    }
  }

  if (!options_.optimize_points) {
    for (const auto& [track_id, track] : reconstruction.Points3D()) {
      if (problem_->HasParameterBlock(
              reconstruction.Point3D(track_id).xyz.data())) {
        problem_->SetParameterBlockConstant(
            reconstruction.Point3D(track_id).xyz.data());
      }
    }
  }
}

}  // namespace glomap

// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/bundle_adjustment_ceres_utils.h"

#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"

#include <vector>

namespace colmap {

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

}  // namespace colmap

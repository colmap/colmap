// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/estimators/cost_functions/quaternion_utils.h"
#include "colmap/estimators/cost_functions/utils.h"

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

// Angular error in radians between two camera rotations and a relative
// rotation, expressed in the frame of sensor 1. Parameter blocks are unit
// quaternions in Eigen (x, y, z, w) order. The residual follows the right
// perturbation convention, i.e., it is the rotation vector r such that
// hat_sensor2_from_sensor1 = sensor2_from_sensor1_prior * Exp(r), so a
// covariance of the relative rotation prior must be expressed in the frame of
// sensor 1.
struct RelativeRotationCostFunctor {
  // Shape of the non-rig cost with two rotation blocks.
  static constexpr int kNumResiduals = 3;
  using kParameterDims = std::integer_sequence<int, 4, 4>;

  template <typename T>
  bool operator()(const T* const sensor1_from_world_rotation,
                  const T* const sensor2_from_world_rotation,
                  T* residuals) const {
    const T* parameters[] = {sensor1_from_world_rotation,
                             sensor2_from_world_rotation};
    return (*this)(parameters, residuals);
  }

  // Creates a cost with two blocks: sensor1_from_world_rotation, then
  // sensor2_from_world_rotation.
  static ceres::CostFunction* Create(const Eigen::Quaterniond& cam2_from_cam1) {
    return new ceres::
        AutoDiffCostFunction<RelativeRotationCostFunctor, 3, 4, 4>(
            new RelativeRotationCostFunctor{cam2_from_cam1});
  }

  // For rigs, parameters[0] and [1] are rig1_from_world and rig2_from_world.
  // sensor1_index and sensor2_index locate the sensor_from_rig blocks;
  // -1 denotes identity, and shared sensors may use the same block.
  // With same_frame=true, omit the rig_from_world blocks and index only
  // the sensor_from_rig blocks.
  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    // Compute hat_sensor2_from_sensor1 in sensor 2 / sensor 1 frames; the
    // shared world frame rotation cancels from the 3D residual vector.
    Eigen::Quaternion<T> error =
        same_frame ? Eigen::Quaternion<T>::Identity()
                   : EigenQuaternionMap<T>(parameters[1]) *
                         EigenQuaternionMap<T>(parameters[0]).conjugate();
    if (sensor2_index >= 0) {
      error = EigenQuaternionMap<T>(parameters[sensor2_index]) * error;
    }
    if (sensor1_index >= 0) {
      error =
          error * EigenQuaternionMap<T>(parameters[sensor1_index]).conjugate();
    }
    error = sensor2_from_sensor1_prior.conjugate().cast<T>() * error;
    AngleAxisFromEigenQuaternion(error.coeffs().data(), residuals);
    return true;
  }

  Eigen::Quaterniond sensor2_from_sensor1_prior;
  int sensor1_index = -1;
  int sensor2_index = -1;
  bool same_frame = false;
};

// Computes the error between a translation direction and the direction formed
// from two positions such that: t_ij - scale * (p_j - p_i) is minimized.
// The positions can either be two camera centers or one camera center and one
// 3D point.
// Reference: Zhuang et al., "Baseline Desensitizing In Translation Averaging",
// CVPR 2018.
struct BATAPairwiseDirectionCostFunctor
    : public AutoDiffCostFunctor<BATAPairwiseDirectionCostFunctor, 3, 3, 3, 1> {
  explicit BATAPairwiseDirectionCostFunctor(
      const Eigen::Vector3d& pos2_from_pos1_dir)
      : pos2_from_pos1_dir_(pos2_from_pos1_dir) {}

  template <typename T>
  bool operator()(const T* pos1,
                  const T* pos2,
                  const T* scale,
                  T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec = pos2_from_pos1_dir_.cast<T>() -
                    scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(pos2) -
                                Eigen::Map<const Eigen::Matrix<T, 3, 1>>(pos1));
    return true;
  }

  const Eigen::Vector3d pos2_from_pos1_dir_;
};

// Computes the error between a translation direction and the direction formed
// from a camera (c) and 3D point (p) with constant rig extrinsics, such that:
// t_ij - scale * (p - c + t_rig) is minimized.
struct RigBATAPairwiseDirectionConstantRigCostFunctor
    : public AutoDiffCostFunctor<RigBATAPairwiseDirectionConstantRigCostFunctor,
                                 3,
                                 3,
                                 3,
                                 1> {
  RigBATAPairwiseDirectionConstantRigCostFunctor(
      const Eigen::Vector3d& cam_from_point3D_dir,
      const Eigen::Vector3d& cam_from_rig_translation)
      : cam_from_point3D_dir_(cam_from_point3D_dir),
        cam_from_rig_translation_(cam_from_rig_translation) {}

  template <typename T>
  bool operator()(const T* point3D,
                  const T* rig_in_world,
                  const T* scale,
                  T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec =
        cam_from_point3D_dir_.cast<T>() -
        scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(point3D) -
                    Eigen::Map<const Eigen::Matrix<T, 3, 1>>(rig_in_world) +
                    cam_from_rig_translation_.cast<T>());
    return true;
  }

  const Eigen::Vector3d cam_from_point3D_dir_;
  const Eigen::Vector3d cam_from_rig_translation_;
};

// Computes the error between a translation direction and the direction formed
// from a camera (c) and 3D point (p) with variable rig extrinsics, such that:
// t_ij - scale * (p - c + t_rig) is minimized.
struct RigBATAPairwiseDirectionCostFunctor
    : public AutoDiffCostFunctor<RigBATAPairwiseDirectionCostFunctor,
                                 3,
                                 3,
                                 3,
                                 3,
                                 1> {
  RigBATAPairwiseDirectionCostFunctor(
      const Eigen::Vector3d& cam_from_point3D_dir,
      const Eigen::Quaterniond& rig_from_world_rot)
      : cam_from_point3D_dir_(cam_from_point3D_dir),
        world_from_rig_rot_(rig_from_world_rot.inverse()) {}

  template <typename T>
  bool operator()(const T* point3D,
                  const T* rig_in_world,
                  const T* cam_in_rig,
                  const T* scale,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> cam_from_rig_translation =
        world_from_rig_rot_.cast<T>() *
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(cam_in_rig);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec =
        cam_from_point3D_dir_.cast<T>() -
        scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(point3D) -
                    Eigen::Map<const Eigen::Matrix<T, 3, 1>>(rig_in_world) -
                    cam_from_rig_translation);
    return true;
  }

  const Eigen::Vector3d cam_from_point3D_dir_;
  const Eigen::Quaterniond world_from_rig_rot_;
};

}  // namespace colmap

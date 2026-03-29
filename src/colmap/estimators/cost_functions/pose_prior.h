// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/geometry/rigid3.h"

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

template <typename T>
inline void EigenQuaternionToAngleAxis(const T* eigen_quaternion,
                                       T* angle_axis) {
  const T quaternion[4] = {eigen_quaternion[3],
                           eigen_quaternion[0],
                           eigen_quaternion[1],
                           eigen_quaternion[2]};
  ceres::QuaternionToAngleAxis(quaternion, angle_axis);
}

// 6-DoF error on the absolute sensor pose. The residual is the log of the error
// pose, splitting SE(3) into SO(3) x R^3. The residual is computed in the
// sensor frame. Its first and last three components correspond to the rotation
// and translation errors, respectively.
struct AbsolutePosePriorCostFunctor
    : public AutoDiffCostFunctor<AbsolutePosePriorCostFunctor, 6, 7> {
 public:
  explicit AbsolutePosePriorCostFunctor(const Rigid3d& sensor_from_world_prior)
      : world_from_sensor_prior_(Inverse(sensor_from_world_prior)) {}

  template <typename T>
  bool operator()(const T* const sensor_from_world, T* residuals_ptr) const {
    const Eigen::Quaternion<T> param_from_prior_rotation =
        EigenQuaternionMap<T>(sensor_from_world) *
        world_from_sensor_prior_.rotation().cast<T>();
    EigenQuaternionToAngleAxis(param_from_prior_rotation.coeffs().data(),
                               residuals_ptr);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_prior_translation(
        residuals_ptr + 3);
    param_from_prior_translation =
        EigenVector3Map<T>(sensor_from_world + 4) +
        EigenQuaternionMap<T>(sensor_from_world) *
            world_from_sensor_prior_.translation().cast<T>();

    return true;
  }

 private:
  const Rigid3d world_from_sensor_prior_;
};

// 3-DoF error on the sensor position in the world coordinate frame.
struct AbsolutePosePositionPriorCostFunctor
    : public AutoDiffCostFunctor<AbsolutePosePositionPriorCostFunctor, 3, 7> {
 public:
  explicit AbsolutePosePositionPriorCostFunctor(
      const Eigen::Vector3d& position_in_world_prior)
      : position_in_world_prior_(position_in_world_prior) {}

  template <typename T>
  bool operator()(const T* const sensor_from_world, T* residuals_ptr) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    residuals = position_in_world_prior_.cast<T>() +
                EigenQuaternionMap<T>(sensor_from_world).inverse() *
                    EigenVector3Map<T>(sensor_from_world + 4);
    return true;
  }

 private:
  const Eigen::Vector3d position_in_world_prior_;
};

// 3-DoF error on the rig sensor position in the world coordinate frame.
struct AbsoluteRigPosePositionPriorCostFunctor
    : public AutoDiffCostFunctor<AbsoluteRigPosePositionPriorCostFunctor,
                                 3,
                                 7,
                                 7> {
 public:
  explicit AbsoluteRigPosePositionPriorCostFunctor(
      const Eigen::Vector3d& position_in_world_prior)
      : position_in_world_prior_(position_in_world_prior) {}

  template <typename T>
  bool operator()(const T* const sensor_from_rig,
                  const T* const rig_from_world,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> sensor_from_world_rotation =
        EigenQuaternionMap<T>(sensor_from_rig) *
        EigenQuaternionMap<T>(rig_from_world);
    const Eigen::Matrix<T, 3, 1> sensor_from_world_translation =
        EigenVector3Map<T>(sensor_from_rig + 4) +
        EigenQuaternionMap<T>(sensor_from_rig) *
            EigenVector3Map<T>(rig_from_world + 4);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    residuals =
        position_in_world_prior_.cast<T>() +
        sensor_from_world_rotation.inverse() * sensor_from_world_translation;
    return true;
  }

 private:
  const Eigen::Vector3d position_in_world_prior_;
};

// 6-DoF error between two absolute camera poses based on a prior on their
// relative pose, with identical scale for the translation. The residual is
// computed in the frame of camera i. Its first and last three components
// correspond to the rotation and translation errors, respectively.
//
// Derivation:
//    i_T_w = ΔT_i·i_T_j·j_T_w
//    where ΔT_i = exp(η_i) is the resjdual in SE(3) and η_i in tangent space.
//    Thus η_i = log(i_T_w·j_T_w⁻¹·j_T_i)
//    Rotation term: ΔR = log(i_R_w·j_R_w⁻¹·j_R_i)
//    Translation term: Δt = i_t_w + i_R_w·j_R_w⁻¹·(j_t_i -j_t_w)
struct RelativePosePriorCostFunctor
    : public AutoDiffCostFunctor<RelativePosePriorCostFunctor, 6, 7, 7> {
 public:
  explicit RelativePosePriorCostFunctor(const Rigid3d& i_from_j_prior)
      : j_from_i_prior_(Inverse(i_from_j_prior)) {}

  template <typename T>
  bool operator()(const T* const i_from_world,
                  const T* const j_from_world,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> i_from_j_rotation =
        EigenQuaternionMap<T>(i_from_world) *
        EigenQuaternionMap<T>(j_from_world).inverse();
    const Eigen::Quaternion<T> param_from_prior_rotation =
        i_from_j_rotation * j_from_i_prior_.rotation().template cast<T>();
    EigenQuaternionToAngleAxis(param_from_prior_rotation.coeffs().data(),
                               residuals_ptr);

    const Eigen::Matrix<T, 3, 1> j_from_i_prior_translation =
        j_from_i_prior_.translation().cast<T>() -
        EigenVector3Map<T>(j_from_world + 4);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_prior_translation(
        residuals_ptr + 3);
    param_from_prior_translation =
        EigenVector3Map<T>(i_from_world + 4) +
        i_from_j_rotation * j_from_i_prior_translation;

    return true;
  }

 private:
  const Rigid3d j_from_i_prior_;
};

}  // namespace colmap

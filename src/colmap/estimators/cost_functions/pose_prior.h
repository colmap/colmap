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

#include "colmap/estimators/cost_functions/quaternion_utils.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/geometry/rigid3.h"

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

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
    AngleAxisFromEigenQuaternion(param_from_prior_rotation.coeffs().data(),
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

// 3-DoF chordal error on the "down" direction (gravity) predicted by the
// sensor's rotation vs. a measured sensor-frame down direction. `world_down`
// and `measured_down_in_sensor` are constants baked in at construction,
// exactly like AbsolutePosePositionPriorCostFunctor's
// `position_in_world_prior`.
//
// residual = predicted_down_in_sensor - measured_down_in_sensor
//
// Its norm is 2*sin(theta/2) for angular error theta between the two unit
// vectors: strictly increasing and zero only at theta=0 over the full domain
// [0, pi]. This is deliberately NOT the residual projected onto the tangent
// plane orthogonal to measured_down (norm sin(theta)): that projection is
// also zero at theta=180 degrees (an inverted prediction), since
// tangent_basis^T * (-m - m) = -2 * tangent_basis^T * m = 0 by construction
// (the tangent basis is orthogonal to m). A gravity residual with a false
// minimum at the antipode can hide or even attract an upside-down solution,
// so the full chordal difference is used instead.
//
// The residual is 3-dimensional but its local rank at the true solution is 2
// (predicted_down_in_sensor is constrained to the unit sphere, which is
// 2-dimensional), so a chi-square gate for the robust loss on this residual
// should still use 2 degrees of freedom -- see prior_gravity_loss_scale.
//
// Exactly yaw-invariant (not just to first order): rotating sensor_from_world
// about its own predicted-down axis by any angle leaves
// predicted_down_in_sensor unchanged, because rotation about an axis fixes
// that axis -- so a pure yaw perturbation produces an identically zero
// residual, satisfying "roll/pitch only, yaw free".
struct AbsoluteGravityPriorCostFunctor
    : public AutoDiffCostFunctor<AbsoluteGravityPriorCostFunctor, 3, 7> {
 public:
  AbsoluteGravityPriorCostFunctor(
      const Eigen::Vector3d& world_down,
      const Eigen::Vector3d& measured_down_in_sensor)
      : world_down_(world_down.normalized()),
        measured_down_in_sensor_(measured_down_in_sensor.normalized()) {}

  template <typename T>
  bool operator()(const T* const sensor_from_world, T* residuals_ptr) const {
    const Eigen::Matrix<T, 3, 1> predicted_down_in_sensor =
        EigenQuaternionMap<T>(sensor_from_world) * world_down_.cast<T>();
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    residuals = predicted_down_in_sensor - measured_down_in_sensor_.cast<T>();
    return true;
  }

 private:
  const Eigen::Vector3d world_down_;
  const Eigen::Vector3d measured_down_in_sensor_;
};

// Rig variant of AbsoluteGravityPriorCostFunctor, for non-reference sensors.
struct AbsoluteRigGravityPriorCostFunctor
    : public AutoDiffCostFunctor<AbsoluteRigGravityPriorCostFunctor, 3, 7, 7> {
 public:
  AbsoluteRigGravityPriorCostFunctor(
      const Eigen::Vector3d& world_down,
      const Eigen::Vector3d& measured_down_in_sensor)
      : world_down_(world_down.normalized()),
        measured_down_in_sensor_(measured_down_in_sensor.normalized()) {}

  template <typename T>
  bool operator()(const T* const sensor_from_rig,
                  const T* const rig_from_world,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> sensor_from_world_rotation =
        EigenQuaternionMap<T>(sensor_from_rig) *
        EigenQuaternionMap<T>(rig_from_world);
    const Eigen::Matrix<T, 3, 1> predicted_down_in_sensor =
        sensor_from_world_rotation * world_down_.cast<T>();
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    residuals = predicted_down_in_sensor - measured_down_in_sensor_.cast<T>();
    return true;
  }

 private:
  const Eigen::Vector3d world_down_;
  const Eigen::Vector3d measured_down_in_sensor_;
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
    AngleAxisFromEigenQuaternion(param_from_prior_rotation.coeffs().data(),
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

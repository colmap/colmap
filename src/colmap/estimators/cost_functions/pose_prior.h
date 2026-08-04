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

// 1-DoF wrap-safe angular error between a measured compass heading and the
// heading the sensor's rotation predicts.
//
// A heading is one number -- an azimuth -- and it is represented as one
// number. Promoting it to a quaternion would let the solver read roll and
// pitch out of a measurement that contains neither, and those fabricated two
// degrees of freedom would compete with the gravity residual, which does
// measure them.
//
// The azimuth is only defined relative to a horizontal plane, and the plane
// this fork has is the one the measured camera-frame down vector `d`
// establishes. Both the measured and the predicted north directions are
// projected into that plane and compared there, so the residual is unaffected
// by how the camera is tilted:
//
//   f     = (0, 0, 1)                       camera forward (COLMAP axes)
//   f_h   = normalize(f - d * dot(d, f))    forward, projected horizontal
//   r_h   = normalize(cross(d, f_h))        right, completing the frame
//   n_m   = cos(h) * f_h - sin(h) * r_h     measured north, in camera frame
//   n_p   = R_sensor_from_world * north_world      predicted north
//   n_p_h = normalize(n_p - d * dot(d, n_p))
//   residual = atan2(dot(d, cross(n_m, n_p_h)), dot(n_m, n_p_h))
//
// The atan2 form is what makes it wrap-safe: it is signed, continuous across
// 0/2*pi, and returns magnitude pi -- not a false zero -- for an exactly
// opposite heading. A residual built from a vector difference or from a
// subtraction of angles would either be blind to the antipode or discontinuous
// at the wrap point, and a compass that is 180 degrees out is a real failure
// mode (a sign convention flipped somewhere upstream), not a hypothetical.
//
// `north_world` is this row's own north taken from the shared ENU frame, not
// one constant direction reused across the scene. `measured_down_in_sensor`
// must be the same vector the gravity residual uses for this image.
//
// Caller responsibilities, because this functor cannot check them:
//   - Reject the row when norm(f - d * dot(d, f)) is below
//     kMinHeadingHorizontalProjectionNorm: the camera is then pointing so
//     nearly straight up or down that its azimuth is not defined, and f_h
//     would be numerical noise scaled to unit length.
//   - Divide the residual by that row's heading_stddev_rad and apply the
//     fixed 1-DoF robust radius.
struct AbsoluteHeadingPriorCostFunctor
    : public AutoDiffCostFunctor<AbsoluteHeadingPriorCostFunctor, 1, 7> {
 public:
  AbsoluteHeadingPriorCostFunctor(
      const Eigen::Vector3d& north_world,
      const Eigen::Vector3d& measured_down_in_sensor,
      double measured_heading_rad)
      : north_world_(north_world.normalized()),
        down_(measured_down_in_sensor.normalized()),
        measured_north_in_sensor_(
            MeasuredNorthInSensor(down_, measured_heading_rad)) {}

  // The measured heading as a direction in the camera frame. Constant at
  // construction: it depends only on the measurement, never on the pose.
  static Eigen::Vector3d MeasuredNorthInSensor(const Eigen::Vector3d& down,
                                               double heading_rad) {
    const Eigen::Vector3d forward(0.0, 0.0, 1.0);
    const Eigen::Vector3d forward_horizontal =
        (forward - down * down.dot(forward)).normalized();
    const Eigen::Vector3d right_horizontal =
        down.cross(forward_horizontal).normalized();
    return std::cos(heading_rad) * forward_horizontal -
           std::sin(heading_rad) * right_horizontal;
  }

  template <typename T>
  bool operator()(const T* const sensor_from_world, T* residuals_ptr) const {
    const Eigen::Matrix<T, 3, 1> down = down_.cast<T>();
    const Eigen::Matrix<T, 3, 1> predicted_north =
        EigenQuaternionMap<T>(sensor_from_world) * north_world_.cast<T>();
    const Eigen::Matrix<T, 3, 1> predicted_north_horizontal =
        (predicted_north - down * down.dot(predicted_north)).normalized();
    const Eigen::Matrix<T, 3, 1> measured_north =
        measured_north_in_sensor_.cast<T>();
    residuals_ptr[0] =
        ceres::atan2(down.dot(measured_north.cross(predicted_north_horizontal)),
                     measured_north.dot(predicted_north_horizontal));
    return true;
  }

 private:
  const Eigen::Vector3d north_world_;
  const Eigen::Vector3d down_;
  const Eigen::Vector3d measured_north_in_sensor_;
};

// Rig variant of AbsoluteHeadingPriorCostFunctor, for non-reference sensors.
struct AbsoluteRigHeadingPriorCostFunctor
    : public AutoDiffCostFunctor<AbsoluteRigHeadingPriorCostFunctor, 1, 7, 7> {
 public:
  AbsoluteRigHeadingPriorCostFunctor(
      const Eigen::Vector3d& north_world,
      const Eigen::Vector3d& measured_down_in_sensor,
      double measured_heading_rad)
      : north_world_(north_world.normalized()),
        down_(measured_down_in_sensor.normalized()),
        measured_north_in_sensor_(
            AbsoluteHeadingPriorCostFunctor::MeasuredNorthInSensor(
                down_, measured_heading_rad)) {}

  template <typename T>
  bool operator()(const T* const sensor_from_rig,
                  const T* const rig_from_world,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> sensor_from_world_rotation =
        EigenQuaternionMap<T>(sensor_from_rig) *
        EigenQuaternionMap<T>(rig_from_world);
    const Eigen::Matrix<T, 3, 1> down = down_.cast<T>();
    const Eigen::Matrix<T, 3, 1> predicted_north =
        sensor_from_world_rotation * north_world_.cast<T>();
    const Eigen::Matrix<T, 3, 1> predicted_north_horizontal =
        (predicted_north - down * down.dot(predicted_north)).normalized();
    const Eigen::Matrix<T, 3, 1> measured_north =
        measured_north_in_sensor_.cast<T>();
    residuals_ptr[0] =
        ceres::atan2(down.dot(measured_north.cross(predicted_north_horizontal)),
                     measured_north.dot(predicted_north_horizontal));
    return true;
  }

 private:
  const Eigen::Vector3d north_world_;
  const Eigen::Vector3d down_;
  const Eigen::Vector3d measured_north_in_sensor_;
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

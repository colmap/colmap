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
#include "colmap/estimators/imu_preintegration.h"
#include "colmap/geometry/pose.h"
#include "colmap/util/logging.h"

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

// IMU preintegration cost function operating on body-frame poses in a
// gravity-aligned, metric world frame. This is the standard formulation
// for VIO bundle adjustment (Forster et al. TRO 16).
//
// Gravity is a fixed constructor argument (not optimized), typically
// [0, 0, -9.81] for a gravity-aligned world frame.
//
// Takes a pointer to externally-owned PreintegratedImuData so that a
// ReintegrationCallback can update the data between Ceres iterations
// without rebuilding cost functions.
//
// Residual: 15-dimensional
//   [0:3]   rotation error (angle-axis)
//   [3:6]   position error (body frame i)
//   [6:9]   velocity error (body frame i)
//   [9:15]  bias random walk (gyro then accel)
//
// Parameter blocks:
//   [0] body_from_world_i:  7  (qx,qy,qz,qw, tx,ty,tz)
//       Body (IMU) pose at frame i in gravity-aligned metric world.
//   [1] imu_state_i:        9  (vx,vy,vz, bgx,bgy,bgz, bax,bay,baz)
//       Velocity and biases at frame i, in the world frame.
//   [2] body_from_world_j:  7
//       Body (IMU) pose at frame j.
//   [3] imu_state_j:        9
//       Velocity and biases at frame j.
class ImuPreintegrationCostFunctor {
 public:
  ImuPreintegrationCostFunctor(const PreintegratedImuData* data,
                               const Eigen::Vector3d& gravity)
      : data_(data), gravity_(gravity) {
    THROW_CHECK(!data_->sqrt_information.isZero())
        << "PreintegratedImuData must be finalized before use in cost "
           "function. Call Extract() or Update() on the integrator, or "
           "Finalize() on the data directly.";
  }

  static ceres::CostFunction* Create(const PreintegratedImuData* data,
                                     const Eigen::Vector3d& gravity) {
    return (
        new ceres::
            AutoDiffCostFunction<ImuPreintegrationCostFunctor, 15, 7, 9, 7, 9>(
                new ImuPreintegrationCostFunctor(data, gravity)));
  }

  template <typename T>
  bool operator()(const T* const body_from_world_i,
                  const T* const imu_state_i,
                  const T* const body_from_world_j,
                  const T* const imu_state_j,
                  T* residuals) const {
    // IMU state: [velocity(3), gyro_bias(3), acc_bias(3)].
    EigenVector3Map<T> v_i(imu_state_i);
    EigenVector3Map<T> v_j(imu_state_j);
    Eigen::Matrix<T, 6, 1> delta_b =
        Eigen::Map<const Eigen::Matrix<T, 6, 1>>(imu_state_i + 3) -
        data_->biases.cast<T>();
    EigenVector3Map<T> delta_b_g(delta_b.data());
    EigenVector3Map<T> delta_b_a(delta_b.data() + 3);
    const T dt = T(data_->delta_t);
    const Eigen::Matrix<T, 3, 1> gravity = gravity_.cast<T>();

    // World-frame positions: p_W = -R_WB * t_BW.
    Eigen::Quaternion<T> q_BW_i = EigenQuaternionMap<T>(body_from_world_i);
    Eigen::Quaternion<T> q_WB_i = q_BW_i.inverse();
    Eigen::Matrix<T, 3, 1> p_W_i =
        q_WB_i * EigenVector3Map<T>(body_from_world_i + 4) * T(-1.);
    Eigen::Quaternion<T> q_BW_j = EigenQuaternionMap<T>(body_from_world_j);
    Eigen::Quaternion<T> q_WB_j = q_BW_j.inverse();
    Eigen::Matrix<T, 3, 1> p_W_j =
        q_WB_j * EigenVector3Map<T>(body_from_world_j + 4) * T(-1.);

    // Rotation residual (Forster et al. TRO 16, Eq. 44).
    // Our preintegrator uses right-multiply (Forster convention), producing:
    //   delta_R = R_BW_i^{-1} * R_BW_j = q_WB_i * q_BW_j
    // The measured relative rotation from the poses must match this convention.
    const Eigen::Quaternion<T> delta_R_measured = q_WB_i * q_BW_j;
    // First-order bias correction: delta_R_corrected = delta_R * Exp(dR_dbg *
    // dbg).
    Eigen::Matrix<T, 3, 1> omega_bias = data_->dR_dbg.cast<T>() * delta_b_g;
    Eigen::Quaternion<T> Dq_bias;
    AngleAxisToEigenQuaternion(omega_bias.data(), Dq_bias.coeffs().data());
    const Eigen::Quaternion<T> delta_R_corrected =
        data_->delta_R.cast<T>() * Dq_bias;
    const Eigen::Quaternion<T> rotation_error =
        (delta_R_corrected.inverse() * delta_R_measured).normalized();
    EigenQuaternionToAngleAxis(rotation_error.coeffs().data(), residuals);

    // Position residual.
    const Eigen::Matrix<T, 3, 1> dp_W =
        p_W_j - p_W_i - v_i * dt - 0.5 * gravity * dt * dt;
    Eigen::Matrix<T, 3, 1> est_dp = q_BW_i * dp_W;
    Eigen::Matrix<T, 3, 1> Dp = data_->delta_p.cast<T>() +
                                data_->dp_dba.cast<T>() * delta_b_a +
                                data_->dp_dbg.cast<T>() * delta_b_g;
    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_measured_p(residuals + 3);
    param_from_measured_p = est_dp - Dp;

    // Velocity residual.
    const Eigen::Matrix<T, 3, 1> dv_W = v_j - v_i - gravity * dt;
    Eigen::Matrix<T, 3, 1> est_dv = q_BW_i * dv_W;
    Eigen::Matrix<T, 3, 1> Dv = data_->delta_v.cast<T>() +
                                data_->dv_dba.cast<T>() * delta_b_a +
                                data_->dv_dbg.cast<T>() * delta_b_g;
    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_measured_v(residuals + 6);
    param_from_measured_v = est_dv - Dv;

    // Bias random walk residual.
    for (size_t i = 0; i < 6; ++i) {
      residuals[i + 9] = imu_state_j[i + 3] - imu_state_i[i + 3];
    }

    // Weight by sqrt information.
    Eigen::Map<Eigen::Matrix<T, 15, 1>> residuals_data(residuals);
    residuals_data.applyOnTheLeft(data_->sqrt_information.cast<T>());
    return true;
  }

 private:
  const PreintegratedImuData* data_;
  Eigen::Vector3d gravity_;
};

// IMU preintegration cost function for COLMAP's post-hoc visual-inertial
// refinement. Extends ImuPreintegrationCostFunctor with additional parameter
// blocks for metric scale, gravity direction (in the arbitrary SfM frame),
// and IMU-camera extrinsics.
//
// The gravity direction is optimized as a unit vector in the SfM world frame,
// constrained with SphereManifold(3). This implicitly captures the rotation
// between the SfM frame and the physical gravity-aligned frame.
//
// Takes a pointer to externally-owned PreintegratedImuData so that a
// ReintegrationCallback can update the data between Ceres iterations
// without rebuilding cost functions.
//
// Residual: 15-dimensional (same as ImuPreintegrationCostFunctor)
//
// Parameter blocks:
//   [0] log_scale:          1
//       Logarithm of the metric scale factor applied to SfM translations.
//   [1] gravity_direction:  3  (gx,gy,gz) unit vector
//       Gravity direction in the visual (SfM) world frame. This accounts
//       for the fact that the SfM world frame has arbitrary orientation.
//       Constrain with SphereManifold(3). The gravity magnitude is taken
//       from PreintegratedImuData::gravity_magnitude.
//   [2] imu_from_cam:       7  (qx,qy,qz,qw, tx,ty,tz)
//       Rigid transform from camera to IMU body frame.
//   [3] i_from_world:       7  (qx,qy,qz,qw, tx,ty,tz)
//       Camera pose at frame i (cam_from_world in SfM frame).
//   [4] i_imu_state:        9  (vx,vy,vz, bgx,bgy,bgz, bax,bay,baz)
//       Velocity and biases at frame i, in the SfM world frame.
//   [5] j_from_world:       7
//       Camera pose at frame j.
//   [6] j_imu_state:        9
//       Velocity and biases at frame j.
class VisualCentricImuPreintegrationCostFunctor {
 public:
  explicit VisualCentricImuPreintegrationCostFunctor(
      const PreintegratedImuData* data)
      : data_(data) {
    THROW_CHECK(!data_->sqrt_information.isZero())
        << "PreintegratedImuData must be finalized before use in cost "
           "function. Call Extract() or Update() on the integrator, or "
           "Finalize() on the data directly.";
  }

  static ceres::CostFunction* Create(const PreintegratedImuData* data) {
    return (new ceres::AutoDiffCostFunction<
            VisualCentricImuPreintegrationCostFunctor,
            15,
            1,
            3,
            7,
            7,
            9,
            7,
            9>(new VisualCentricImuPreintegrationCostFunctor(data)));
  }

  template <typename T>
  bool operator()(const T* const log_scale,
                  const T* const gravity_direction,
                  const T* const imu_from_cam,
                  const T* const i_from_world,
                  const T* const i_imu_state,
                  const T* const j_from_world,
                  const T* const j_imu_state,
                  T* residuals) const {
    // IMU state: [velocity(3), gyro_bias(3), acc_bias(3)].
    EigenVector3Map<T> v_i_data(i_imu_state);
    EigenVector3Map<T> v_j_data(j_imu_state);
    Eigen::Matrix<T, 6, 1> delta_b =
        Eigen::Map<const Eigen::Matrix<T, 6, 1>>(i_imu_state + 3) -
        data_->biases.cast<T>();
    EigenVector3Map<T> delta_b_g(delta_b.data());
    EigenVector3Map<T> delta_b_a(delta_b.data() + 3);
    const T dt = T(data_->delta_t);

    // Gravity in the visual (SfM) world frame.
    Eigen::Matrix<T, 3, 1> gravity =
        EigenVector3Map<T>(gravity_direction) * T(data_->gravity_magnitude);

    // Convert cam_from_world to world_from_imu.
    // world_from_imu = world_from_cam * cam_from_imu
    //                = inverse(cam_from_world) * inverse(imu_from_cam)
    Eigen::Quaternion<T> cam_from_imu_q =
        EigenQuaternionMap<T>(imu_from_cam).inverse();
    Eigen::Matrix<T, 3, 1> cam_from_imu_t =
        cam_from_imu_q * EigenVector3Map<T>(imu_from_cam + 4) * T(-1.);
    Eigen::Quaternion<T> world_from_i_q =
        EigenQuaternionMap<T>(i_from_world).inverse();
    Eigen::Matrix<T, 3, 1> world_from_i_t =
        world_from_i_q * EigenVector3Map<T>(i_from_world + 4) * T(-1.);
    Eigen::Quaternion<T> world_from_j_q =
        EigenQuaternionMap<T>(j_from_world).inverse();
    Eigen::Matrix<T, 3, 1> world_from_j_t =
        world_from_j_q * EigenVector3Map<T>(j_from_world + 4) * T(-1.);
    // Compose: world_from_imu = world_from_cam * cam_from_imu.
    Eigen::Quaternion<T> world_from_i_imu_q = world_from_i_q * cam_from_imu_q;
    Eigen::Matrix<T, 3, 1> world_from_i_imu_t =
        world_from_i_q * cam_from_imu_t + world_from_i_t;
    Eigen::Quaternion<T> world_from_j_imu_q = world_from_j_q * cam_from_imu_q;
    Eigen::Matrix<T, 3, 1> world_from_j_imu_t =
        world_from_j_q * cam_from_imu_t + world_from_j_t;

    // Apply metric scale to positions and velocities.
    T scale = ceres::exp(log_scale[0]);
    world_from_i_imu_t = world_from_i_imu_t * scale;
    world_from_j_imu_t = world_from_j_imu_t * scale;
    Eigen::Matrix<T, 3, 1> v_i = v_i_data * scale;
    Eigen::Matrix<T, 3, 1> v_j = v_j_data * scale;

    // Rotation residual (Forster et al. TRO 16, Eq. 44).
    // Our preintegrator uses right-multiply (Forster convention), producing:
    //   delta_R = R_BW_i^{-1} * R_BW_j
    // In terms of world_from_imu quaternions (q_WB = q_BW^{-1}):
    //   delta_R = q_WB_i^{-1}^{-1} * q_WB_j^{-1} -- but more simply:
    //   delta_R_measured = world_from_i * world_from_j^{-1}
    //                    = q_WB_i * q_WB_j^{-1} = q_BW_i^{-1} * q_BW_j
    const Eigen::Quaternion<T> delta_R_measured =
        world_from_i_imu_q * world_from_j_imu_q.inverse();
    // First-order bias correction.
    Eigen::Matrix<T, 3, 1> omega_bias = data_->dR_dbg.cast<T>() * delta_b_g;
    Eigen::Quaternion<T> Dq_bias;
    AngleAxisToEigenQuaternion(omega_bias.data(), Dq_bias.coeffs().data());
    const Eigen::Quaternion<T> delta_R_corrected =
        data_->delta_R.cast<T>() * Dq_bias;
    const Eigen::Quaternion<T> rotation_error =
        (delta_R_corrected.inverse() * delta_R_measured).normalized();
    EigenQuaternionToAngleAxis(rotation_error.coeffs().data(), residuals);

    // Position residual: Eq. (45) from Forster et al. TRO 16.
    const Eigen::Matrix<T, 3, 1> j_from_i_p =
        world_from_j_imu_t - world_from_i_imu_t;
    Eigen::Matrix<T, 3, 1> est_dp =
        world_from_i_imu_q.inverse() *
        (j_from_i_p - v_i * dt - 0.5 * gravity * dt * dt);
    Eigen::Matrix<T, 3, 1> Dp = data_->delta_p.cast<T>() +
                                data_->dp_dba.cast<T>() * delta_b_a +
                                data_->dp_dbg.cast<T>() * delta_b_g;
    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_measured_p(residuals + 3);
    param_from_measured_p = est_dp - Dp;

    // Velocity residual.
    const Eigen::Matrix<T, 3, 1> j_from_i_v = v_j - v_i;
    Eigen::Matrix<T, 3, 1> est_dv =
        world_from_i_imu_q.inverse() * (j_from_i_v - gravity * dt);
    Eigen::Matrix<T, 3, 1> Dv = data_->delta_v.cast<T>() +
                                data_->dv_dba.cast<T>() * delta_b_a +
                                data_->dv_dbg.cast<T>() * delta_b_g;
    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_measured_v(residuals + 6);
    param_from_measured_v = est_dv - Dv;

    // Bias random walk residual.
    for (size_t i = 0; i < 6; ++i) {
      residuals[i + 9] = j_imu_state[i + 3] - i_imu_state[i + 3];
    }

    // Weight by sqrt information.
    Eigen::Map<Eigen::Matrix<T, 15, 1>> residuals_data(residuals);
    residuals_data.applyOnTheLeft(data_->sqrt_information.cast<T>());
    return true;
  }

 private:
  const PreintegratedImuData* data_;
};

}  // namespace colmap

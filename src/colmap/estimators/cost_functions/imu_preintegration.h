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
    // IMU state: [velocity(3), bias_gyro(3), bias_accel(3)].
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
    Eigen::Quaternion<T> q_WB_i = q_BW_i.conjugate();
    Eigen::Matrix<T, 3, 1> p_W_i =
        q_WB_i * EigenVector3Map<T>(body_from_world_i + 4) * T(-1.);
    Eigen::Quaternion<T> q_BW_j = EigenQuaternionMap<T>(body_from_world_j);
    Eigen::Quaternion<T> q_WB_j = q_BW_j.conjugate();
    Eigen::Matrix<T, 3, 1> p_W_j =
        q_WB_j * EigenVector3Map<T>(body_from_world_j + 4) * T(-1.);

    // Rotation residual.
    // Left convention: delta_R = body_from_world_j * world_from_body_i.
    const Eigen::Quaternion<T> delta_R_measured = q_BW_j * q_WB_i;
    // First-order bias correction.
    Eigen::Matrix<T, 3, 1> omega_bias = data_->dR_dbg.cast<T>() * delta_b_g;
    Eigen::Quaternion<T> Dq_bias;
    AngleAxisToEigenQuaternion(omega_bias.data(), Dq_bias.coeffs().data());
    const Eigen::Quaternion<T> delta_R_corrected =
        data_->delta_R.cast<T>() * Dq_bias;
    // 2 * vec(q) rotation error: standard VIO parameterization (Forster et al.,
    // VINS-Mono, ORB-SLAM3). Equivalent to angle-axis for small errors.
    const Eigen::Quaternion<T> rotation_error =
        (delta_R_corrected.conjugate() * delta_R_measured).normalized();
    residuals[0] = T(2.0) * rotation_error.x();
    residuals[1] = T(2.0) * rotation_error.y();
    residuals[2] = T(2.0) * rotation_error.z();

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

// Analytical-Jacobian version of ImuPreintegrationCostFunctor.
// Same residual and parameter layout, but implements Evaluate() with
// hand-derived Jacobians instead of Ceres AutoDiff.
//
// Approximation: the rotation Jacobian w.r.t. gyro bias uses a first-order
// linearization that drops the Exp(dR_dbg * dbg) nonlinearity (Forster et al.
// TRO 2016, Eq. 53). This is standard in VIO systems (VINS-Mono, ORB-SLAM3)
// and degrades only with large bias corrections.
//
// Quaternion convention: Jacobians are derived for unit quaternions. Use with
// EigenQuaternionManifold (or ProductManifold with EuclideanManifold<3> for
// the translation) to ensure the unit-norm constraint is maintained.
//
// Residual: 15-dimensional (same ordering as ImuPreintegrationCostFunctor)
//   [0:3]   rotation error (2 * vec(q_error), small-angle approx)
//   [3:6]   position error (body frame i)
//   [6:9]   velocity error (body frame i)
//   [9:15]  bias random walk (gyro then accel)
//
// Parameter blocks:
//   [0] body_from_world_i:  7  (qx,qy,qz,qw, tx,ty,tz)
//   [1] imu_state_i:        9  (vx,vy,vz, bgx,bgy,bgz, bax,bay,baz)
//   [2] body_from_world_j:  7
//   [3] imu_state_j:        9
class AnalyticalImuPreintegrationCostFunction
    : public ceres::SizedCostFunction<15, 7, 9, 7, 9> {
 public:
  AnalyticalImuPreintegrationCostFunction(const PreintegratedImuData* data,
                                          const Eigen::Vector3d& gravity)
      : data_(data), gravity_(gravity) {
    THROW_CHECK(!data_->sqrt_information.isZero())
        << "PreintegratedImuData must be finalized before use in cost "
           "function.";
  }

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    // Extract parameters.
    Eigen::Map<const Eigen::Quaterniond> q_BW_i(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> t_BW_i(parameters[0] + 4);

    Eigen::Map<const Eigen::Vector3d> v_i(parameters[1]);
    Eigen::Map<const Eigen::Vector3d> bg_i(parameters[1] + 3);
    Eigen::Map<const Eigen::Vector3d> ba_i(parameters[1] + 6);

    Eigen::Map<const Eigen::Quaterniond> q_BW_j(parameters[2]);
    Eigen::Map<const Eigen::Vector3d> t_BW_j(parameters[2] + 4);

    Eigen::Map<const Eigen::Vector3d> v_j(parameters[3]);
    Eigen::Map<const Eigen::Vector3d> bg_j(parameters[3] + 3);
    Eigen::Map<const Eigen::Vector3d> ba_j(parameters[3] + 6);

    // Rotation matrices.
    const Eigen::Matrix3d R_BW_i = q_BW_i.normalized().toRotationMatrix();
    const Eigen::Matrix3d R_WB_i = R_BW_i.transpose();
    const Eigen::Matrix3d R_WB_j =
        q_BW_j.normalized().toRotationMatrix().transpose();

    // World-frame positions: p_W = -R_WB * t_BW.
    const Eigen::Vector3d p_W_i = -R_WB_i * t_BW_i;
    const Eigen::Vector3d p_W_j = -R_WB_j * t_BW_j;

    const double dt = data_->delta_t;

    // First-order bias correction.
    const Eigen::Vector3d dbg = bg_i - data_->biases.head<3>();
    const Eigen::Vector3d dba = ba_i - data_->biases.tail<3>();

    const Eigen::Vector3d delta_p =
        data_->delta_p + data_->dp_dbg * dbg + data_->dp_dba * dba;
    const Eigen::Vector3d delta_v =
        data_->delta_v + data_->dv_dbg * dbg + data_->dv_dba * dba;

    const Eigen::Quaterniond dq_correction =
        QuaternionFromAngleAxis(data_->dR_dbg * dbg);
    const Eigen::Quaterniond delta_q =
        (data_->delta_R * dq_correction).normalized();

    // World-frame prediction errors.
    const Eigen::Vector3d dp_W =
        p_W_j - p_W_i - v_i * dt - 0.5 * gravity_ * dt * dt;
    const Eigen::Vector3d dv_W = v_j - v_i - gravity_ * dt;

    // Residuals.
    Eigen::Map<Eigen::Matrix<double, 15, 1>> r(residuals);
    // Rotation: 2 * vec(delta_q^{-1} * q_BW_j * q_BW_i^{-1}).
    const Eigen::Quaterniond q_error =
        delta_q.conjugate() * q_BW_j * q_BW_i.conjugate();
    r.segment<3>(0) = 2.0 * q_error.vec();
    // Position.
    r.segment<3>(3) = R_BW_i * dp_W - delta_p;
    // Velocity.
    r.segment<3>(6) = R_BW_i * dv_W - delta_v;
    // Bias random walk.
    r.segment<3>(9) = bg_j - bg_i;
    r.segment<3>(12) = ba_j - ba_i;

    // Weight by sqrt information.
    r = data_->sqrt_information * r;

    if (jacobians == nullptr) return true;

    // dvec: extracts xyz from xyzw quaternion (3x4 matrix).
    Eigen::Matrix<double, 3, 4> dvec = Eigen::Matrix<double, 3, 4>::Zero();
    dvec(0, 0) = 1.0;
    dvec(1, 1) = 1.0;
    dvec(2, 2) = 1.0;

    // dconj: d(q^{-1})/d(q) for unit quaternion (negate xyz, keep w).
    Eigen::Matrix4d dconj = Eigen::Vector4d(-1, -1, -1, 1).asDiagonal();

    // Jacobian w.r.t. body_from_world_i [7].
    if (jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J(jacobians[0]);
      J.setZero();

      // r_rot = 2*vec(delta_q^{-1} * q_j * q_i^{-1})
      // dr_rot/dq_i = 2 * dvec * L(delta_q^{-1} * q_j) * dconj
      Eigen::Quaterniond q_left_i = delta_q.conjugate() * q_BW_j;
      J.block<3, 4>(0, 0) =
          2.0 * dvec * QuaternionLeftMultMatrix(q_left_i) * dconj;

      // r_pos = R(q_i) * dp_W - delta_p where dp_W depends on q_i through
      // p_W_i = -R(conj(q_i)) * t_i. Full derivative:
      //   d(R(q_i)*dp_W)/dq_i = d(R(q_i)*v)/dq_i|_{v=dp_W}
      //                       + R(q_i) * d(dp_W)/dq_i
      // where d(dp_W)/dq_i = d(R(conj(q_i))*t_i)/dq_i (cross term from p_W_i).
      Eigen::Matrix<double, 3, 4, Eigen::RowMajor> J_Rq_dpW;
      QuaternionRotatePointWithJac(parameters[0], dp_W.data(), J_Rq_dpW.data());
      // Cross term: d(R(conj(q_i))*t_i)/dq_i = dRt_dqconj * dconj
      Eigen::Quaterniond q_BW_i_conj = q_BW_i.conjugate();
      double q_conj_i[4] = {
          q_BW_i_conj.x(), q_BW_i_conj.y(), q_BW_i_conj.z(), q_BW_i_conj.w()};
      Eigen::Matrix<double, 3, 4, Eigen::RowMajor> dRt_dqconj;
      QuaternionRotatePointWithJac(q_conj_i, t_BW_i.data(), dRt_dqconj.data());
      J.block<3, 4>(3, 0) = J_Rq_dpW + R_BW_i * dRt_dqconj * dconj;
      // dr_pos/dt_i: dp_W has -p_W_i, d(-p_W_i)/dt_i = R(conj(q_i)),
      // so d(R(q_i)*dp_W)/dt_i = R(q_i)*R(conj(q_i)) = I for unit q.
      J.block<3, 3>(3, 4) = Eigen::Matrix3d::Identity();

      // r_vel = R(q_i) * dv_W - delta_v
      // dr_vel/dq_i = d(R(q_i)*dv_W)/dq_i
      Eigen::Matrix<double, 3, 4, Eigen::RowMajor> J_vel_qi;
      QuaternionRotatePointWithJac(parameters[0], dv_W.data(), J_vel_qi.data());
      J.block<3, 4>(6, 0) = J_vel_qi;

      J = data_->sqrt_information * J;
    }

    // Jacobian w.r.t. imu_state_i [9].
    if (jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J(jacobians[1]);
      J.setZero();

      // r_rot w.r.t. bg: first-order approximation (drops Exp nonlinearity).
      J.block<3, 3>(0, 3) = -data_->dR_dbg;
      // r_pos w.r.t. v_i: R_BW_i * d(dp_W)/d(v_i) = R_BW_i * (-dt * I)
      J.block<3, 3>(3, 0) = -dt * R_BW_i;
      // r_pos w.r.t. bg, ba: -dp_dbg, -dp_dba
      J.block<3, 3>(3, 3) = -data_->dp_dbg;
      J.block<3, 3>(3, 6) = -data_->dp_dba;
      // r_vel w.r.t. v_i: R_BW_i * d(dv_W)/d(v_i) = R_BW_i * (-I)
      J.block<3, 3>(6, 0) = -R_BW_i;
      // r_vel w.r.t. bg, ba: -dv_dbg, -dv_dba
      J.block<3, 3>(6, 3) = -data_->dv_dbg;
      J.block<3, 3>(6, 6) = -data_->dv_dba;
      // r_bias w.r.t. bg_i, ba_i: -I
      J.block<3, 3>(9, 3) = -Eigen::Matrix3d::Identity();
      J.block<3, 3>(12, 6) = -Eigen::Matrix3d::Identity();

      J = data_->sqrt_information * J;
    }

    // Jacobian w.r.t. body_from_world_j [7].
    if (jacobians[2] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J(jacobians[2]);
      J.setZero();

      // r_rot = 2*vec(delta_q^{-1} * q_j * q_i^{-1})
      // dr_rot/dq_j = 2 * dvec * L(delta_q^{-1}) * R(q_i^{-1})
      J.block<3, 4>(0, 0) = 2.0 * dvec *
                            QuaternionLeftMultMatrix(delta_q.conjugate()) *
                            QuaternionRightMultMatrix(q_BW_i.conjugate());

      // r_pos = R_BW_i * (p_W_j - ...) - delta_p
      // p_W_j = -R_WB_j * t_BW_j, so dp_W_j/dt_j = -R_WB_j
      // dr_pos/dt_j = R_BW_i * (-R_WB_j) = -R_BW_i * R_WB_j
      J.block<3, 3>(3, 4) = -R_BW_i * R_WB_j;

      // dr_pos/dq_j via p_W_j = -R(q_j^{-1}) * t_j
      // = R_BW_i * d(-R(q_j^{-1}) * t_j)/dq_j
      Eigen::Quaterniond q_BW_j_conj = q_BW_j.conjugate();
      double q_conj_arr[4] = {
          q_BW_j_conj.x(), q_BW_j_conj.y(), q_BW_j_conj.z(), q_BW_j_conj.w()};
      Eigen::Matrix<double, 3, 4, Eigen::RowMajor> dRtv_dq;
      QuaternionRotatePointWithJac(q_conj_arr, t_BW_j.data(), dRtv_dq.data());
      J.block<3, 4>(3, 0) = -R_BW_i * dRtv_dq * dconj;

      J = data_->sqrt_information * J;
    }

    // Jacobian w.r.t. imu_state_j [9].
    if (jacobians[3] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J(jacobians[3]);
      J.setZero();

      // r_vel w.r.t. v_j: R_BW_i
      J.block<3, 3>(6, 0) = R_BW_i;
      // r_bias w.r.t. bg_j, ba_j: +I
      J.block<3, 3>(9, 3) = Eigen::Matrix3d::Identity();
      J.block<3, 3>(12, 6) = Eigen::Matrix3d::Identity();

      J = data_->sqrt_information * J;
    }

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
    // IMU state: [velocity(3), bias_gyro(3), bias_accel(3)].
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
        EigenQuaternionMap<T>(imu_from_cam).conjugate();
    Eigen::Matrix<T, 3, 1> cam_from_imu_t =
        cam_from_imu_q * EigenVector3Map<T>(imu_from_cam + 4) * T(-1.);
    Eigen::Quaternion<T> world_from_i_q =
        EigenQuaternionMap<T>(i_from_world).conjugate();
    Eigen::Matrix<T, 3, 1> world_from_i_t =
        world_from_i_q * EigenVector3Map<T>(i_from_world + 4) * T(-1.);
    Eigen::Quaternion<T> world_from_j_q =
        EigenQuaternionMap<T>(j_from_world).conjugate();
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

    // Rotation residual.
    // Left convention: delta_R = body_from_world_j * world_from_body_i.
    // In world_from_imu quaternions: world_from_j^{-1} * world_from_i.
    const Eigen::Quaternion<T> delta_R_measured =
        world_from_j_imu_q.conjugate() * world_from_i_imu_q;
    // First-order bias correction.
    Eigen::Matrix<T, 3, 1> omega_bias = data_->dR_dbg.cast<T>() * delta_b_g;
    Eigen::Quaternion<T> Dq_bias;
    AngleAxisToEigenQuaternion(omega_bias.data(), Dq_bias.coeffs().data());
    const Eigen::Quaternion<T> delta_R_corrected =
        data_->delta_R.cast<T>() * Dq_bias;
    // 2 * vec(q) rotation error: standard VIO parameterization (Forster et al.,
    // VINS-Mono, ORB-SLAM3). Equivalent to angle-axis for small errors.
    const Eigen::Quaternion<T> rotation_error =
        (delta_R_corrected.conjugate() * delta_R_measured).normalized();
    residuals[0] = T(2.0) * rotation_error.x();
    residuals[1] = T(2.0) * rotation_error.y();
    residuals[2] = T(2.0) * rotation_error.z();

    // Position residual.
    const Eigen::Matrix<T, 3, 1> j_from_i_p =
        world_from_j_imu_t - world_from_i_imu_t;
    Eigen::Matrix<T, 3, 1> est_dp =
        world_from_i_imu_q.conjugate() *
        (j_from_i_p - v_i * dt - 0.5 * gravity * dt * dt);
    Eigen::Matrix<T, 3, 1> Dp = data_->delta_p.cast<T>() +
                                data_->dp_dba.cast<T>() * delta_b_a +
                                data_->dp_dbg.cast<T>() * delta_b_g;
    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_measured_p(residuals + 3);
    param_from_measured_p = est_dp - Dp;

    // Velocity residual.
    const Eigen::Matrix<T, 3, 1> j_from_i_v = v_j - v_i;
    Eigen::Matrix<T, 3, 1> est_dv =
        world_from_i_imu_q.conjugate() * (j_from_i_v - gravity * dt);
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

// Analytical-Jacobian version of VisualCentricImuPreintegrationCostFunctor.
// Same residual and parameter layout, but implements Evaluate() with
// hand-derived Jacobians instead of Ceres AutoDiff.
//
// Approximations (same as AnalyticalImuPreintegrationCostFunction):
//   1. Rotation Jacobian w.r.t. gyro bias uses first-order linearization
//      (Forster et al. TRO 2016, Eq. 53).
//   2. Quaternion Jacobians assume unit-norm quaternions.
//
// Residual: 15-dimensional (same as VisualCentricImuPreintegrationCostFunctor)
//
// Parameter blocks:
//   [0] log_scale:          1
//   [1] gravity_direction:  3
//   [2] imu_from_cam:       7
//   [3] i_from_world:       7
//   [4] i_imu_state:        9
//   [5] j_from_world:       7
//   [6] j_imu_state:        9
class AnalyticalVisualCentricImuPreintegrationCostFunction
    : public ceres::SizedCostFunction<15, 1, 3, 7, 7, 9, 7, 9> {
 public:
  explicit AnalyticalVisualCentricImuPreintegrationCostFunction(
      const PreintegratedImuData* data)
      : data_(data) {
    THROW_CHECK(!data_->sqrt_information.isZero())
        << "PreintegratedImuData must be finalized before use in cost "
           "function.";
  }

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    // Extract parameters.
    const double log_scale = parameters[0][0];
    Eigen::Map<const Eigen::Vector3d> gravity_dir(parameters[1]);
    Eigen::Map<const Eigen::Quaterniond> q_IC(parameters[2]);
    Eigen::Map<const Eigen::Vector3d> t_IC(parameters[2] + 4);
    Eigen::Map<const Eigen::Quaterniond> q_CW_i(parameters[3]);
    Eigen::Map<const Eigen::Vector3d> t_CW_i(parameters[3] + 4);
    Eigen::Map<const Eigen::Vector3d> v_i_data(parameters[4]);
    Eigen::Map<const Eigen::Vector3d> bg_i(parameters[4] + 3);
    Eigen::Map<const Eigen::Vector3d> ba_i(parameters[4] + 6);
    Eigen::Map<const Eigen::Quaterniond> q_CW_j(parameters[5]);
    Eigen::Map<const Eigen::Vector3d> t_CW_j(parameters[5] + 4);
    Eigen::Map<const Eigen::Vector3d> v_j_data(parameters[6]);
    Eigen::Map<const Eigen::Vector3d> bg_j(parameters[6] + 3);
    Eigen::Map<const Eigen::Vector3d> ba_j(parameters[6] + 6);

    const double dt = data_->delta_t;
    const double scale = std::exp(log_scale);
    const double grav_mag = data_->gravity_magnitude;
    const Eigen::Vector3d gravity = gravity_dir * grav_mag;

    // Intermediate transforms.
    const Eigen::Quaterniond q_CI = q_IC.conjugate();
    const Eigen::Matrix3d R_CI = q_CI.toRotationMatrix();
    const Eigen::Vector3d t_CI = -(R_CI * t_IC);

    const Eigen::Quaterniond q_WC_i = q_CW_i.conjugate();
    const Eigen::Matrix3d R_WC_i = q_WC_i.toRotationMatrix();
    const Eigen::Vector3d t_WC_i = -(R_WC_i * t_CW_i);

    const Eigen::Quaterniond q_WC_j = q_CW_j.conjugate();
    const Eigen::Matrix3d R_WC_j = q_WC_j.toRotationMatrix();
    const Eigen::Vector3d t_WC_j = -(R_WC_j * t_CW_j);

    const Eigen::Quaterniond q_WI_i = q_WC_i * q_CI;
    const Eigen::Vector3d t_WI_i_unscaled = R_WC_i * t_CI + t_WC_i;
    const Eigen::Quaterniond q_WI_j = q_WC_j * q_CI;
    const Eigen::Vector3d t_WI_j_unscaled = R_WC_j * t_CI + t_WC_j;

    const Eigen::Vector3d p_i = t_WI_i_unscaled * scale;
    const Eigen::Vector3d p_j = t_WI_j_unscaled * scale;
    const Eigen::Vector3d v_i = v_i_data * scale;
    const Eigen::Vector3d v_j = v_j_data * scale;

    const Eigen::Quaterniond q_IW_i = q_WI_i.conjugate();
    const Eigen::Matrix3d R_IW_i = q_IW_i.toRotationMatrix();

    // Bias correction.
    const Eigen::Vector3d dbg = bg_i - data_->biases.head<3>();
    const Eigen::Vector3d dba = ba_i - data_->biases.tail<3>();
    const Eigen::Vector3d delta_p =
        data_->delta_p + data_->dp_dbg * dbg + data_->dp_dba * dba;
    const Eigen::Vector3d delta_v =
        data_->delta_v + data_->dv_dbg * dbg + data_->dv_dba * dba;
    const Eigen::Quaterniond dq_correction =
        QuaternionFromAngleAxis(data_->dR_dbg * dbg);
    const Eigen::Quaterniond delta_q =
        (data_->delta_R * dq_correction).normalized();

    // Residuals.
    Eigen::Map<Eigen::Matrix<double, 15, 1>> r(residuals);
    const Eigen::Quaterniond q_error =
        delta_q.conjugate() * q_WI_j.conjugate() * q_WI_i;
    r.segment<3>(0) = 2.0 * q_error.vec();

    const Eigen::Vector3d dp_W = p_j - p_i - v_i * dt - 0.5 * gravity * dt * dt;
    r.segment<3>(3) = R_IW_i * dp_W - delta_p;

    const Eigen::Vector3d dv_W = v_j - v_i - gravity * dt;
    r.segment<3>(6) = R_IW_i * dv_W - delta_v;

    r.segment<3>(9) = bg_j - bg_i;
    r.segment<3>(12) = ba_j - ba_i;

    r = data_->sqrt_information * r;

    if (jacobians == nullptr) return true;

    // Common Jacobian helpers.
    Eigen::Matrix<double, 3, 4> dvec = Eigen::Matrix<double, 3, 4>::Zero();
    dvec(0, 0) = 1.0;
    dvec(1, 1) = 1.0;
    dvec(2, 2) = 1.0;
    Eigen::Matrix4d dconj = Eigen::Vector4d(-1, -1, -1, 1).asDiagonal();

    // [0] log_scale (1).
    if (jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 15, 1>> J(jacobians[0]);
      J.setZero();
      // d(r_pos)/d(log_s) = R_IW_i * (t_WI_j - t_WI_i - v_i_data*dt) * scale
      J.segment<3>(3) =
          R_IW_i * (t_WI_j_unscaled - t_WI_i_unscaled - v_i_data * dt) * scale;
      // d(r_vel)/d(log_s) = R_IW_i * (v_j_data - v_i_data) * scale
      J.segment<3>(6) = R_IW_i * (v_j_data - v_i_data) * scale;
      J = data_->sqrt_information * J;
    }

    // [1] gravity_direction (3).
    if (jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[1]);
      J.setZero();
      J.block<3, 3>(3, 0) = -0.5 * dt * dt * grav_mag * R_IW_i;
      J.block<3, 3>(6, 0) = -dt * grav_mag * R_IW_i;
      J = data_->sqrt_information * J;
    }

    // [2] imu_from_cam (7).
    if (jacobians[2] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J(jacobians[2]);
      J.setZero();

      // Rotation residual: q_error = A * conj(q_WI_j) * q_WI_i
      // where q_WI_k = q_WC_k * q_CI, q_CI = conj(q_IC).
      // Using: q_error = A * conj(q_CI) * P * q_CI where P =
      // conj(q_WC_j)*q_WC_i. d(q_error)/d(q_IC) via quaternion product rule:
      //   d(conj(q_CI)*P*q_CI)/d(q_CI) = L(conj(q_CI))*L(P) + R(P*q_CI)*dconj
      //   Then chain through d(q_CI)/d(q_IC) = dconj.
      const Eigen::Quaterniond A = delta_q.conjugate();
      const Eigen::Quaterniond P = q_WC_j.conjugate() * q_WC_i;
      J.block<3, 4>(0, 0) = 2.0 * dvec * QuaternionLeftMultMatrix(A) *
                            (QuaternionLeftMultMatrix(q_CI.conjugate()) *
                                 QuaternionLeftMultMatrix(P) +
                             QuaternionRightMultMatrix(P * q_CI) * dconj) *
                            dconj;

      // Position residual: through R_IW_i and through p_i, p_j.
      // q_IW_i = conj(q_CI)*conj(q_WC_i) = q_IC*conj(q_WC_i)
      // d(q_IW_i)/d(q_IC) = R(conj(q_WC_i))
      Eigen::Matrix4d dqIWi_dqIC =
          QuaternionRightMultMatrix(q_WC_i.conjugate());
      Eigen::Matrix<double, 3, 4, Eigen::RowMajor> dRdpW_dqIWi;
      {
        double q_arr[4] = {q_IW_i.x(), q_IW_i.y(), q_IW_i.z(), q_IW_i.w()};
        QuaternionRotatePointWithJac(q_arr, dp_W.data(), dRdpW_dqIWi.data());
      }

      // t_CI = -R(q_CI)*t_IC, d(t_CI)/d(q_IC) = -dRtIC_dqCI * dconj
      Eigen::Matrix<double, 3, 4, Eigen::RowMajor> dRtIC_dqCI;
      {
        double q_arr[4] = {q_CI.x(), q_CI.y(), q_CI.z(), q_CI.w()};
        QuaternionRotatePointWithJac(q_arr, t_IC.data(), dRtIC_dqCI.data());
      }
      Eigen::Matrix<double, 3, 4> dtCI_dqIC = -dRtIC_dqCI * dconj;

      J.block<3, 4>(3, 0) = dRdpW_dqIWi * dqIWi_dqIC +
                            R_IW_i * scale * (R_WC_j - R_WC_i) * dtCI_dqIC;
      J.block<3, 3>(3, 4) = scale * R_IW_i * (R_WC_i - R_WC_j) * R_CI;

      // Velocity residual: only through R_IW_i.
      Eigen::Matrix<double, 3, 4, Eigen::RowMajor> dRdvW_dqIWi;
      {
        double q_arr[4] = {q_IW_i.x(), q_IW_i.y(), q_IW_i.z(), q_IW_i.w()};
        QuaternionRotatePointWithJac(q_arr, dv_W.data(), dRdvW_dqIWi.data());
      }
      J.block<3, 4>(6, 0) = dRdvW_dqIWi * dqIWi_dqIC;

      J = data_->sqrt_information * J;
    }

    // [3] i_from_world (7).
    if (jacobians[3] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J(jacobians[3]);
      J.setZero();

      // q_WI_i = conj(q_CW_i) * q_CI.
      // d(q_WI_i)/d(q_CW_i) = R(q_CI) * dconj
      Eigen::Matrix4d dqWIi_dqCWi = QuaternionRightMultMatrix(q_CI) * dconj;

      // Rotation: d(q_error)/d(q_WI_i) = L(A * conj(q_WI_j))
      const Eigen::Quaterniond A_qIWj =
          delta_q.conjugate() * q_WI_j.conjugate();
      J.block<3, 4>(0, 0) =
          2.0 * dvec * QuaternionLeftMultMatrix(A_qIWj) * dqWIi_dqCWi;

      // Position: through R_IW_i and p_i.
      // q_IW_i = q_IC * q_CW_i, d(q_IW_i)/d(q_CW_i) = L(q_IC)
      Eigen::Matrix4d dqIWi_dqCWi = QuaternionLeftMultMatrix(q_IC);
      Eigen::Matrix<double, 3, 4, Eigen::RowMajor> dRdpW_dqIWi;
      {
        double q_arr[4] = {q_IW_i.x(), q_IW_i.y(), q_IW_i.z(), q_IW_i.w()};
        QuaternionRotatePointWithJac(q_arr, dp_W.data(), dRdpW_dqIWi.data());
      }
      // t_WI_i = R(conj(q_CW_i)) * (t_CI - t_CW_i)
      const Eigen::Vector3d v_tWIi = t_CI - t_CW_i;
      Eigen::Matrix<double, 3, 4, Eigen::RowMajor> dRv_dqWCi;
      {
        double q_arr[4] = {q_WC_i.x(), q_WC_i.y(), q_WC_i.z(), q_WC_i.w()};
        QuaternionRotatePointWithJac(q_arr, v_tWIi.data(), dRv_dqWCi.data());
      }
      J.block<3, 4>(3, 0) =
          dRdpW_dqIWi * dqIWi_dqCWi - scale * R_IW_i * dRv_dqWCi * dconj;
      J.block<3, 3>(3, 4) = scale * R_IW_i * R_WC_i;

      // Velocity: through R_IW_i only.
      Eigen::Matrix<double, 3, 4, Eigen::RowMajor> dRdvW_dqIWi;
      {
        double q_arr[4] = {q_IW_i.x(), q_IW_i.y(), q_IW_i.z(), q_IW_i.w()};
        QuaternionRotatePointWithJac(q_arr, dv_W.data(), dRdvW_dqIWi.data());
      }
      J.block<3, 4>(6, 0) = dRdvW_dqIWi * dqIWi_dqCWi;

      J = data_->sqrt_information * J;
    }

    // [4] i_imu_state (9).
    if (jacobians[4] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J(jacobians[4]);
      J.setZero();
      J.block<3, 3>(0, 3) = -data_->dR_dbg;
      J.block<3, 3>(3, 0) = -dt * scale * R_IW_i;
      J.block<3, 3>(3, 3) = -data_->dp_dbg;
      J.block<3, 3>(3, 6) = -data_->dp_dba;
      J.block<3, 3>(6, 0) = -scale * R_IW_i;
      J.block<3, 3>(6, 3) = -data_->dv_dbg;
      J.block<3, 3>(6, 6) = -data_->dv_dba;
      J.block<3, 3>(9, 3) = -Eigen::Matrix3d::Identity();
      J.block<3, 3>(12, 6) = -Eigen::Matrix3d::Identity();
      J = data_->sqrt_information * J;
    }

    // [5] j_from_world (7).
    if (jacobians[5] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J(jacobians[5]);
      J.setZero();

      // q_WI_j = conj(q_CW_j) * q_CI.
      // d(q_WI_j)/d(q_CW_j) = R(q_CI) * dconj
      Eigen::Matrix4d dqWIj_dqCWj = QuaternionRightMultMatrix(q_CI) * dconj;

      // Rotation: d(q_error)/d(q_WI_j) = L(A) * R(q_WI_i) * dconj
      J.block<3, 4>(0, 0) =
          2.0 * dvec * QuaternionLeftMultMatrix(delta_q.conjugate()) *
          QuaternionRightMultMatrix(q_WI_i) * dconj * dqWIj_dqCWj;

      // Position: through p_j only.
      // t_WI_j = R(conj(q_CW_j)) * (t_CI - t_CW_j)
      const Eigen::Vector3d v_tWIj = t_CI - t_CW_j;
      Eigen::Matrix<double, 3, 4, Eigen::RowMajor> dRv_dqWCj;
      {
        double q_arr[4] = {q_WC_j.x(), q_WC_j.y(), q_WC_j.z(), q_WC_j.w()};
        QuaternionRotatePointWithJac(q_arr, v_tWIj.data(), dRv_dqWCj.data());
      }
      J.block<3, 4>(3, 0) = scale * R_IW_i * dRv_dqWCj * dconj;
      J.block<3, 3>(3, 4) = -scale * R_IW_i * R_WC_j;

      J = data_->sqrt_information * J;
    }

    // [6] j_imu_state (9).
    if (jacobians[6] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J(jacobians[6]);
      J.setZero();
      J.block<3, 3>(6, 0) = scale * R_IW_i;
      J.block<3, 3>(9, 3) = Eigen::Matrix3d::Identity();
      J.block<3, 3>(12, 6) = Eigen::Matrix3d::Identity();
      J = data_->sqrt_information * J;
    }

    return true;
  }

 private:
  const PreintegratedImuData* data_;
};

}  // namespace colmap

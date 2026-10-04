// SPDX-License-Identifier: BSD-3-Clause

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

// IMU preintegration cost function for COLMAP's post-hoc visual-inertial
// refinement. Includes additional parameter blocks for metric scale and
// gravity direction (in the arbitrary SfM frame).
//
// The gravity direction is optimized as a unit vector in the SfM world frame,
// constrained with SphereManifold(3). This implicitly captures the rotation
// between the SfM frame and the physical gravity-aligned frame.
//
// Because the IMU is the reference sensor of the rig, i_from_world and
// j_from_world are rig_from_world (T_IW) in the unscaled SfM world frame.
// Camera-to-IMU extrinsics are estimated via visual reprojection errors on the
// rig and do not appear in this cost function.
//
// Takes a pointer to externally-owned PreintegratedImuData so that a
// ReintegrationCallback can update the data between Ceres iterations
// without rebuilding cost functions.
//
// Residual: 15-dimensional
//
// Parameter blocks:
//   [0] log_scale:          1
//       Logarithm of the metric scale factor applied to SfM translations.
//   [1] gravity_direction:  3  (gx,gy,gz) unit vector
//       Gravity direction in the visual (SfM) world frame. Constrain with
//       SphereManifold(3). The gravity magnitude is taken from
//       PreintegratedImuData::gravity_magnitude.
//   [2] i_from_world:       7  (qx,qy,qz,qw, tx,ty,tz)
//       Rig (IMU) pose at frame i (rig_from_world in unscaled SfM frame).
//   [3] i_velocity:         3  (vx,vy,vz)
//       Velocity at frame i, in the unscaled SfM world frame.
//   [4] i_imu_state:        6  (bgx,bgy,bgz, bax,bay,baz)
//       Biases at frame i.
//   [5] j_from_world:       7
//       Rig (IMU) pose at frame j (rig_from_world in unscaled SfM frame).
//   [6] j_velocity:         3
//       Velocity at frame j, in the unscaled SfM world frame.
//   [7] j_imu_state:        6
//       Biases at frame j.
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
            3,
            6,
            7,
            3,
            6>(new VisualCentricImuPreintegrationCostFunctor(data)));
  }

  template <typename T>
  bool operator()(const T* const log_scale,
                  const T* const gravity_direction,
                  const T* const i_from_world,
                  const T* const i_velocity,
                  const T* const i_imu_state,
                  const T* const j_from_world,
                  const T* const j_velocity,
                  const T* const j_imu_state,
                  T* residuals) const {
    EigenVector3Map<T> v_i_data(i_velocity);
    EigenVector3Map<T> v_j_data(j_velocity);
    Eigen::Matrix<T, 6, 1> delta_b =
        Eigen::Map<const Eigen::Matrix<T, 6, 1>>(i_imu_state) -
        data_->biases.cast<T>();
    EigenVector3Map<T> delta_b_g(delta_b.data());
    EigenVector3Map<T> delta_b_a(delta_b.data() + 3);
    const T dt = T(data_->delta_t);

    // Gravity in the visual (SfM) world frame.
    Eigen::Matrix<T, 3, 1> gravity =
        EigenVector3Map<T>(gravity_direction) * T(data_->gravity_magnitude);

    // World-frame positions: p_W = -R_WI * t_IW = -R(conj(q_IW)) * t_IW.
    Eigen::Quaternion<T> q_IW_i = EigenQuaternionMap<T>(i_from_world);
    Eigen::Quaternion<T> q_WI_i = q_IW_i.conjugate();
    Eigen::Matrix<T, 3, 1> world_from_i_t =
        q_WI_i * EigenVector3Map<T>(i_from_world + 4) * T(-1.);

    Eigen::Quaternion<T> q_IW_j = EigenQuaternionMap<T>(j_from_world);
    Eigen::Quaternion<T> q_WI_j = q_IW_j.conjugate();
    Eigen::Matrix<T, 3, 1> world_from_j_t =
        q_WI_j * EigenVector3Map<T>(j_from_world + 4) * T(-1.);

    // Apply metric scale to positions and velocities.
    T scale = ceres::exp(log_scale[0]);
    Eigen::Matrix<T, 3, 1> p_i = world_from_i_t * scale;
    Eigen::Matrix<T, 3, 1> p_j = world_from_j_t * scale;
    Eigen::Matrix<T, 3, 1> v_i = v_i_data * scale;
    Eigen::Matrix<T, 3, 1> v_j = v_j_data * scale;

    // Rotation residual.
    // Left convention: delta_R = body_from_world_j * world_from_body_i
    //                          = q_IW_j * q_WI_i.
    const Eigen::Quaternion<T> delta_R_measured = q_IW_j * q_WI_i;
    // First-order bias correction.
    Eigen::Matrix<T, 3, 1> omega_bias = data_->dR_dbg.cast<T>() * delta_b_g;
    Eigen::Quaternion<T> Dq_bias;
    EigenQuaternionFromAngleAxis(omega_bias.data(), Dq_bias.coeffs().data());
    const Eigen::Quaternion<T> delta_R_corrected =
        data_->delta_R.cast<T>() * Dq_bias;
    // 2 * vec(q) rotation error: standard VIO parameterization (Forster et al.,
    // VINS-Mono, ORB-SLAM3). Equivalent to angle-axis for small errors.
    const Eigen::Quaternion<T> rotation_error =
        delta_R_corrected.conjugate() * delta_R_measured;
    residuals[0] = T(2.0) * rotation_error.x();
    residuals[1] = T(2.0) * rotation_error.y();
    residuals[2] = T(2.0) * rotation_error.z();

    // Position residual.
    const Eigen::Matrix<T, 3, 1> dp_W =
        p_j - p_i - v_i * dt - 0.5 * gravity * dt * dt;
    Eigen::Matrix<T, 3, 1> est_dp = q_IW_i * dp_W;
    Eigen::Matrix<T, 3, 1> Dp = data_->delta_p.cast<T>() +
                                data_->dp_dba.cast<T>() * delta_b_a +
                                data_->dp_dbg.cast<T>() * delta_b_g;
    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_measured_p(residuals + 3);
    param_from_measured_p = est_dp - Dp;

    // Velocity residual.
    const Eigen::Matrix<T, 3, 1> dv_W = v_j - v_i - gravity * dt;
    Eigen::Matrix<T, 3, 1> est_dv = q_IW_i * dv_W;
    Eigen::Matrix<T, 3, 1> Dv = data_->delta_v.cast<T>() +
                                data_->dv_dba.cast<T>() * delta_b_a +
                                data_->dv_dbg.cast<T>() * delta_b_g;
    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_measured_v(residuals + 6);
    param_from_measured_v = est_dv - Dv;

    // Bias random walk residual.
    for (size_t i = 0; i < 6; ++i) {
      residuals[i + 9] = j_imu_state[i] - i_imu_state[i];
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
// Approximations:
//   1. Rotation Jacobian w.r.t. gyro bias uses first-order linearization
//      (Forster et al. TRO 2016, Eq. 53).
//   2. Quaternion Jacobians assume unit-norm quaternions.
//
// Residual: 15-dimensional (same as VisualCentricImuPreintegrationCostFunctor)
//
// Parameter blocks:
//   [0] log_scale:          1
//   [1] gravity_direction:  3
//   [2] i_from_world:       7
//   [3] i_velocity:         3
//   [4] i_imu_state:        6
//   [5] j_from_world:       7
//   [6] j_velocity:         3
//   [7] j_imu_state:        6
class AnalyticalVisualCentricImuPreintegrationCostFunction
    : public ceres::SizedCostFunction<15, 1, 3, 7, 3, 6, 7, 3, 6> {
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

    Eigen::Map<const Eigen::Quaterniond> q_IW_i(parameters[2]);
    Eigen::Map<const Eigen::Vector3d> t_IW_i(parameters[2] + 4);

    Eigen::Map<const Eigen::Vector3d> v_i_data(parameters[3]);
    Eigen::Map<const Eigen::Vector3d> bg_i(parameters[4]);
    Eigen::Map<const Eigen::Vector3d> ba_i(parameters[4] + 3);

    Eigen::Map<const Eigen::Quaterniond> q_IW_j(parameters[5]);
    Eigen::Map<const Eigen::Vector3d> t_IW_j(parameters[5] + 4);

    Eigen::Map<const Eigen::Vector3d> v_j_data(parameters[6]);
    Eigen::Map<const Eigen::Vector3d> bg_j(parameters[7]);
    Eigen::Map<const Eigen::Vector3d> ba_j(parameters[7] + 3);

    const double dt = data_->delta_t;
    const double scale = std::exp(log_scale);
    const double grav_mag = data_->gravity_magnitude;
    const Eigen::Vector3d gravity = gravity_dir * grav_mag;

    // Rotation matrices.
    const Eigen::Matrix3d R_IW_i = q_IW_i.normalized().toRotationMatrix();
    const Eigen::Matrix3d R_WI_i = R_IW_i.transpose();
    const Eigen::Matrix3d R_WI_j =
        q_IW_j.normalized().toRotationMatrix().transpose();

    // World positions unscaled (SfM world frame): p_W_sfm = -R_WI * t_IW.
    const Eigen::Vector3d p_WI_i_unscaled = -R_WI_i * t_IW_i;
    const Eigen::Vector3d p_WI_j_unscaled = -R_WI_j * t_IW_j;

    // Scaled positions and velocities (metric frame):
    const Eigen::Vector3d p_i = p_WI_i_unscaled * scale;
    const Eigen::Vector3d p_j = p_WI_j_unscaled * scale;
    const Eigen::Vector3d v_i = v_i_data * scale;
    const Eigen::Vector3d v_j = v_j_data * scale;

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
    // Rotation: 2 * vec(delta_q^{-1} * q_IW_j * q_IW_i^{-1})
    const Eigen::Quaterniond q_error =
        delta_q.conjugate() * q_IW_j * q_IW_i.conjugate();
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
      J.segment<3>(3) =
          R_IW_i * (p_WI_j_unscaled - p_WI_i_unscaled - v_i_data * dt) * scale;
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

    // [2] i_from_world (7).
    if (jacobians[2] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J(jacobians[2]);
      J.setZero();

      // Rotation: dr_rot/dq_i = 2 * dvec * L(delta_q^{-1} * q_j) * dconj
      Eigen::Quaterniond q_left_i = delta_q.conjugate() * q_IW_j;
      J.block<3, 4>(0, 0) =
          2.0 * dvec * QuaternionLeftMultMatrix(q_left_i) * dconj;

      // Position:
      Eigen::Matrix<double, 3, 4, Eigen::RowMajor> J_Rq_dpW;
      QuaternionRotatePointWithJac(parameters[2], dp_W.data(), J_Rq_dpW.data());
      Eigen::Quaterniond q_IW_i_conj = q_IW_i.conjugate();
      double q_conj_i[4] = {
          q_IW_i_conj.x(), q_IW_i_conj.y(), q_IW_i_conj.z(), q_IW_i_conj.w()};
      Eigen::Matrix<double, 3, 4, Eigen::RowMajor> dRt_dqconj;
      QuaternionRotatePointWithJac(q_conj_i, t_IW_i.data(), dRt_dqconj.data());
      J.block<3, 4>(3, 0) = J_Rq_dpW + scale * R_IW_i * dRt_dqconj * dconj;
      J.block<3, 3>(3, 4) = scale * Eigen::Matrix3d::Identity();

      // Velocity:
      Eigen::Matrix<double, 3, 4, Eigen::RowMajor> J_vel_qi;
      QuaternionRotatePointWithJac(parameters[2], dv_W.data(), J_vel_qi.data());
      J.block<3, 4>(6, 0) = J_vel_qi;

      J = data_->sqrt_information * J;
    }

    // [3] i_velocity (3).
    if (jacobians[3] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[3]);
      J.setZero();
      J.block<3, 3>(3, 0) = -scale * dt * R_IW_i;
      J.block<3, 3>(6, 0) = -scale * R_IW_i;
      J = data_->sqrt_information * J;
    }

    // [4] i_imu_state (6).
    if (jacobians[4] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> J(jacobians[4]);
      J.setZero();
      J.block<3, 3>(0, 0) = -data_->dR_dbg;
      J.block<3, 3>(3, 0) = -data_->dp_dbg;
      J.block<3, 3>(3, 3) = -data_->dp_dba;
      J.block<3, 3>(6, 0) = -data_->dv_dbg;
      J.block<3, 3>(6, 3) = -data_->dv_dba;
      J.block<3, 3>(9, 0) = -Eigen::Matrix3d::Identity();
      J.block<3, 3>(12, 3) = -Eigen::Matrix3d::Identity();
      J = data_->sqrt_information * J;
    }

    // [5] j_from_world (7).
    if (jacobians[5] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J(jacobians[5]);
      J.setZero();

      // Rotation: dr_rot/dq_j = 2 * dvec * L(delta_q^{-1}) * R(q_i^{-1})
      J.block<3, 4>(0, 0) = 2.0 * dvec *
                            QuaternionLeftMultMatrix(delta_q.conjugate()) *
                            QuaternionRightMultMatrix(q_IW_i.conjugate());

      // Position:
      J.block<3, 3>(3, 4) = -scale * R_IW_i * R_WI_j;

      Eigen::Quaterniond q_IW_j_conj = q_IW_j.conjugate();
      double q_conj_j[4] = {
          q_IW_j_conj.x(), q_IW_j_conj.y(), q_IW_j_conj.z(), q_IW_j_conj.w()};
      Eigen::Matrix<double, 3, 4, Eigen::RowMajor> dRt_dqconj;
      QuaternionRotatePointWithJac(q_conj_j, t_IW_j.data(), dRt_dqconj.data());
      J.block<3, 4>(3, 0) = -scale * R_IW_i * dRt_dqconj * dconj;

      J = data_->sqrt_information * J;
    }

    // [6] j_velocity (3).
    if (jacobians[6] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[6]);
      J.setZero();
      J.block<3, 3>(6, 0) = scale * R_IW_i;
      J = data_->sqrt_information * J;
    }

    // [7] j_imu_state (6).
    if (jacobians[7] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> J(jacobians[7]);
      J.setZero();
      J.block<3, 3>(9, 0) = Eigen::Matrix3d::Identity();
      J.block<3, 3>(12, 3) = Eigen::Matrix3d::Identity();
      J = data_->sqrt_information * J;
    }

    return true;
  }

 private:
  const PreintegratedImuData* data_;
};

// IMU preintegration cost function operating on body-frame poses in a
// gravity-aligned, metric world frame. This is the standard formulation
// for VIO bundle adjustment (Forster et al. TRO 16).
//
// Gravity is a fixed constructor argument (not optimized), typically
// [0, 0, -9.81] for a gravity-aligned world frame.
//
// Thin wrapper around VisualCentricImuPreintegrationCostFunctor with
// log_scale fixed to 0 and gravity_direction fixed to gravity /
// data->gravity_magnitude.
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
//   [1] v_i:                3  (vx,vy,vz)
//       Velocity at frame i, in the world frame.
//   [2] imu_state_i:        6  (bgx,bgy,bgz, bax,bay,baz)
//       Biases at frame i.
//   [3] body_from_world_j:  7
//       Body (IMU) pose at frame j.
//   [4] v_j:                3
//       Velocity at frame j, in the world frame.
//   [5] imu_state_j:        6
//       Biases at frame j.
class ImuPreintegrationCostFunctor {
 public:
  ImuPreintegrationCostFunctor(const PreintegratedImuData* data,
                               const Eigen::Vector3d& gravity)
      : cost_(THROW_CHECK_NOTNULL(data)), gravity_direction_([&]() {
          THROW_CHECK_GT(data->gravity_magnitude, 0.0);
          return gravity / data->gravity_magnitude;
        }()) {}

  static ceres::CostFunction* Create(const PreintegratedImuData* data,
                                     const Eigen::Vector3d& gravity) {
    return (new ceres::AutoDiffCostFunction<ImuPreintegrationCostFunctor,
                                            15,
                                            7,
                                            3,
                                            6,
                                            7,
                                            3,
                                            6>(
        new ImuPreintegrationCostFunctor(data, gravity)));
  }

  template <typename T>
  bool operator()(const T* const body_from_world_i,
                  const T* const v_i,
                  const T* const imu_state_i,
                  const T* const body_from_world_j,
                  const T* const v_j,
                  const T* const imu_state_j,
                  T* residuals) const {
    const T log_scale(0.0);
    const Eigen::Matrix<T, 3, 1> gravity_direction =
        gravity_direction_.cast<T>();
    return cost_(&log_scale,
                 gravity_direction.data(),
                 body_from_world_i,
                 v_i,
                 imu_state_i,
                 body_from_world_j,
                 v_j,
                 imu_state_j,
                 residuals);
  }

 private:
  const VisualCentricImuPreintegrationCostFunctor cost_;
  const Eigen::Vector3d gravity_direction_;
};

// Analytical-Jacobian version of ImuPreintegrationCostFunctor.
// Thin wrapper around AnalyticalVisualCentricImuPreintegrationCostFunction
// with log_scale and gravity_direction fixed.
//
// Residual: 15-dimensional (same ordering as ImuPreintegrationCostFunctor)
//   [0:3]   rotation error (2 * vec(q_error), small-angle approx)
//   [3:6]   position error (body frame i)
//   [6:9]   velocity error (body frame i)
//   [9:15]  bias random walk (gyro then accel)
//
// Parameter blocks:
//   [0] body_from_world_i:  7  (qx,qy,qz,qw, tx,ty,tz)
//   [1] v_i:                3  (vx,vy,vz)
//   [2] imu_state_i:        6  (bgx,bgy,bgz, bax,bay,baz)
//   [3] body_from_world_j:  7
//   [4] v_j:                3
//   [5] imu_state_j:        6
class AnalyticalImuPreintegrationCostFunction
    : public ceres::SizedCostFunction<15, 7, 3, 6, 7, 3, 6> {
 public:
  AnalyticalImuPreintegrationCostFunction(const PreintegratedImuData* data,
                                          const Eigen::Vector3d& gravity)
      : cost_(THROW_CHECK_NOTNULL(data)), gravity_direction_([&]() {
          THROW_CHECK_GT(data->gravity_magnitude, 0.0);
          return gravity / data->gravity_magnitude;
        }()) {}

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    const double log_scale = 0.0;
    const double* full_parameters[8] = {&log_scale,
                                        gravity_direction_.data(),
                                        parameters[0],
                                        parameters[1],
                                        parameters[2],
                                        parameters[3],
                                        parameters[4],
                                        parameters[5]};
    if (jacobians == nullptr) {
      return cost_.Evaluate(full_parameters, residuals, nullptr);
    }
    double* full_jacobians[8] = {nullptr,
                                 nullptr,
                                 jacobians[0],
                                 jacobians[1],
                                 jacobians[2],
                                 jacobians[3],
                                 jacobians[4],
                                 jacobians[5]};
    return cost_.Evaluate(full_parameters, residuals, full_jacobians);
  }

 private:
  const AnalyticalVisualCentricImuPreintegrationCostFunction cost_;
  const Eigen::Vector3d gravity_direction_;
};

}  // namespace colmap

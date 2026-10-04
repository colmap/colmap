// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/quaternion_utils.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/estimators/imu_preintegration.h"
#include "colmap/geometry/pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/util/logging.h"

#include <array>
#include <memory>

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

// Robust computation of square-root information matrix for a principal
// sub-block of the preintegration covariance via self-adjoint
// eigendecomposition.
template <int N>
inline Eigen::Matrix<double, N, N> ComputeSubBlockSqrtInformation(
    const Eigen::Matrix<double, N, N>& cov_sub,
    double max_condition_number = 1e12) {
  THROW_CHECK(max_condition_number > 0.0 || max_condition_number == -1.0)
      << "max_condition_number must be positive or -1 (disabled)";

  // Enforce symmetry.
  const Eigen::Matrix<double, N, N> sym_cov =
      (cov_sub + cov_sub.transpose()) / 2.0;

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> saes(
      sym_cov);

  const double max_eval = saes.eigenvalues().maxCoeff();
  const double tol = (max_condition_number > 0.0 && max_eval > 0.0)
                         ? max_eval / max_condition_number
                         : 0.0;
  Eigen::Matrix<double, N, 1> d_inv_sqrt;
  for (int i = 0; i < N; ++i) {
    const double eval = std::max(saes.eigenvalues()(i), tol);
    d_inv_sqrt(i) = (eval > 0.0) ? 1.0 / std::sqrt(eval) : 0.0;
  }
  return d_inv_sqrt.asDiagonal() * saes.eigenvectors().transpose();
}

inline Eigen::Matrix<double, 6, 6> ExtractRotationGyroBiasCovariance(
    const Eigen::Matrix<double, 15, 15>& cov) {
  // Indices: {0, 1, 2, 9, 10, 11}
  Eigen::Matrix<double, 6, 6> sub;
  const std::array<int, 6> idx = {0, 1, 2, 9, 10, 11};
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      sub(r, c) = cov(idx[r], idx[c]);
    }
  }
  return sub;
}

inline Eigen::Matrix<double, 9, 9> ExtractPositionVelocityAccelBiasCovariance(
    const Eigen::Matrix<double, 15, 15>& cov) {
  // Indices: {3, 4, 5, 6, 7, 8, 12, 13, 14}
  Eigen::Matrix<double, 9, 9> sub;
  const std::array<int, 9> idx = {3, 4, 5, 6, 7, 8, 12, 13, 14};
  for (int r = 0; r < 9; ++r) {
    for (int c = 0; c < 9; ++c) {
      sub(r, c) = cov(idx[r], idx[c]);
    }
  }
  return sub;
}

inline Eigen::Matrix<double, 6, 6> ExtractRotationGyroBiasSqrtInformation(
    const PreintegratedImuData& data, double max_condition_number = 1e12) {
  return ComputeSubBlockSqrtInformation<6>(
      ExtractRotationGyroBiasCovariance(data.covariance), max_condition_number);
}

inline Eigen::Matrix<double, 9, 9>
ExtractPositionVelocityAccelBiasSqrtInformation(
    const PreintegratedImuData& data, double max_condition_number = 1e12) {
  return ComputeSubBlockSqrtInformation<9>(
      ExtractPositionVelocityAccelBiasCovariance(data.covariance),
      max_condition_number);
}

// Computes inertial rotation residual (3D) and gyro bias random walk residual
// (3D). Residual:
//   rotation_residuals [0:3]: 2 * vec(delta_R_corrected^{-1} *
//   delta_R_measured) gyro_bias_residuals [0:3]: j_imu_state[3:6] -
//   i_imu_state[3:6]
template <typename T>
inline void ComputeInertialRotationResiduals(
    const PreintegratedImuData& data,
    const Eigen::Quaternion<T>& world_from_i_imu_q,
    const Eigen::Quaternion<T>& world_from_j_imu_q,
    const T* const i_imu_state,
    const T* const j_imu_state,
    T* rotation_residuals,
    T* gyro_bias_residuals = nullptr) {
  // delta_R_measured = world_from_j_imu^{-1} * world_from_i_imu
  const Eigen::Quaternion<T> delta_R_measured =
      world_from_j_imu_q.conjugate() * world_from_i_imu_q;

  // First-order bias correction on preintegrated rotation:
  const Eigen::Matrix<T, 3, 1> delta_b_g =
      Eigen::Map<const Eigen::Matrix<T, 3, 1>>(i_imu_state + 3) -
      data.biases.head<3>().cast<T>();
  const Eigen::Matrix<T, 3, 1> omega_bias = data.dR_dbg.cast<T>() * delta_b_g;
  Eigen::Quaternion<T> Dq_bias;
  EigenQuaternionFromAngleAxis(omega_bias.data(), Dq_bias.coeffs().data());
  const Eigen::Quaternion<T> delta_R_corrected =
      data.delta_R.cast<T>() * Dq_bias;

  const Eigen::Quaternion<T> rotation_error =
      delta_R_corrected.conjugate() * delta_R_measured;
  rotation_residuals[0] = T(2.0) * rotation_error.x();
  rotation_residuals[1] = T(2.0) * rotation_error.y();
  rotation_residuals[2] = T(2.0) * rotation_error.z();

  if (gyro_bias_residuals != nullptr) {
    gyro_bias_residuals[0] = j_imu_state[3] - i_imu_state[3];
    gyro_bias_residuals[1] = j_imu_state[4] - i_imu_state[4];
    gyro_bias_residuals[2] = j_imu_state[5] - i_imu_state[5];
  }
}

// Computes inertial position residual (3D), velocity residual (3D),
// and accel bias random walk residual (3D).
// Residual:
//   position_residuals [0:3]: R_IW_i * (p_j - p_i - v_i * dt - 0.5 * g * dt^2)
//   - delta_p_corr velocity_residuals [0:3]: R_IW_i * (v_j - v_i - g * dt) -
//   delta_v_corr accel_bias_residuals [0:3]: j_imu_state[6:9] -
//   i_imu_state[6:9]
template <typename T>
inline void ComputeInertialPositionVelocityResiduals(
    const PreintegratedImuData& data,
    const Eigen::Quaternion<T>& world_from_i_imu_q,
    const Eigen::Matrix<T, 3, 1>& p_i,
    const Eigen::Matrix<T, 3, 1>& p_j,
    const Eigen::Matrix<T, 3, 1>& v_i,
    const Eigen::Matrix<T, 3, 1>& v_j,
    const Eigen::Matrix<T, 3, 1>& gravity,
    const T* const i_imu_state,
    const T* const j_imu_state,
    T* position_residuals,
    T* velocity_residuals,
    T* accel_bias_residuals = nullptr) {
  const T dt = T(data.delta_t);
  const Eigen::Matrix<T, 6, 1> delta_b =
      Eigen::Map<const Eigen::Matrix<T, 6, 1>>(i_imu_state + 3) -
      data.biases.cast<T>();
  EigenVector3Map<T> delta_b_g(delta_b.data());
  EigenVector3Map<T> delta_b_a(delta_b.data() + 3);

  // Position residual.
  const Eigen::Matrix<T, 3, 1> dp_W =
      p_j - p_i - v_i * dt - T(0.5) * gravity * dt * dt;
  const Eigen::Matrix<T, 3, 1> est_dp = world_from_i_imu_q.conjugate() * dp_W;
  const Eigen::Matrix<T, 3, 1> Dp = data.delta_p.cast<T>() +
                                    data.dp_dba.cast<T>() * delta_b_a +
                                    data.dp_dbg.cast<T>() * delta_b_g;
  Eigen::Map<Eigen::Matrix<T, 3, 1>> r_p(position_residuals);
  r_p = est_dp - Dp;

  // Velocity residual.
  const Eigen::Matrix<T, 3, 1> dv_W = v_j - v_i - gravity * dt;
  const Eigen::Matrix<T, 3, 1> est_dv = world_from_i_imu_q.conjugate() * dv_W;
  const Eigen::Matrix<T, 3, 1> Dv = data.delta_v.cast<T>() +
                                    data.dv_dba.cast<T>() * delta_b_a +
                                    data.dv_dbg.cast<T>() * delta_b_g;
  Eigen::Map<Eigen::Matrix<T, 3, 1>> r_v(velocity_residuals);
  r_v = est_dv - Dv;

  if (accel_bias_residuals != nullptr) {
    accel_bias_residuals[0] = j_imu_state[6] - i_imu_state[6];
    accel_bias_residuals[1] = j_imu_state[7] - i_imu_state[7];
    accel_bias_residuals[2] = j_imu_state[8] - i_imu_state[8];
  }
}

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
    EigenQuaternionFromAngleAxis(omega_bias.data(), Dq_bias.coeffs().data());
    const Eigen::Quaternion<T> delta_R_corrected =
        data_->delta_R.cast<T>() * Dq_bias;
    // 2 * vec(q) rotation error: standard VIO parameterization (Forster et al.,
    // VINS-Mono, ORB-SLAM3). Equivalent to angle-axis for small errors.
    // Omit .normalized() because all input quaternions are already unit-norm,
    // and normalizing a ceres::Jet quaternion introduces a radial projection
    // derivative that causes a discrepancy with analytical Jacobians when the
    // rotation residual is non-zero.
    const Eigen::Quaternion<T> rotation_error =
        delta_R_corrected.conjugate() * delta_R_measured;
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

    ComputeInertialRotationResiduals(*data_,
                                     world_from_i_imu_q,
                                     world_from_j_imu_q,
                                     i_imu_state,
                                     j_imu_state,
                                     residuals,
                                     residuals + 9);

    ComputeInertialPositionVelocityResiduals(*data_,
                                             world_from_i_imu_q,
                                             world_from_i_imu_t,
                                             world_from_j_imu_t,
                                             v_i,
                                             v_j,
                                             gravity,
                                             i_imu_state,
                                             j_imu_state,
                                             residuals + 3,
                                             residuals + 6,
                                             residuals + 12);

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

// Inertial rotation cost functor for rotation averaging.
// Evaluates a 6-dimensional residual over 3D Angle-Axis camera rotations
// and 9D per-frame IMU states:
//   [0:3] 2 * vec(rotation_error)
//   [3:6] bias_gyro_j - bias_gyro_i
//
// Parameter blocks:
//   [0] i_from_world:  3 (Angle-Axis rotation R_CW,i in visual frame)
//   [1] i_imu_state:   9 (velocity(3), bg(3), ba(3))
//   [2] j_from_world:  3 (Angle-Axis rotation R_CW,j in visual frame)
//   [3] j_imu_state:   9 (velocity(3), bg(3), ba(3))
class InertialRotationCostFunctor {
 public:
  InertialRotationCostFunctor(
      const PreintegratedImuData* data,
      const Eigen::Quaterniond& imu_from_cam_q,
      const Eigen::Matrix<double, 6, 6>& sqrt_information =
          Eigen::Matrix<double, 6, 6>::Zero())
      : data_(data),
        imu_from_cam_q_(imu_from_cam_q.normalized()),
        sqrt_information_6x6_(
            sqrt_information.isZero()
                ? ExtractRotationGyroBiasSqrtInformation(*data)
                : sqrt_information) {
    THROW_CHECK(!data_->sqrt_information.isZero())
        << "PreintegratedImuData must be finalized before use in cost "
           "function.";
  }

  InertialRotationCostFunctor(
      const PreintegratedImuData* data,
      const Rigid3d& imu_from_cam,
      const Eigen::Matrix<double, 6, 6>& sqrt_information =
          Eigen::Matrix<double, 6, 6>::Zero())
      : InertialRotationCostFunctor(
            data, imu_from_cam.rotation(), sqrt_information) {}

  static ceres::CostFunction* Create(
      const PreintegratedImuData* data,
      const Eigen::Quaterniond& imu_from_cam_q,
      const Eigen::Matrix<double, 6, 6>& sqrt_information =
          Eigen::Matrix<double, 6, 6>::Zero()) {
    return new ceres::
        AutoDiffCostFunction<InertialRotationCostFunctor, 6, 3, 9, 3, 9>(
            new InertialRotationCostFunctor(
                data, imu_from_cam_q, sqrt_information));
  }

  static ceres::CostFunction* Create(
      const PreintegratedImuData* data,
      const Rigid3d& imu_from_cam,
      const Eigen::Matrix<double, 6, 6>& sqrt_information =
          Eigen::Matrix<double, 6, 6>::Zero()) {
    return Create(data, imu_from_cam.rotation(), sqrt_information);
  }

  template <typename T>
  bool operator()(const T* const i_from_world_aa,
                  const T* const i_imu_state,
                  const T* const j_from_world_aa,
                  const T* const j_imu_state,
                  T* residuals) const {
    Eigen::Quaternion<T> i_from_world_q;
    EigenQuaternionFromAngleAxis(i_from_world_aa,
                                 i_from_world_q.coeffs().data());
    Eigen::Quaternion<T> j_from_world_q;
    EigenQuaternionFromAngleAxis(j_from_world_aa,
                                 j_from_world_q.coeffs().data());

    const Eigen::Quaternion<T> world_from_i_q = i_from_world_q.conjugate();
    const Eigen::Quaternion<T> world_from_j_q = j_from_world_q.conjugate();

    const Eigen::Quaternion<T> cam_from_imu_q =
        imu_from_cam_q_.cast<T>().conjugate();

    const Eigen::Quaternion<T> world_from_i_imu_q =
        world_from_i_q * cam_from_imu_q;
    const Eigen::Quaternion<T> world_from_j_imu_q =
        world_from_j_q * cam_from_imu_q;

    ComputeInertialRotationResiduals(*data_,
                                     world_from_i_imu_q,
                                     world_from_j_imu_q,
                                     i_imu_state,
                                     j_imu_state,
                                     residuals,
                                     residuals + 3);

    Eigen::Map<Eigen::Matrix<T, 6, 1>> r_map(residuals);
    r_map.applyOnTheLeft(sqrt_information_6x6_.cast<T>());
    return true;
  }

 private:
  const PreintegratedImuData* data_;
  Eigen::Quaterniond imu_from_cam_q_;
  Eigen::Matrix<double, 6, 6> sqrt_information_6x6_;
};

// Inertial position and velocity cost functor for global positioning.
// Evaluates a 9-dimensional residual over 3D camera optical centers in world,
// 9D per-frame IMU states, gravity direction, and metric log_scale:
//   [0:3] position error in body frame i
//   [3:6] velocity error in body frame i
//   [6:9] bias_accel_j - bias_accel_i
//
// Parameter blocks:
//   [0] log_scale:          1 (log-scale factor)
//   [1] gravity_direction:  3 (unit vector in unscaled SfM world)
//   [2] i_center_in_world:  3 (camera center c_{W,i} in unscaled SfM world)
//   [3] i_imu_state:        9 (velocity(3) in unscaled SfM world, bg(3), ba(3))
//   [4] j_center_in_world:  3 (camera center c_{W,j} in unscaled SfM world)
//   [5] j_imu_state:        9 (velocity(3) in unscaled SfM world, bg(3), ba(3))
//
// Note: If electronic stabilization (q_iori = cam_from_unrot_cam) is present,
// the unstabilized (physical) camera orientations
// (q_iori_i.conjugate() * cam_from_world_i, etc.) should be passed as
// i_from_world_q and j_from_world_q.
class InertialGlobalPositioningCostFunctor {
 public:
  InertialGlobalPositioningCostFunctor(
      const PreintegratedImuData* data,
      const Rigid3d& imu_from_cam,
      const Eigen::Quaterniond& i_from_world_q,
      const Eigen::Quaterniond& j_from_world_q,
      const Eigen::Matrix<double, 9, 9>& sqrt_information =
          Eigen::Matrix<double, 9, 9>::Zero())
      : data_(data),
        sqrt_information_9x9_(
            sqrt_information.isZero()
                ? ExtractPositionVelocityAccelBiasSqrtInformation(*data)
                : sqrt_information) {
    THROW_CHECK(!data_->sqrt_information.isZero())
        << "PreintegratedImuData must be finalized before use in cost "
           "function.";

    const Eigen::Quaterniond cam_from_imu_q =
        imu_from_cam.rotation().conjugate();
    const Eigen::Vector3d cam_from_imu_t =
        -(cam_from_imu_q * imu_from_cam.translation());

    const Eigen::Quaterniond world_from_i_q = i_from_world_q.conjugate();
    const Eigen::Quaterniond world_from_j_q = j_from_world_q.conjugate();

    world_from_i_imu_q_ = world_from_i_q * cam_from_imu_q;
    d_W_i_ = world_from_i_q * cam_from_imu_t;
    d_W_j_ = world_from_j_q * cam_from_imu_t;
  }

  static ceres::CostFunction* Create(
      const PreintegratedImuData* data,
      const Rigid3d& imu_from_cam,
      const Eigen::Quaterniond& i_from_world_q,
      const Eigen::Quaterniond& j_from_world_q,
      const Eigen::Matrix<double, 9, 9>& sqrt_information =
          Eigen::Matrix<double, 9, 9>::Zero()) {
    return new ceres::AutoDiffCostFunction<InertialGlobalPositioningCostFunctor,
                                           9,
                                           1,
                                           3,
                                           3,
                                           9,
                                           3,
                                           9>(
        new InertialGlobalPositioningCostFunctor(data,
                                                 imu_from_cam,
                                                 i_from_world_q,
                                                 j_from_world_q,
                                                 sqrt_information));
  }

  template <typename T>
  bool operator()(const T* const log_scale,
                  const T* const gravity_direction,
                  const T* const i_center_in_world,
                  const T* const i_imu_state,
                  const T* const j_center_in_world,
                  const T* const j_imu_state,
                  T* residuals) const {
    const T scale = ceres::exp(log_scale[0]);
    const Eigen::Matrix<T, 3, 1> p_i =
        (EigenVector3Map<T>(i_center_in_world) + d_W_i_.cast<T>()) * scale;
    const Eigen::Matrix<T, 3, 1> p_j =
        (EigenVector3Map<T>(j_center_in_world) + d_W_j_.cast<T>()) * scale;
    const Eigen::Matrix<T, 3, 1> v_i = EigenVector3Map<T>(i_imu_state) * scale;
    const Eigen::Matrix<T, 3, 1> v_j = EigenVector3Map<T>(j_imu_state) * scale;

    const Eigen::Matrix<T, 3, 1> gravity =
        EigenVector3Map<T>(gravity_direction) * T(data_->gravity_magnitude);

    ComputeInertialPositionVelocityResiduals(*data_,
                                             world_from_i_imu_q_.cast<T>(),
                                             p_i,
                                             p_j,
                                             v_i,
                                             v_j,
                                             gravity,
                                             i_imu_state,
                                             j_imu_state,
                                             residuals,
                                             residuals + 3,
                                             residuals + 6);

    Eigen::Map<Eigen::Matrix<T, 9, 1>> r_map(residuals);
    r_map.applyOnTheLeft(sqrt_information_9x9_.cast<T>());
    return true;
  }

 private:
  const PreintegratedImuData* data_;
  Eigen::Quaterniond world_from_i_imu_q_;
  Eigen::Vector3d d_W_i_;
  Eigen::Vector3d d_W_j_;
  Eigen::Matrix<double, 9, 9> sqrt_information_9x9_;
};

#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
inline std::unique_ptr<ceres::Manifold> CreateImuStateGyroOnlyManifold() {
  return CreateSubsetManifold(9, {0, 1, 2, 6, 7, 8});
}

inline std::unique_ptr<ceres::Manifold> CreateImuStateVelAccelBiasManifold() {
  return CreateSubsetManifold(9, {3, 4, 5});
}
#else
inline std::unique_ptr<ceres::LocalParameterization>
CreateImuStateGyroOnlyManifold() {
  return CreateSubsetManifold(9, {0, 1, 2, 6, 7, 8});
}

inline std::unique_ptr<ceres::LocalParameterization>
CreateImuStateVelAccelBiasManifold() {
  return CreateSubsetManifold(9, {3, 4, 5});
}
#endif

}  // namespace colmap

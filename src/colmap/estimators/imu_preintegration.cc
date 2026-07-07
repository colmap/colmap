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

#include "colmap/estimators/imu_preintegration.h"

#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/geometry/pose.h"
#include "colmap/util/logging.h"
#include "colmap/util/timestamp.h"

#include <array>
#include <cmath>

#include <Eigen/Dense>

namespace colmap {
namespace {

// Cross-product (skew) matrices of the basis vectors, precomputed once and
// reused across calls (Identity() is not cached: it is a free lazy expression).
const std::array<Eigen::Matrix3d, 3> kEx = {
    CrossProductMatrix(Eigen::Vector3d::UnitX()),
    CrossProductMatrix(Eigen::Vector3d::UnitY()),
    CrossProductMatrix(Eigen::Vector3d::UnitZ())};

}  // namespace

void PreintegratedImuData::Finalize(double max_condition_number) {
  THROW_CHECK(max_condition_number > 0.0 || max_condition_number == -1.0)
      << "max_condition_number must be positive or -1 (disabled)";

  // Enforce symmetry.
  covariance = (covariance + covariance.transpose()) / 2.0;

  // Eigendecomposition for robust sqrt-information.
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15>> saes(
      covariance);

  // Clamp small eigenvalues to limit the condition number of the
  // information matrix.
  const double max_eval = saes.eigenvalues().maxCoeff();
  const double tol =
      (max_condition_number > 0.0) ? max_eval / max_condition_number : 0.0;
  Eigen::Matrix<double, 15, 1> D_inv_sqrt;
  for (int i = 0; i < 15; ++i) {
    double eval = std::max(saes.eigenvalues()(i), tol);
    D_inv_sqrt(i) = (eval > 0.0) ? 1.0 / std::sqrt(eval) : 0.0;
  }

  // sqrt_information = D^{-1/2} * V^T
  // so that sqrt_info^T * sqrt_info = V * D^{-1} * V^T = information.
  sqrt_information = D_inv_sqrt.asDiagonal() * saes.eigenvectors().transpose();
}

ImuPreintegrator::ImuPreintegrator(const ImuPreintegrationOptions& options,
                                   const ImuCalibration& calib,
                                   timestamp_t t_start,
                                   timestamp_t t_end)
    : t_start_(t_start),
      t_end_(t_end),
      options_(options),
      integrator_(ImuIntegrator::Create(options.method)),
      calib_(calib),
      accel_rect_mat_inv_(calib.accel_rectification.inverse()),
      gyro_rect_mat_inv_(calib.gyro_rectification.inverse()) {
  THROW_CHECK_LT(t_start, t_end);
  Reset();
}

void ImuPreintegrator::Reset() {
  data_ = PreintegratedImuData();
  // Restore the fields that are not zero-initialized defaults.
  data_.biases = biases_;
  data_.gravity_magnitude = calib_.gravity_magnitude;
  has_started_ = false;
}

void ImuPreintegrator::SetLinearizationBiases(const Eigen::Vector6d& biases) {
  biases_ = biases;
  data_.biases = biases;
}

namespace {

// Midpoint (trapezoidal) integration with numerical bias Jacobians.
class MidpointImuIntegrator : public ImuIntegrator {
 public:
  void Integrate(const ImuPreintegrationOptions& options,
                 const ImuCalibration& calib,
                 const Eigen::Vector3d& accel_true,
                 const Eigen::Vector3d& gyro_true,
                 double dt,
                 double accel_noise_density,
                 double gyro_noise_density,
                 PreintegratedImuData* data) const override {
    // Left convention midpoint (trapezoidal) integration.
    // Based on Forster et al. "On-Manifold Preintegration for Real-Time
    // Visual-Inertial Odometry", TRO 2016. The original paper uses
    // right-multiply integration; here we use left-multiply integration with
    // right-perturbation bias Jacobians. delta_R^T rotates accel from body_k
    // to body_i frame.

    // Integration step.
    // dR = Exp(-omega * dt): body_from_world evolves as
    // R_BW_{k+1} = Exp(-w*dt) * R_BW_k.
    const Eigen::Quaterniond dq = QuaternionFromAngleAxis(-gyro_true * dt);
    const Eigen::Matrix3d Rs = data->delta_R.toRotationMatrix();
    const Eigen::Matrix3d Rs_T = Rs.transpose();
    // translation
    data->delta_p += data->delta_v * dt + Rs_T * accel_true * 0.5 * dt * dt;
    // velocity
    data->delta_v += Rs_T * accel_true * dt;
    // rotation: delta_R_{k+1} = Exp(-omega * dt) * delta_R_k.
    data->delta_R = dq * data->delta_R;
    // time
    data->delta_t += dt;

    // Update jacobians over bias.
    const Eigen::Matrix3d skew_accel = CrossProductMatrix(accel_true);

    // Covariance propagation.
    // Step 1: jacobian-based propagation.
    // Note: we use right-perturbation model (delta_R * Exp(delta)) for
    // Jacobians, which propagates trivially under left-multiply integration:
    // A(0,0) = I.
    Eigen::Matrix<double, 15, 15> A = Eigen::Matrix<double, 15, 15>::Identity();

    // translation
    A.block<3, 3>(3, 0) = -0.5 * Rs_T * skew_accel * dt * dt;
    A.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * dt;

    // velocity
    A.block<3, 3>(6, 0) = -Rs_T * skew_accel * dt;

    // Fill in the bias-related jacobians.
    // Covariance state: [rotation(3), position(3), velocity(3),
    //                    bias_gyro(3), bias_accel(3)]
    // NOTE: rotation must be updated last — translation and velocity
    // read the old dR_dbg.

    // translation
    A.block<3, 3>(3, 9) = (data->dv_dbg * dt + 0.5 * Rs_T * skew_accel * Rs *
                                                   data->dR_dbg * dt * dt);
    A.block<3, 3>(3, 12) = data->dv_dba * dt - 0.5 * Rs_T * dt * dt;
    data->dp_dbg += A.block<3, 3>(3, 9);
    data->dp_dba += A.block<3, 3>(3, 12);

    // velocity
    A.block<3, 3>(6, 9) = Rs_T * skew_accel * Rs * data->dR_dbg * dt;
    A.block<3, 3>(6, 12) = -Rs_T * dt;
    data->dv_dbg += A.block<3, 3>(6, 9);
    data->dv_dba += A.block<3, 3>(6, 12);

    // rotation: bias Jacobian transport (right-perturbation, additive).
    // From BCH: Exp(-w*dt + dbg*dt) ≈ Exp(-w*dt) * Exp(Jl(w*dt) * dbg * dt).
    // Right-perturbation transport gives the additive update:
    //   dR_dbg_{k+1} = dR_dbg_k + delta_R_k^T * Jl(w*dt) * dt
    const Eigen::Matrix3d Jl = LeftJacobianFromAngleAxis(gyro_true * dt);
    const Eigen::Matrix3d dR_dbg_updated = data->dR_dbg + Rs_T * Jl * dt;
    A.block<3, 3>(0, 9) = dR_dbg_updated - data->dR_dbg;
    data->dR_dbg = dR_dbg_updated;

    // propagate
    data->covariance = A * data->covariance * A.transpose();

    // Step 2: add noise.
    const double vars_v = accel_noise_density * accel_noise_density * dt;
    const double vars_omega = gyro_noise_density * gyro_noise_density * dt;
    const double vars_p =
        0.5 * vars_v * dt * dt + options.integration_noise_density *
                                     options.integration_noise_density * dt;
    const double vars_ba = calib.bias_accel_random_walk_sigma *
                           calib.bias_accel_random_walk_sigma * dt;
    const double vars_bg = calib.bias_gyro_random_walk_sigma *
                           calib.bias_gyro_random_walk_sigma * dt;
    data->covariance.block<3, 3>(0, 0).diagonal().array() += vars_omega;
    data->covariance.block<3, 3>(3, 3).diagonal().array() += vars_p;
    data->covariance.block<3, 3>(6, 6).diagonal().array() += vars_v;
    data->covariance.block<3, 3>(9, 9).diagonal().array() += vars_bg;
    data->covariance.block<3, 3>(12, 12).diagonal().array() += vars_ba;
  }
};

// Closed-form rotation integrals with analytical bias Jacobians and RK4
// covariance propagation.
class Rk4ImuIntegrator : public ImuIntegrator {
 public:
  void Integrate(const ImuPreintegrationOptions& options,
                 const ImuCalibration& calib,
                 const Eigen::Vector3d& accel_true,
                 const Eigen::Vector3d& gyro_true,
                 double dt,
                 double accel_noise_density,
                 double gyro_noise_density,
                 PreintegratedImuData* data) const override {
    // Left convention with closed-form rotation integrals, analytical bias
    // Jacobians, and RK4 covariance propagation.
    //
    // Based on Eckenhoff et al. "Closed-form Preintegration Methods for
    // Graph-based Visual-Inertial Navigation", IJRR 2018 (left convention).

    const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
    const double dt2 = dt * dt;

    // Angular velocity quantities.
    const double mag_w = gyro_true.norm();
    const double w_dt = mag_w * dt;
    const bool small_w = (mag_w < 1e-6);
    const double cos_wt = std::cos(w_dt);
    const double sin_wt = std::sin(w_dt);

    const Eigen::Matrix3d w_x = CrossProductMatrix(gyro_true);
    const Eigen::Matrix3d a_x = CrossProductMatrix(accel_true);
    const Eigen::Matrix3d w_x_2 = w_x * w_x;

    //==========================================================================
    // Step 1: State update (closed-form rotation + analytical integrals).
    //==========================================================================

    // Incremental rotation Exp(-w*dt) via Rodrigues formula.
    // Body-from-world evolves as R_BW_{k+1} = Exp(-w*dt) * R_BW_k.
    const Eigen::Matrix3d dR =
        small_w ? I3 - dt * w_x + (dt2 / 2) * w_x_2
                : I3 - (sin_wt / mag_w) * w_x +
                      ((1.0 - cos_wt) / (mag_w * mag_w)) * w_x_2;

    // Accumulated rotation before and after update.
    const Eigen::Matrix3d Rs = data->delta_R.toRotationMatrix();
    const Eigen::Matrix3d Rs_new = dR * Rs;

    // Closed-form integral coefficients for position and velocity.
    double f_1, f_2, f_3, f_4;
    if (small_w) {
      f_1 = -(dt2 * dt / 3);
      f_2 = (dt2 * dt2 / 8);
      f_3 = -(dt2 / 2);
      f_4 = (dt2 * dt / 6);
    } else {
      const double mag_w2 = mag_w * mag_w;
      f_1 = (w_dt * cos_wt - sin_wt) / (mag_w2 * mag_w);
      f_2 = (w_dt * w_dt - 2 * cos_wt - 2 * w_dt * sin_wt + 2) /
            (2 * mag_w2 * mag_w2);
      f_3 = -(1 - cos_wt) / mag_w2;
      f_4 = (w_dt - sin_wt) / (mag_w2 * mag_w);
    }

    // Integration matrices for position (H_p) and velocity (H_v).
    // Closed-form integral coefficients (same signs as Eckenhoff IJRR 2018).
    const Eigen::Matrix3d alpha_arg =
        (0.5 * dt2) * I3 + f_1 * w_x + f_2 * w_x_2;
    const Eigen::Matrix3d beta_arg = dt * I3 + f_3 * w_x + f_4 * w_x_2;
    // Rotation for closed-form integrals: Rs_new^T = (Exp(-w*dt) * Rs)^T.
    // This rotates the integral result into the body_i reference frame.
    const Eigen::Matrix3d R_integral = Rs_new.transpose();
    const Eigen::Matrix3d H_p = R_integral * alpha_arg;
    const Eigen::Matrix3d H_v = R_integral * beta_arg;

    // Update state.
    data->delta_p += data->delta_v * dt + H_p * accel_true;
    data->delta_v += H_v * accel_true;
    data->delta_R = Eigen::Quaterniond(Rs_new);
    data->delta_t += dt;

    //==========================================================================
    // Step 2: Bias Jacobians (Eckenhoff analytical).
    //==========================================================================
    // Analytical derivatives of the closed-form integrals w.r.t. gyro/accel
    // biases. Based on Eckenhoff et al. IJRR 2018.

    const Eigen::Matrix3d Rs_T = Rs.transpose();

    // Accel bias Jacobians (bias convention directly).
    data->dp_dba += dt * data->dv_dba - H_p;
    data->dv_dba -= H_v;

    // Rotation bias Jacobian (additive transport, bias convention).
    // dR_dbg_{k+1} = dR_dbg_k + delta_R_k^T * Jl(w*dt) * dt
    Eigen::Matrix3d Jl = LeftJacobianFromAngleAxis(gyro_true * dt);

    data->dR_dbg += Rs_T * Jl * dt;

    // The Eckenhoff d_R_bw formula expects the multiplicative-transport J_q
    // (≈ Jr(w*T)*T), which is the transpose of our data->dR_dbg (≈ Jl(w*T)*T).
    const Eigen::Matrix3d J_q = data->dR_dbg.transpose();

    // Derivatives of R_integral (Eckenhoff d_R_bw).
    const std::array<Eigen::Matrix3d, 3> d_R_bw = {
        -R_integral * CrossProductMatrix(J_q.col(0)),
        -R_integral * CrossProductMatrix(J_q.col(1)),
        -R_integral * CrossProductMatrix(J_q.col(2))};

    // Derivatives of f1-f4 coefficients w.r.t. omega components.
    double df_dw[4][3];
    {
      double g[4];
      if (small_w) {
        g[0] = -(dt2 * dt2 * dt / 15);
        g[1] = (dt2 * dt2 * dt2 / 72);
        g[2] = -(dt2 * dt2 / 12);
        g[3] = (dt2 * dt2 * dt / 60);
      } else {
        const double mw2 = mag_w * mag_w;
        const double mw3 = mw2 * mag_w;
        const double mw4 = mw2 * mw2;
        const double mw5 = mw3 * mw2;
        const double mw6 = mw3 * mw3;
        g[0] = (w_dt * w_dt * sin_wt - 3 * sin_wt + 3 * w_dt * cos_wt) / mw5;
        g[1] = (w_dt * w_dt - 4 * cos_wt - 4 * w_dt * sin_wt +
                w_dt * w_dt * cos_wt + 4) /
               mw6;
        g[2] = (2 * (cos_wt - 1) + w_dt * sin_wt) / mw4;
        g[3] = (2 * w_dt + w_dt * cos_wt - 3 * sin_wt) / mw5;
      }
      for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < 4; ++j) {
          df_dw[j][k] = gyro_true(k) * g[j];
        }
      }
    }

    // Gyro bias Jacobians: position and velocity (Eckenhoff IJRR 2018).
    // TODO: The d_R_bw term (-R * [J_q * e_k]_x) introduces a first-order
    // approximation that causes ~1e-5 absolute error on dp_dbg diagonal
    // entries where d_R_bw and df contributions nearly cancel.
    data->dp_dbg += data->dv_dbg * dt;
    for (int k = 0; k < 3; ++k) {
      const Eigen::Matrix3d& e_kx = kEx[k];
      const Eigen::Matrix3d d_alpha_dw_k =
          d_R_bw[k] * alpha_arg +
          R_integral * (df_dw[0][k] * w_x - f_1 * e_kx + df_dw[1][k] * w_x_2 -
                        f_2 * (e_kx * w_x + w_x * e_kx));
      const Eigen::Matrix3d d_beta_dw_k =
          d_R_bw[k] * beta_arg +
          R_integral * (df_dw[2][k] * w_x - f_3 * e_kx + df_dw[3][k] * w_x_2 -
                        f_4 * (e_kx * w_x + w_x * e_kx));
      data->dp_dbg.col(k) += d_alpha_dw_k * accel_true;
      data->dv_dbg.col(k) += d_beta_dw_k * accel_true;
    }

    //==========================================================================
    // Step 3: Covariance propagation (RK4).
    //==========================================================================

    // Continuous-time noise matrix Q_c (12x12):
    // [gyro_noise(3), gyro_walk(3), accel_noise(3), accel_walk(3)].
    Eigen::Matrix<double, 12, 12> Q_c = Eigen::Matrix<double, 12, 12>::Zero();
    Q_c.block<3, 3>(0, 0).diagonal().setConstant(gyro_noise_density *
                                                 gyro_noise_density);
    Q_c.block<3, 3>(3, 3).diagonal().setConstant(
        calib.bias_gyro_random_walk_sigma * calib.bias_gyro_random_walk_sigma);
    Q_c.block<3, 3>(6, 6).diagonal().setConstant(accel_noise_density *
                                                 accel_noise_density);
    Q_c.block<3, 3>(9, 9).diagonal().setConstant(
        calib.bias_accel_random_walk_sigma *
        calib.bias_accel_random_walk_sigma);

    // Continuous-time error-state Jacobian F (15x15) and noise mapping G
    // (15x12).
    // Error state: [rotation(3), position(3), velocity(3),
    //               bias_gyro(3), bias_accel(3)]
    // Noise: [gyro_noise(3), gyro_walk(3), accel_noise(3), accel_walk(3)]
    auto build_F_G = [&](const Eigen::Matrix3d& R_eval)
        -> std::pair<Eigen::Matrix<double, 15, 15>,
                     Eigen::Matrix<double, 15, 12>> {
      Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Zero();
      F.block<3, 3>(0, 0) = -w_x;                       // d(rot)/d(rot)
      F.block<3, 3>(0, 9) = -I3;                        // d(rot)/d(bias_gyro)
      F.block<3, 3>(3, 6) = I3;                         // d(pos)/d(vel)
      F.block<3, 3>(6, 0) = -R_eval.transpose() * a_x;  // d(vel)/d(rot)
      F.block<3, 3>(6, 12) = -R_eval.transpose();       // d(vel)/d(bias_accel)

      Eigen::Matrix<double, 15, 12> G = Eigen::Matrix<double, 15, 12>::Zero();
      G.block<3, 3>(0, 0) = -I3;                  // rot ← gyro_noise
      G.block<3, 3>(9, 3) = I3;                   // bias_gyro ← gyro_walk
      G.block<3, 3>(6, 6) = -R_eval.transpose();  // vel ← accel_noise
      G.block<3, 3>(12, 9) = I3;                  // bias_accel ← accel_walk

      return {F, G};
    };

    // Midpoint rotation for k2/k3 evaluation: Exp(-w*dt/2) * Rs.
    const double half_w_dt = mag_w * 0.5 * dt;
    const Eigen::Matrix3d R_mid =
        small_w ? I3 - 0.5 * dt * w_x + (0.125 * dt2) * w_x_2
                : I3 - (std::sin(half_w_dt) / mag_w) * w_x +
                      ((1.0 - std::cos(half_w_dt)) / (mag_w * mag_w)) * w_x_2;
    const Eigen::Matrix3d R_eval_mid = R_mid * Rs;

    // k1: evaluate at start rotation.
    const auto [F1, G1] = build_F_G(Rs);
    const Eigen::Matrix<double, 15, 15> P_dot_1 =
        F1 * data->covariance + data->covariance * F1.transpose() +
        G1 * Q_c * G1.transpose();

    // k2: evaluate at midpoint rotation.
    const auto [F2, G2] = build_F_G(R_eval_mid);
    const Eigen::Matrix<double, 15, 15> P_2 =
        data->covariance + P_dot_1 * dt / 2.0;
    const Eigen::Matrix<double, 15, 15> P_dot_2 =
        F2 * P_2 + P_2 * F2.transpose() + G2 * Q_c * G2.transpose();

    // k3: same F as k2 (same midpoint).
    const Eigen::Matrix<double, 15, 15> P_3 =
        data->covariance + P_dot_2 * dt / 2.0;
    const Eigen::Matrix<double, 15, 15> P_dot_3 =
        F2 * P_3 + P_3 * F2.transpose() + G2 * Q_c * G2.transpose();

    // k4: evaluate at end rotation.
    const auto [F4, G4] = build_F_G(Rs_new);
    const Eigen::Matrix<double, 15, 15> P_4 = data->covariance + P_dot_3 * dt;
    const Eigen::Matrix<double, 15, 15> P_dot_4 =
        F4 * P_4 + P_4 * F4.transpose() + G4 * Q_c * G4.transpose();

    // Combine RK4 increments.
    data->covariance +=
        (dt / 6.0) * (P_dot_1 + 2.0 * P_dot_2 + 2.0 * P_dot_3 + P_dot_4);
    // Add integration noise to position covariance (same as midpoint path).
    data->covariance.block<3, 3>(3, 3).diagonal().array() +=
        options.integration_noise_density * options.integration_noise_density *
        dt;
    data->covariance = 0.5 * (data->covariance + data->covariance.transpose());
  }
};

}  // namespace

std::unique_ptr<const ImuIntegrator> ImuIntegrator::Create(
    ImuIntegrationMethod method) {
  switch (method) {
    case ImuIntegrationMethod::MIDPOINT:
      return std::make_unique<MidpointImuIntegrator>();
    case ImuIntegrationMethod::RK4:
      return std::make_unique<Rk4ImuIntegrator>();
  }
  LOG(FATAL) << "Unhandled ImuIntegrationMethod";
  return nullptr;
}

void ImuPreintegrator::IntegrateOneMeasurement(const ImuMeasurement& prev,
                                               const ImuMeasurement& curr) {
  Eigen::Vector3d accel_s = prev.accel;
  Eigen::Vector3d gyro_s = prev.gyro;
  Eigen::Vector3d accel_e = curr.accel;
  Eigen::Vector3d gyro_e = curr.gyro;

  // Get dt and update boundaries.
  const timestamp_t interval_t_start = std::max(prev.timestamp, t_start_);
  const timestamp_t interval_t_end = std::min(curr.timestamp, t_end_);
  const double dt = TimestampDiffSeconds(interval_t_end, interval_t_start);
  THROW_CHECK_GT(dt, 0.0);
  const double imu_dt = TimestampDiffSeconds(curr.timestamp, prev.timestamp);

  // Interpolate at boundaries if needed.
  Eigen::Vector3d accel_s_tmp = accel_s;
  Eigen::Vector3d gyro_s_tmp = gyro_s;
  Eigen::Vector3d accel_e_tmp = accel_e;
  Eigen::Vector3d gyro_e_tmp = gyro_e;
  if (interval_t_start > prev.timestamp) {
    const double ratio_s =
        TimestampDiffSeconds(interval_t_start, prev.timestamp) / imu_dt;
    accel_s_tmp = (1.0 - ratio_s) * accel_s + ratio_s * accel_e;
    gyro_s_tmp = (1.0 - ratio_s) * gyro_s + ratio_s * gyro_e;
  }
  if (interval_t_end < curr.timestamp) {
    const double ratio_e =
        TimestampDiffSeconds(interval_t_end, prev.timestamp) / imu_dt;
    accel_e_tmp = (1.0 - ratio_e) * accel_s + ratio_e * accel_e;
    gyro_e_tmp = (1.0 - ratio_e) * gyro_s + ratio_e * gyro_e;
  }
  accel_s = accel_s_tmp;
  gyro_s = gyro_s_tmp;
  accel_e = accel_e_tmp;
  gyro_e = gyro_e_tmp;

  Eigen::Vector3d accel_true = 0.5 * (accel_s + accel_e) - biases_.tail<3>();
  accel_true = accel_rect_mat_inv_ * accel_true;
  Eigen::Vector3d gyro_true = 0.5 * (gyro_s + gyro_e) - biases_.head<3>();
  gyro_true = gyro_rect_mat_inv_ * gyro_true;

  // Check saturation.
  double accel_noise_density = calib_.accel_noise_density;
  if (accel_s.cwiseAbs().maxCoeff() > calib_.accel_saturation_max ||
      accel_e.cwiseAbs().maxCoeff() > calib_.accel_saturation_max) {
    accel_noise_density *= 100.0;
  }
  double gyro_noise_density = calib_.gyro_noise_density;
  if (gyro_s.cwiseAbs().maxCoeff() > calib_.gyro_saturation_max ||
      gyro_e.cwiseAbs().maxCoeff() > calib_.gyro_saturation_max) {
    gyro_noise_density *= 100.0;
  }

  integrator_->Integrate(options_,
                         calib_,
                         accel_true,
                         gyro_true,
                         dt,
                         accel_noise_density,
                         gyro_noise_density,
                         &data_);
}

void ImuPreintegrator::FeedImu(const ImuMeasurement& m) {
  // Check if this is the first measurement
  if (!HasStarted()) {
    THROW_CHECK_LE(m.timestamp, t_start_)
        << "The timestamp of the first IMU measurement should not be later "
           "than the start of integration";
    measurements_.Insert(m);
    has_started_ = true;
    return;
  }

  // Assertion check: the new measurement needs to be later than measurement.
  ImuMeasurement last_measurement = measurements_.back();
  THROW_CHECK_GT(m.timestamp, last_measurement.timestamp);
  if (m.timestamp <= t_start_) {
    LOG(WARNING) << "The timestamp of this measurement is earlier than "
                    "t_start. Ignore the previous measurements.";
    measurements_.Clear();
    measurements_.Insert(m);
    return;
  }
  if (last_measurement.timestamp >= t_end_) {
    LOG(WARNING) << "The timestamp of the last measurement has already reached "
                    "t_end. Ignore the current measurement.";
    return;
  }

  // Append measurements
  measurements_.Insert(m);
  IntegrateOneMeasurement(last_measurement, m);
}

void ImuPreintegrator::FeedImu(const ImuMeasurements& ms) {
  for (const auto& m : ms) {
    FeedImu(m);
  }
}

PreintegratedImuData ImuPreintegrator::Extract() {
  data_.Finalize(options_.max_condition_number);
  return data_;
}

void ImuPreintegrator::Update(PreintegratedImuData* data) { *data = data_; }

void ImuPreintegrator::Reintegrate() {
  Reset();
  has_started_ = true;
  for (size_t i = 1; i < measurements_.Size(); ++i) {
    IntegrateOneMeasurement(measurements_[i - 1], measurements_[i]);
  }
  data_.Finalize(options_.max_condition_number);
}

void ImuPreintegrator::Reintegrate(const Eigen::Vector6d& biases) {
  SetLinearizationBiases(biases);
  Reintegrate();
}

void ImuReintegrationCallback::AddEdge(ImuPreintegrator* integrator,
                                       PreintegratedImuData* data,
                                       const double* imu_state) {
  edges_.push_back({integrator, data, imu_state});
}

bool ImuReintegrationCallback::ShouldReintegrate(
    const PreintegratedImuData& data, const Eigen::Vector6d& biases) const {
  const Eigen::Vector6d diff = biases - data.biases;
  const double delta_t = data.delta_t;
  if (diff.head<3>().norm() * delta_t > options_.reintegrate_angle_norm_thres) {
    return true;
  }
  if (diff.tail<3>().norm() * delta_t > options_.reintegrate_vel_norm_thres) {
    return true;
  }
  return false;
}

ceres::CallbackReturnType ImuReintegrationCallback::operator()(
    const ceres::IterationSummary& /*summary*/) {
  for (auto& edge : edges_) {
    // Read current biases from the optimized IMU state: [v(3), bg(3), ba(3)].
    const Eigen::Vector6d biases(edge.imu_state + 3);
    if (ShouldReintegrate(*edge.data, biases)) {
      edge.integrator->Reintegrate(biases);
      edge.integrator->Update(edge.data);
    }
  }
  return ceres::SOLVER_CONTINUE;
}

}  // namespace colmap

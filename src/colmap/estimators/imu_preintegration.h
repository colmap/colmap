// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/estimators/cost_functions.h"
#include "colmap/geometry/pose.h"
#include "colmap/sensor/imu.h"

#include <memory>
#include <unordered_set>

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

struct ImuPreintegrationOptions {
  // check whether to reintegrate
  double reintegrate_vel_norm_thres = 0.0001;
};

class PreintegratedImuMeasurement {
 public:
  PreintegratedImuMeasurement(const ImuPreintegrationOptions& options,
                              const ImuCalibration& calib,
                              double t_start,
                              double t_end);
  ~PreintegratedImuMeasurement() = default;

  // Reset
  void Reset();
  bool HasStarted() const;

  // Set biases
  void SetBiases(const Eigen::Vector6d& biases);

  // Add measurements: measurements need to be added in chronological order
  void AddMeasurement(const ImuMeasurement& m);
  void AddMeasurements(const ImuMeasurements& ms);
  void Finish();
  bool HasFinished() const;

  // Reintegrate
  bool CheckReintegrate(const Eigen::Vector6d& biases) const;
  void Reintegrate();
  void Reintegrate(
      const Eigen::Vector6d& biases);  // SetBiases(biases) + Reintegrate().

  // data interfaces
  const double DeltaT() const;
  const Eigen::Quaterniond& DeltaR() const;
  const Eigen::Vector3d& DeltaP() const;
  const Eigen::Vector3d& DeltaV() const;
  const Eigen::Matrix3d dR_dbg() const;
  const Eigen::Matrix3d dp_dba() const;
  const Eigen::Matrix3d dp_dbg() const;
  const Eigen::Matrix3d dv_dba() const;
  const Eigen::Matrix3d dv_dbg() const;
  const Eigen::Vector6d& Biases() const;
  const Eigen::Matrix<double, 15, 15> LMatrix() const;
  const double GravityMagnitude() const;
  const ImuMeasurements Measurements() const;

 private:
  // Options
  ImuPreintegrationOptions options_;

  // IMU Calibration
  ImuCalibration calib_;

  // Clock
  double t_start_ = 0.0;  // from what time to start the integration.
  double t_end_ = 0.0;    // until what time to do the integration.

  // Flag to check if the first measurement has already been added.
  bool has_started_ = false;

  // Flag to check if LLT is performed
  bool has_finished_ = false;

  // Preintegrated measurements (imu to gravity-aligned metric world)
  Eigen::Quaterniond delta_R_ij_ =
      Eigen::Quaterniond::Identity();                     // relative rotation
  Eigen::Vector3d delta_p_ij_ = Eigen::Vector3d::Zero();  // position changes
  Eigen::Vector3d delta_v_ij_ = Eigen::Vector3d::Zero();  // velocity changes
  double delta_t_ = 0;                                    // accumulated time
  Eigen::Vector6d biases_ =
      Eigen::Vector6d::Zero();  // bias on acc (3-DoF) + gyro (3-DoF)

  // Accounting for bias changes
  Eigen::Matrix<double, 9, 6> jacobian_biases_ =
      Eigen::Matrix<double, 9, 6>::Zero();  // jacobian of rotation,
                                            // translation, velocity over biases

  // Covariance propagation
  Eigen::Matrix<double, 15, 15> covs_ =
      Eigen::Matrix<double, 15, 15>::Zero();  // covariances (rotation +
                                              // translation + velocity + acc
                                              // bias + gyro bias)

  // LLT decomposition of the information matrix for least squares optimization
  Eigen::Matrix<double, 15, 15> L_matrix_;

  // Measurements
  ImuMeasurements measurements_;

  // Methods
  void integrate(const Eigen::Vector3d& acc_true,
                 const Eigen::Vector3d& gyro_true,
                 double dt,
                 double acc_noise_density,
                 double gyro_noise_density);
};

class PreintegratedImuMeasurementCostFunction {
 public:
  explicit PreintegratedImuMeasurementCostFunction(
      const PreintegratedImuMeasurement& m)
      : measurement_(m) {
    if (measurement_.HasFinished()) measurement_.Finish();
  }

  static ceres::CostFunction* Create(const PreintegratedImuMeasurement& m) {
    return (
        new ceres::AutoDiffCostFunction<PreintegratedImuMeasurementCostFunction,
                                        15,
                                        4,
                                        3,
                                        1,
                                        3,
                                        4,
                                        3,
                                        9,
                                        4,
                                        3,
                                        9>(
            new PreintegratedImuMeasurementCostFunction(m)));
  }

  template <typename T>
  bool operator()(const T* const cam_to_imu_q,
                  const T* const cam_to_imu_t,
                  const T* const log_scale,
                  const T* const gravity_direction,
                  const T* const i_from_world_q,
                  const T* const i_from_world_t,
                  const T* const i_imu_state,
                  const T* const j_from_world_q,
                  const T* const j_from_world_t,
                  const T* const j_imu_state,
                  T* residuals) const {
    // Check and perform reintegration when needed
    Eigen::Vector6d biases_double;
    for (size_t i = 0; i < 6; ++i) {
      biases_double(i) = ConvertToDouble<T>::convert(i_imu_state[i + 3]);
    }
    if (measurement_.CheckReintegrate(biases_double))
      measurement_.Reintegrate(biases_double);
    // Compute residuals
    // imu state
    EigenVector3Map<T> v_i_data(i_imu_state);
    EigenVector3Map<T> v_j_data(j_imu_state);
    Eigen::Matrix<T, 6, 1> delta_b =
        Eigen::Map<const Eigen::Matrix<T, 6, 1>>(i_imu_state + 3) -
        measurement_.Biases().cast<T>();
    EigenVector3Map<T> delta_b_a(delta_b.data());
    EigenVector3Map<T> delta_b_g(delta_b.data() + 3);
    const T dt = T(measurement_.DeltaT());
    Eigen::Matrix<T, 3, 1> gravity = EigenVector3Map<T>(gravity_direction) *
                                     T(measurement_.GravityMagnitude());

    // change frame (measure the extrinsics from imu to metric)
    // T_imu_to_metric = T_imu_to_cam * T_cam_to_world * T_world_to_metric
    Eigen::Quaternion<T> imu_to_cam_q =
        EigenQuaternionMap<T>(cam_to_imu_q).inverse();
    Eigen::Matrix<T, 3, 1> imu_to_cam_t =
        imu_to_cam_q * EigenVector3Map<T>(cam_to_imu_t) * T(-1.);
    Eigen::Quaternion<T> i_to_world_q =
        EigenQuaternionMap<T>(i_from_world_q).inverse();
    Eigen::Matrix<T, 3, 1> i_to_world_t =
        i_to_world_q * EigenVector3Map<T>(i_from_world_t) * T(-1.);
    Eigen::Quaternion<T> j_to_world_q =
        EigenQuaternionMap<T>(j_from_world_q).inverse();
    Eigen::Matrix<T, 3, 1> j_to_world_t =
        j_to_world_q * EigenVector3Map<T>(j_from_world_t) * T(-1.);
    // compose
    Eigen::Quaternion<T> i_imu_to_world_q = imu_to_cam_q * i_to_world_q;
    Eigen::Matrix<T, 3, 1> i_imu_to_world_t =
        imu_to_cam_q * i_to_world_t + imu_to_cam_t;
    Eigen::Quaternion<T> j_imu_to_world_q = imu_to_cam_q * j_to_world_q;
    Eigen::Matrix<T, 3, 1> j_imu_to_world_t =
        imu_to_cam_q * j_to_world_t + imu_to_cam_t;
    // scale
    T scale = ceres::exp(log_scale[0]);
    i_imu_to_world_t = i_imu_to_world_t * scale;
    j_imu_to_world_t = j_imu_to_world_t * scale;
    // velocities should be multiplied with scale as well
    Eigen::Matrix<T, 3, 1> v_i = v_i_data * scale;
    Eigen::Matrix<T, 3, 1> v_j = v_j_data * scale;

    // Eq. (44) and (45) from Forster et al. "On-Manifold Preintegration for
    // Real-time Visual-Inertial Odometry" TRO 16. rotation: residuals[0:3]
    const Eigen::Quaternion<T> j_from_i_q =
        i_imu_to_world_q.inverse() * j_imu_to_world_q;
    Eigen::Matrix<T, 3, 1> omega_bias =
        measurement_.dR_dbg().cast<T>() * delta_b_g;
    Eigen::Quaternion<T> Dq_bias;
    EigenAngleAxisToQuaternion(omega_bias.data(), Dq_bias.coeffs().data());
    const Eigen::Quaternion<T> Dq = measurement_.DeltaR().cast<T>() * Dq_bias;
    const Eigen::Quaternion<T> param_from_measured_q =
        Dq.inverse() * j_from_i_q;
    EigenQuaternionToAngleAxis(param_from_measured_q.coeffs().data(),
                               residuals);
    // translation: residuals[3:6]
    const Eigen::Matrix<T, 3, 1> j_from_i_p =
        j_imu_to_world_t - i_imu_to_world_t;
    Eigen::Matrix<T, 3, 1> est_dp =
        i_imu_to_world_q.inverse() *
        (j_from_i_p - v_i * dt - 0.5 * gravity * dt * dt);
    Eigen::Matrix<T, 3, 1> Dp = measurement_.DeltaP().cast<T>() +
                                measurement_.dp_dba().cast<T>() * delta_b_a +
                                measurement_.dp_dbg().cast<T>() * delta_b_g;
    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_measured_p(residuals + 3);
    param_from_measured_p = est_dp - Dp;

    // velocity: residuals[6:9]
    const Eigen::Matrix<T, 3, 1> j_from_i_v = v_j - v_i;
    Eigen::Matrix<T, 3, 1> est_dv =
        i_imu_to_world_q.inverse() * (j_from_i_v - gravity * dt);
    Eigen::Matrix<T, 3, 1> Dv = measurement_.DeltaV().cast<T>() +
                                measurement_.dv_dba().cast<T>() * delta_b_a +
                                measurement_.dv_dbg().cast<T>() * delta_b_g;
    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_measured_v(residuals + 6);
    param_from_measured_v = est_dv - Dv;

    // bias: residuals[9:15]
    for (size_t i = 0; i < 6; ++i) {
      residuals[i + 9] = j_imu_state[i + 3] - i_imu_state[i + 3];
    }

    // Weight by the covariance inverse
    Eigen::Map<Eigen::Matrix<T, 15, 1>> residuals_data(residuals);
    residuals_data.applyOnTheLeft(measurement_.LMatrix().cast<T>());
    return true;
  }

 private:
  // TODO: dubious design on making it mutable. But no other way out up to now
  mutable PreintegratedImuMeasurement measurement_;

  // Convert from type T (including ceres::Jet) to double
  // default case: return the value directly
  template <typename T>
  struct ConvertToDouble {
    static double convert(const T& value) { return static_cast<double>(value); }
  };
  // specialization for Jet type
  template <typename Scalar, int N>
  struct ConvertToDouble<ceres::Jet<Scalar, N>> {
    static double convert(const ceres::Jet<Scalar, N>& value) {
      return value.a;
    }
  };
};

}  // namespace colmap

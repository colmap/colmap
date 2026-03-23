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

#include "colmap/sensor/imu.h"
#include "colmap/util/enum_utils.h"
#include "colmap/util/types.h"

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

namespace colmap {

// Midpoint: trapezoidal integration, sufficient for typical 200-1000 Hz IMUs.
// RK4: closed-form rotation integrals with analytical bias Jacobians and
//      RK4 covariance propagation, more accurate for lower IMU rates.
MAKE_ENUM_CLASS(ImuIntegrationMethod, 0, MIDPOINT, RK4);

struct ImuPreintegrationOptions {
  ImuIntegrationMethod method = ImuIntegrationMethod::RK4;

  // Whether to add integration noise to the covariance propagation.
  bool use_integration_noise = false;
  // Integration noise density. [m/s*1/sqrt(Hz)]
  double integration_noise_density = 0.2;

  // Threshold on velocity bias change norm to trigger reintegration. [m/s]
  double reintegrate_vel_norm_thres = 0.0001;
  // Threshold on gyro bias change norm to trigger reintegration. [rad/s]
  double reintegrate_angle_norm_thres = 0.0001;
};

// Pure data struct holding preintegrated IMU quantities.
// Serializable, trivially copyable across threads, and consumable by
// different cost functions without knowledge of the integration algorithm.
//
// Left-multiply integration convention:
//   delta_R = body_from_world_j * world_from_body_i
//             (accumulated via left-multiply: delta_R_{k+1} = dR * delta_R_k)
//   delta_p = delta_R^T * (p_W_j - p_W_i - v_W_i * dt - 0.5 * g * dt^2)
//   delta_v = delta_R^T * (v_W_j - v_W_i - g * dt)
//
// Right-perturbation model for bias Jacobians:
//   delta_R(bg + dbg) ≈ delta_R * Exp(dR_dbg * dbg)
//   The bias correction is always right-multiplied in the cost function.
struct PreintegratedImuData {
  // Preintegrated deltas.
  double delta_t = 0;  // Accumulated time. [seconds]
  Eigen::Quaterniond delta_R =
      Eigen::Quaterniond::Identity();                 // Relative rotation.
  Eigen::Vector3d delta_p = Eigen::Vector3d::Zero();  // Position change.
  Eigen::Vector3d delta_v = Eigen::Vector3d::Zero();  // Velocity change.

  // Bias Jacobians: derivatives of preintegrated [rotation, position, velocity]
  // w.r.t. [bias_gyro, bias_accel].
  Eigen::Matrix3d dR_dbg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_dbg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_dbg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_dba = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_dba = Eigen::Matrix3d::Zero();

  // Linearization biases: [bias_gyro(3), bias_accel(3)].
  Eigen::Vector6d biases = Eigen::Vector6d::Zero();

  // Covariance of the 15-dimensional state:
  // [rotation(3), position(3), velocity(3), bias_gyro(3), bias_accel(3)].
  Eigen::Matrix<double, 15, 15> covariance =
      Eigen::Matrix<double, 15, 15>::Zero();

  // Square root of the information matrix (inverse covariance), computed
  // via LLT decomposition in Finalize().
  Eigen::Matrix<double, 15, 15> sqrt_information =
      Eigen::Matrix<double, 15, 15>::Zero();

  // Gravity magnitude used during preintegration.
  double gravity_magnitude = 9.81;

  // Compute sqrt_information from covariance.
  void Finalize();
};

// Algorithm class that performs IMU preintegration.
// Owns the raw measurements, calibration, and integration options.
// Produces a PreintegratedImuData data struct via Extract().
class ImuPreintegrator {
 public:
  ImuPreintegrator(const ImuPreintegrationOptions& options,
                   const ImuCalibration& calib,
                   timestamp_t t_start,
                   timestamp_t t_end);
  ~ImuPreintegrator() = default;

  // Reset the integrator state.
  void Reset();

  // Set linearization biases.
  void SetBiases(const Eigen::Vector6d& biases);

  // Feed measurements. Must be added in chronological order.
  void FeedImu(const ImuMeasurement& m);
  void FeedImu(const std::vector<ImuMeasurement>& ms);

  // Extract the preintegrated data struct. Calls Finalize() internally.
  PreintegratedImuData Extract();

  // Copy the current (finalized) integration result into an existing data
  // struct. Use after Reintegrate() to update data that a cost function
  // references by pointer, so the cost function sees the new values.
  void Update(PreintegratedImuData* data);

  // Check if reintegration is needed and perform it.
  bool ShouldReintegrate(const Eigen::Vector6d& biases) const;
  void Reintegrate();
  void Reintegrate(const Eigen::Vector6d& biases);

  // State queries.
  bool HasStarted() const { return has_started_; }
  const ImuMeasurements& Measurements() const { return measurements_; }

 private:
  void IntegrateOneMeasurement(const ImuMeasurement& prev,
                               const ImuMeasurement& curr);

  void IntegrateMidpoint(const Eigen::Vector3d& accel_true,
                         const Eigen::Vector3d& gyro_true,
                         double dt,
                         double accel_noise_density,
                         double gyro_noise_density);

  void IntegrateRK4(const Eigen::Vector3d& accel_true,
                    const Eigen::Vector3d& gyro_true,
                    double dt,
                    double accel_noise_density,
                    double gyro_noise_density);

  // Integration time window. [nanoseconds]
  timestamp_t t_start_ = kInvalidTimestamp;
  timestamp_t t_end_ = kInvalidTimestamp;

  bool has_started_ = false;

  // Accumulated preintegrated data.
  PreintegratedImuData data_;

  // Raw measurements, sorted by timestamp. Chronological order is enforced
  // by FeedImu() via THROW_CHECK_GT on consecutive timestamps.
  ImuMeasurements measurements_;

  // Options
  ImuPreintegrationOptions options_;

  // IMU Calibration.
  ImuCalibration calib_;
  Eigen::Matrix3d accel_rect_mat_inv_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d gyro_rect_mat_inv_ = Eigen::Matrix3d::Identity();
  Eigen::Vector6d biases_ =
      Eigen::Vector6d::Zero();  // [bias_gyro(3), bias_accel(3)]
};

// Ceres iteration callback that checks whether any IMU preintegration
// linearization point has drifted beyond threshold, and if so, reintegrates
// from raw measurements and updates the PreintegratedImuData in place.
//
// Usage:
//   ImuReintegrationCallback callback;
//   // For each IMU edge:
//   callback.AddEdge(&integrator, &data, imu_state_ptr);
//   // Then add to solver options:
//   solver_options.callbacks.push_back(&callback);
//   solver_options.update_state_every_iteration = true;
class ImuReintegrationCallback : public ceres::IterationCallback {
 public:
  // Register an IMU edge for reintegration checking.
  // @param integrator   The integrator holding raw measurements and options.
  // @param data         The preintegrated data consumed by the cost function.
  //                     Updated in place when reintegration is triggered.
  // @param imu_state    Pointer to the 9-element IMU state being optimized
  //                     [velocity(3), bias_gyro(3), bias_accel(3)].
  //                     Biases at offset 3 are read to decide reintegration.
  void AddEdge(ImuPreintegrator* integrator,
               PreintegratedImuData* data,
               const double* imu_state);

  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) override;

 private:
  struct Edge {
    ImuPreintegrator* integrator;
    PreintegratedImuData* data;
    const double* imu_state;
  };
  std::vector<Edge> edges_;
};

}  // namespace colmap

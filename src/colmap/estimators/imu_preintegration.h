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

#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/geometry/pose.h"
#include "colmap/sensor/imu.h"
#include "colmap/util/timestamp.h"

#include <memory>
#include <unordered_set>

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

struct ImuPreintegrationOptions {
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
struct PreintegratedImuData {
  // Preintegrated deltas (IMU to gravity-aligned metric world).
  double delta_t = 0;  // Accumulated time. [seconds]
  Eigen::Quaterniond delta_R =
      Eigen::Quaterniond::Identity();                 // Relative rotation.
  Eigen::Vector3d delta_p = Eigen::Vector3d::Zero();  // Position change.
  Eigen::Vector3d delta_v = Eigen::Vector3d::Zero();  // Velocity change.

  // Bias Jacobians: derivatives of preintegrated [rotation, position, velocity]
  // w.r.t. [gyro_bias, acc_bias].
  Eigen::Matrix3d dR_dbg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_dbg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_dbg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_dba = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_dba = Eigen::Matrix3d::Zero();

  // Linearization biases: [gyro_bias(3), acc_bias(3)].
  Eigen::Vector6d biases = Eigen::Vector6d::Zero();

  // Covariance of the 15-dimensional state:
  // [rotation(3), position(3), velocity(3), gyro_bias(3), acc_bias(3)].
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

  // Set rectification matrices and biases.
  void SetAccRectMat(const Eigen::Matrix3d& mat);
  void SetGyroRectMat(const Eigen::Matrix3d& mat);
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

  void Integrate(const Eigen::Vector3d& acc_true,
                 const Eigen::Vector3d& gyro_true,
                 double dt,
                 double acc_noise_density,
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
  Eigen::Matrix3d acc_rect_mat_inv_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d gyro_rect_mat_inv_ = Eigen::Matrix3d::Identity();
  Eigen::Vector6d biases_ =
      Eigen::Vector6d::Zero();  // bias on gyro (3-DoF) + acc (3-DoF)
};

// Cost function for IMU preintegration residuals.
// Takes a pointer to externally-owned PreintegratedImuData so that a
// ReintegrationCallback can update the data between Ceres iterations
// without rebuilding cost functions.
class PreintegratedImuMeasurementCostFunction {
 public:
  // Construct from a pointer to externally-owned data.
  // The pointed-to data must outlive this cost function.
  explicit PreintegratedImuMeasurementCostFunction(
      const PreintegratedImuData* data)
      : data_(data) {
    THROW_CHECK(!data_->sqrt_information.isZero())
        << "PreintegratedImuData must be finalized before use in cost "
           "function. Call Extract() or Update() on the integrator, or "
           "Finalize() on the data directly.";
  }

  static ceres::CostFunction* Create(const PreintegratedImuData* data) {
    return (
        new ceres::AutoDiffCostFunction<PreintegratedImuMeasurementCostFunction,
                                        15,
                                        7,
                                        1,
                                        3,
                                        7,
                                        9,
                                        7,
                                        9>(
            new PreintegratedImuMeasurementCostFunction(data)));
  }

  template <typename T>
  bool operator()(const T* const imu_from_cam,
                  const T* const log_scale,
                  const T* const gravity_direction,
                  const T* const i_from_world,
                  const T* const i_imu_state,
                  const T* const j_from_world,
                  const T* const j_imu_state,
                  T* residuals) const {
    // imu state
    EigenVector3Map<T> v_i_data(i_imu_state);
    EigenVector3Map<T> v_j_data(j_imu_state);
    Eigen::Matrix<T, 6, 1> delta_b =
        Eigen::Map<const Eigen::Matrix<T, 6, 1>>(i_imu_state + 3) -
        data_->biases.cast<T>();
    EigenVector3Map<T> delta_b_g(delta_b.data());
    EigenVector3Map<T> delta_b_a(delta_b.data() + 3);
    const T dt = T(data_->delta_t);
    Eigen::Matrix<T, 3, 1> gravity =
        EigenVector3Map<T>(gravity_direction) * T(data_->gravity_magnitude);

    // change frame (measure the extrinsics from imu to world)
    // T_world_from_imu = T_world_from_cam * T_cam_from_imu
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
    // compose
    Eigen::Quaternion<T> world_from_i_imu_q = world_from_i_q * cam_from_imu_q;
    Eigen::Matrix<T, 3, 1> world_from_i_imu_t =
        world_from_i_q * cam_from_imu_t + world_from_i_t;
    Eigen::Quaternion<T> world_from_j_imu_q = world_from_j_q * cam_from_imu_q;
    Eigen::Matrix<T, 3, 1> world_from_j_imu_t =
        world_from_j_q * cam_from_imu_t + world_from_j_t;
    // scale
    T scale = ceres::exp(log_scale[0]);
    world_from_i_imu_t = world_from_i_imu_t * scale;
    world_from_j_imu_t = world_from_j_imu_t * scale;
    // velocities should be multiplied with scale as well
    Eigen::Matrix<T, 3, 1> v_i = v_i_data * scale;
    Eigen::Matrix<T, 3, 1> v_j = v_j_data * scale;

    // Eq. (44) and (45) from Forster et al. "On-Manifold Preintegration for
    // Real-time Visual-Inertial Odometry" TRO 16. rotation: residuals[0:3]
    const Eigen::Quaternion<T> j_from_i_q =
        world_from_i_imu_q.inverse() * world_from_j_imu_q;
    Eigen::Matrix<T, 3, 1> omega_bias = data_->dR_dbg.cast<T>() * delta_b_g;
    Eigen::Quaternion<T> Dq_bias;
    AngleAxisToEigenQuaternion(omega_bias.data(), Dq_bias.coeffs().data());
    const Eigen::Quaternion<T> Dq = data_->delta_R.cast<T>() * Dq_bias;
    const Eigen::Quaternion<T> param_from_measured_q =
        (Dq.inverse() * j_from_i_q).normalized();
    EigenQuaternionToAngleAxis(param_from_measured_q.coeffs().data(),
                               residuals);
    // translation: residuals[3:6]
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

    // velocity: residuals[6:9]
    const Eigen::Matrix<T, 3, 1> j_from_i_v = v_j - v_i;
    Eigen::Matrix<T, 3, 1> est_dv =
        world_from_i_imu_q.inverse() * (j_from_i_v - gravity * dt);
    Eigen::Matrix<T, 3, 1> Dv = data_->delta_v.cast<T>() +
                                data_->dv_dba.cast<T>() * delta_b_a +
                                data_->dv_dbg.cast<T>() * delta_b_g;
    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_measured_v(residuals + 6);
    param_from_measured_v = est_dv - Dv;

    // bias: residuals[9:15]
    for (size_t i = 0; i < 6; ++i) {
      residuals[i + 9] = j_imu_state[i + 3] - i_imu_state[i + 3];
    }

    // Weight by the covariance inverse
    Eigen::Map<Eigen::Matrix<T, 15, 1>> residuals_data(residuals);
    residuals_data.applyOnTheLeft(data_->sqrt_information.cast<T>());
    return true;
  }

 private:
  const PreintegratedImuData* data_;
};

// Ceres iteration callback that checks whether any IMU preintegration
// linearization point has drifted beyond threshold, and if so, reintegrates
// from raw measurements and updates the PreintegratedImuData in place.
//
// Usage:
//   ReintegrationCallback callback;
//   // For each IMU edge:
//   callback.AddEdge(&integrator, &data, imu_state_ptr);
//   // Then add to solver options:
//   solver_options.callbacks.push_back(&callback);
//   solver_options.update_state_every_iteration = true;
class ReintegrationCallback : public ceres::IterationCallback {
 public:
  // Register an IMU edge for reintegration checking.
  // @param integrator   The integrator holding raw measurements and options.
  // @param data         The preintegrated data consumed by the cost function.
  //                     Updated in place when reintegration is triggered.
  // @param imu_state    Pointer to the 9-element IMU state being optimized
  //                     [velocity(3), gyro_bias(3), acc_bias(3)].
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

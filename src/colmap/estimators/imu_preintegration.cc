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

#include <cmath>

#include <Eigen/Dense>

namespace colmap {

void PreintegratedImuData::Finalize() {
  // Enforce symmetry and stability
  covariance = (covariance + covariance.transpose()) / 2.0;
  covariance += Eigen::Matrix<double, 15, 15>::Identity() * 1e-18;

  // Factorize
  sqrt_information = covariance.inverse().llt().matrixL().transpose();
}

ImuPreintegrator::ImuPreintegrator(const ImuPreintegrationOptions& options,
                                   const ImuCalibration& calib,
                                   timestamp_t t_start,
                                   timestamp_t t_end) {
  options_ = options;
  calib_ = calib;
  THROW_CHECK_LT(t_start, t_end);
  t_start_ = t_start;
  t_end_ = t_end;
  data_.gravity_magnitude = calib.gravity_magnitude;
  acc_rect_mat_inv_ = calib.acc_rectification.inverse();
  gyro_rect_mat_inv_ = calib.gyro_rectification.inverse();
  Reset();
}

void ImuPreintegrator::Reset() {
  data_.delta_R = Eigen::Quaterniond::Identity();
  data_.delta_p = Eigen::Vector3d::Zero();
  data_.delta_v = Eigen::Vector3d::Zero();
  data_.delta_t = 0;
  data_.dR_dbg = Eigen::Matrix3d::Zero();
  data_.dp_dbg = Eigen::Matrix3d::Zero();
  data_.dv_dbg = Eigen::Matrix3d::Zero();
  data_.dp_dba = Eigen::Matrix3d::Zero();
  data_.dv_dba = Eigen::Matrix3d::Zero();
  data_.covariance = Eigen::Matrix<double, 15, 15>::Zero();
  data_.sqrt_information = Eigen::Matrix<double, 15, 15>::Zero();
  data_.biases = biases_;

  has_started_ = false;
}

void ImuPreintegrator::SetBiases(const Eigen::Vector6d& biases) {
  biases_ = biases;
  data_.biases = biases;
}

void ImuPreintegrator::Integrate(const Eigen::Vector3d& acc_true,
                                 const Eigen::Vector3d& gyro_true,
                                 double dt,
                                 double acc_noise_density,
                                 double gyro_noise_density) {
  // [Reference]
  // [A] Forster et al. "On-Manifold Preintegration for Real-Time
  // Visual-Inertial Odometry", TRO 16. Integration step translation: Eq. (37)
  // from [A]
  data_.delta_p +=
      data_.delta_v * dt + data_.delta_R * acc_true * 0.5 * dt * dt;
  // velocity: Eq. (36) from [A]
  data_.delta_v += data_.delta_R * acc_true * dt;
  // rotation: Eq. (35) from [A]. Right convention:
  // delta_R_{k+1} = delta_R_k * Exp(omega * dt).
  Eigen::Quaterniond dq = QuaternionFromAngleAxis(gyro_true * dt);
  Eigen::Matrix3d Rs = data_.delta_R.toRotationMatrix();
  data_.delta_R = data_.delta_R * dq;
  // time
  data_.delta_t += dt;

  // Update jacobians over bias
  // [Reference] end of Appendix B from [A]. Since it is not tagged with
  // equation number, we refer it as Eq. (69 1/2) in the following.
  Eigen::Matrix3d Jr = RightJacobianFromAngleAxis(gyro_true * dt);
  Eigen::Matrix3d skew_acc = CrossProductMatrix(acc_true);

  // Covariance propagation
  // Eq. (63) from [A]
  // Step 1: jacobian-based propagation
  Eigen::Matrix<double, 15, 15> A = Eigen::Matrix<double, 15, 15>::Identity();
  // rotation: Eq. (59) from [A]
  A.block<3, 3>(0, 0) = dq.inverse().toRotationMatrix();

  // translation: Eq. (61) from [A]
  A.block<3, 3>(3, 0) = -0.5 * Rs * skew_acc * dt * dt;
  A.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * dt;

  // velocity: Eq. (60) from [A]
  A.block<3, 3>(6, 0) = -Rs * skew_acc * dt;

  // fill in the bias-related jacobians
  // inversely update t, v, R due to the dependencies.
  // Covariance state: [rotation(3), position(3), velocity(3),
  //                    gyro_bias(3), acc_bias(3)]
  // NOTE: rotation must be updated last — translation and velocity
  // read the old dR_dbg.

  // translation: Eq. (69 1/2) from [A]
  A.block<3, 3>(3, 9) =
      (data_.dv_dbg * dt - 0.5 * Rs * skew_acc * data_.dR_dbg * dt * dt);
  A.block<3, 3>(3, 12) = data_.dv_dba * dt - 0.5 * Rs * dt * dt;
  data_.dp_dbg += A.block<3, 3>(3, 9);
  data_.dp_dba += A.block<3, 3>(3, 12);

  // velocity: Eq. (69 1/2) from [A]
  A.block<3, 3>(6, 9) = -Rs * skew_acc * data_.dR_dbg * dt;
  A.block<3, 3>(6, 12) = -Rs * dt;
  data_.dv_dbg += A.block<3, 3>(6, 9);
  data_.dv_dba += A.block<3, 3>(6, 12);

  // rotation: combining Eq. (69 1/2) and the tricks of Eq. (59) from [A]
  Eigen::Matrix3d dR_dbg_updated =
      dq.inverse().toRotationMatrix() * data_.dR_dbg - Jr * dt;
  A.block<3, 3>(0, 9) = dR_dbg_updated - data_.dR_dbg;
  data_.dR_dbg = dR_dbg_updated;

  // propagate
  data_.covariance = A * data_.covariance * A.transpose();

  // Step 2: add noise
  double vars_v = pow(acc_noise_density, 2) * dt;
  double vars_omega = pow(gyro_noise_density, 2) * dt;
  double vars_p = 0.5 * vars_v * dt * dt;
  if (options_.use_integration_noise) {
    vars_p += pow(options_.integration_noise_density, 2) * dt;
  }
  double vars_ba = pow(calib_.acc_bias_random_walk_sigma, 2) * dt;
  double vars_bg = pow(calib_.gyro_bias_random_walk_sigma, 2) * dt;
  data_.covariance.block<3, 3>(0, 0) +=
      Eigen::Matrix3d::Identity() * vars_omega;  // omit Jr
  data_.covariance.block<3, 3>(3, 3) += Eigen::Matrix3d::Identity() * vars_p;
  data_.covariance.block<3, 3>(6, 6) += Eigen::Matrix3d::Identity() * vars_v;
  data_.covariance.block<3, 3>(9, 9) += Eigen::Matrix3d::Identity() * vars_bg;
  data_.covariance.block<3, 3>(12, 12) += Eigen::Matrix3d::Identity() * vars_ba;
}

void ImuPreintegrator::IntegrateOneMeasurement(const ImuMeasurement& prev,
                                               const ImuMeasurement& curr) {
  Eigen::Vector3d acc_s = prev.accel;
  Eigen::Vector3d gyro_s = prev.gyro;
  Eigen::Vector3d acc_e = curr.accel;
  Eigen::Vector3d gyro_e = curr.gyro;

  // Get dt and update boundaries.
  const timestamp_t interval_t_start = std::max(prev.timestamp, t_start_);
  const timestamp_t interval_t_end = std::min(curr.timestamp, t_end_);
  const double dt = TimestampDiffSeconds(interval_t_end, interval_t_start);
  THROW_CHECK_GT(dt, 0.0);
  const double imu_dt = TimestampDiffSeconds(curr.timestamp, prev.timestamp);

  // Interpolate at boundaries if needed.
  Eigen::Vector3d acc_s_tmp = acc_s;
  Eigen::Vector3d gyro_s_tmp = gyro_s;
  Eigen::Vector3d acc_e_tmp = acc_e;
  Eigen::Vector3d gyro_e_tmp = gyro_e;
  if (interval_t_start > prev.timestamp) {
    const double ratio_s =
        TimestampDiffSeconds(interval_t_start, prev.timestamp) / imu_dt;
    acc_s_tmp = (1.0 - ratio_s) * acc_s + ratio_s * acc_e;
    gyro_s_tmp = (1.0 - ratio_s) * gyro_s + ratio_s * gyro_e;
  }
  if (interval_t_end < curr.timestamp) {
    const double ratio_e =
        TimestampDiffSeconds(interval_t_end, prev.timestamp) / imu_dt;
    acc_e_tmp = (1.0 - ratio_e) * acc_s + ratio_e * acc_e;
    gyro_e_tmp = (1.0 - ratio_e) * gyro_s + ratio_e * gyro_e;
  }
  acc_s = acc_s_tmp;
  gyro_s = gyro_s_tmp;
  acc_e = acc_e_tmp;
  gyro_e = gyro_e_tmp;

  Eigen::Vector3d acc_true = 0.5 * (acc_s + acc_e) - biases_.tail<3>();
  acc_true = acc_rect_mat_inv_ * acc_true;
  Eigen::Vector3d gyro_true = 0.5 * (gyro_s + gyro_e) - biases_.head<3>();
  gyro_true = gyro_rect_mat_inv_ * gyro_true;

  // Check saturation.
  double acc_noise_density = calib_.acc_noise_density;
  if (acc_s.cwiseAbs().maxCoeff() > calib_.acc_saturation_max ||
      acc_e.cwiseAbs().maxCoeff() > calib_.acc_saturation_max) {
    acc_noise_density *= 100.0;
  }
  double gyro_noise_density = calib_.gyro_noise_density;
  if (gyro_s.cwiseAbs().maxCoeff() > calib_.gyro_saturation_max ||
      gyro_e.cwiseAbs().maxCoeff() > calib_.gyro_saturation_max) {
    gyro_noise_density *= 100.0;
  }

  Integrate(acc_true, gyro_true, dt, acc_noise_density, gyro_noise_density);
}

void ImuPreintegrator::FeedImu(const ImuMeasurement& m) {
  // Check if this is the first measurement
  if (!HasStarted()) {
    THROW_CHECK_LE(m.timestamp, t_start_)
        << "The timestamp of the first IMU measurement should not be later "
           "than the start of integration";
    measurements_.push_back(m);
    has_started_ = true;
    return;
  }

  // Assertion check: the new measurement needs to be later than measurement.
  ImuMeasurement last_measurement = measurements_.back();
  THROW_CHECK_GT(m.timestamp, last_measurement.timestamp);
  if (m.timestamp <= t_start_) {
    LOG(WARNING) << "The timestamp of this measurement is earlier than "
                    "t_start. Ignore the previous measurements.";
    measurements_.clear();
    measurements_.push_back(m);
    return;
  }
  if (last_measurement.timestamp >= t_end_) {
    LOG(WARNING) << "The timestamp of the last measurement has already reached "
                    "t_end. Ignore the current measurement.";
    return;
  }

  // Append measurements
  measurements_.push_back(m);
  IntegrateOneMeasurement(last_measurement, m);
}

void ImuPreintegrator::FeedImu(const std::vector<ImuMeasurement>& ms) {
  for (const auto& m : ms) {
    FeedImu(m);
  }
}

PreintegratedImuData ImuPreintegrator::Extract() {
  data_.Finalize();
  return data_;
}

void ImuPreintegrator::Update(PreintegratedImuData* data) { *data = data_; }

bool ImuPreintegrator::ShouldReintegrate(const Eigen::Vector6d& biases) const {
  THROW_CHECK_EQ(HasStarted(), true);
  Eigen::Vector6d diff_biases = biases - biases_;

  // check gyro
  double angle_norm = diff_biases.head<3>().norm() * data_.delta_t;
  if (angle_norm > options_.reintegrate_angle_norm_thres) return true;

  // check acc
  double v_norm = diff_biases.tail<3>().norm() * data_.delta_t;
  if (v_norm > options_.reintegrate_vel_norm_thres) return true;

  return false;
}

void ImuPreintegrator::Reintegrate() {
  Reset();
  has_started_ = true;
  for (size_t i = 1; i < measurements_.size(); ++i) {
    IntegrateOneMeasurement(measurements_[i - 1], measurements_[i]);
  }
  data_.Finalize();
}

void ImuPreintegrator::Reintegrate(const Eigen::Vector6d& biases) {
  SetBiases(biases);
  Reintegrate();
}

void ImuReintegrationCallback::AddEdge(ImuPreintegrator* integrator,
                                       PreintegratedImuData* data,
                                       const double* imu_state) {
  edges_.push_back({integrator, data, imu_state});
}

ceres::CallbackReturnType ImuReintegrationCallback::operator()(
    const ceres::IterationSummary& /*summary*/) {
  for (auto& edge : edges_) {
    // Read current biases from the optimized IMU state: [v(3), bg(3), ba(3)].
    Eigen::Vector6d biases(edge.imu_state + 3);
    if (edge.integrator->ShouldReintegrate(biases)) {
      edge.integrator->Reintegrate(biases);
      edge.integrator->Update(edge.data);
    }
  }
  return ceres::SOLVER_CONTINUE;
}

}  // namespace colmap

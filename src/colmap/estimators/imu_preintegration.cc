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

#include "colmap/estimators/imu_preintegration.h"

#include "colmap/estimators/cost_functions.h"
#include "colmap/geometry/pose.h"
#include "colmap/util/logging.h"

#include <cmath>

#include <Eigen/Dense>

namespace colmap {

PreintegratedImuMeasurement::PreintegratedImuMeasurement(
    const ImuPreintegrationOptions& options,
    const ImuCalibration& calib,
    double t_start,
    double t_end) {
  options_ = options;
  calib_ = calib;
  THROW_CHECK_LT(t_start, t_end);
  t_start_ = t_start;
  t_end_ = t_end;
  Reset();
}

void PreintegratedImuMeasurement::Reset() {
  delta_R_ij_ = Eigen::Quaterniond::Identity();
  delta_p_ij_ = Eigen::Vector3d::Zero();
  delta_v_ij_ = Eigen::Vector3d::Zero();
  delta_t_ = 0;
  jacobian_biases_ = Eigen::Matrix<double, 9, 6>::Zero();
  covs_ = Eigen::Matrix<double, 15, 15>::Zero();
  sqrt_information_ = Eigen::Matrix<double, 15, 15>::Zero();

  has_started_ = false;
  has_finished_ = false;
}

bool PreintegratedImuMeasurement::HasStarted() const { return has_started_; }

bool PreintegratedImuMeasurement::HasFinished() const { return has_finished_; }

void PreintegratedImuMeasurement::SetAccRectMat(const Eigen::Matrix3d& mat) {
  acc_rect_mat_inv_ = mat.inverse();
}

void PreintegratedImuMeasurement::SetGyroRectMat(const Eigen::Matrix3d& mat) {
  gyro_rect_mat_inv_ = mat.inverse();
}

void PreintegratedImuMeasurement::SetBiases(const Eigen::Vector6d& biases) {
  biases_ = biases;
}

void PreintegratedImuMeasurement::integrate(const Eigen::Vector3d& acc_true,
                                            const Eigen::Vector3d& gyro_true,
                                            double dt,
                                            double acc_noise_density,
                                            double gyro_noise_density) {
  has_finished_ = false;

  // [Reference]
  // [A] Forster et al. "On-Manifold Preintegration for Real-Time
  // Visual-Inertial Odometry", TRO 16. Integration step translation: Eq. (37)
  // from [A]
  delta_p_ij_ += delta_v_ij_ * dt + delta_R_ij_ * acc_true * 0.5 * dt * dt;
  // velocity: Eq. (36) from [A]
  delta_v_ij_ += delta_R_ij_ * acc_true * dt;
  // rotation: Eq. (35) from [A].
  Eigen::Quaterniond dq = QuaternionFromAngleAxis(gyro_true * dt);
  Eigen::Matrix3d Rs = delta_R_ij_.toRotationMatrix();
  delta_R_ij_ = delta_R_ij_ * dq;
  // time
  delta_t_ += dt;

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
  // translation: Eq. (69 1/2) from [A]
  A.block<3, 3>(3, 9) =
      jacobian_biases_.block<3, 3>(6, 0) * dt - 0.5 * Rs * dt * dt;
  A.block<3, 3>(3, 12) =
      (jacobian_biases_.block<3, 3>(6, 3) * dt -
       0.5 * Rs * skew_acc * jacobian_biases_.block<3, 3>(0, 3) * dt * dt);
  jacobian_biases_.block<3, 3>(3, 0) += A.block<3, 3>(3, 9);
  jacobian_biases_.block<3, 3>(3, 3) += A.block<3, 3>(3, 12);

  // velocity: Eq. (69 1/2) from [A]
  A.block<3, 3>(6, 9) = -Rs * dt;
  A.block<3, 3>(6, 12) =
      -Rs * skew_acc * jacobian_biases_.block<3, 3>(0, 3) * dt;
  jacobian_biases_.block<3, 3>(6, 0) += A.block<3, 3>(6, 9);
  jacobian_biases_.block<3, 3>(6, 3) += A.block<3, 3>(6, 12);

  // rotation: combining Eq. (69 1/2) and the tricks of Eq. (59) from [A]
  Eigen::Matrix3d dR_dbg_updated =
      dq.inverse().toRotationMatrix() * jacobian_biases_.block<3, 3>(0, 3) -
      Jr * dt;
  A.block<3, 3>(0, 12) = dR_dbg_updated - jacobian_biases_.block<3, 3>(0, 3);
  jacobian_biases_.block<3, 3>(0, 3) = dR_dbg_updated;

  // propagate
  covs_ = A * covs_ * A.transpose();

  // Step 2: add noise
  double vars_v = pow(acc_noise_density, 2) * dt;
  double vars_omega = pow(gyro_noise_density, 2) * dt;
  double vars_p = 0.5 * vars_v * dt * dt;
  if (options_.use_integration_noise) {
    vars_p += pow(options_.integration_noise_density, 2) * dt;
  }
  double vars_ba = pow(calib_.acc_bias_random_walk_sigma, 2) * dt;
  double vars_bg = pow(calib_.gyro_bias_random_walk_sigma, 2) * dt;
  covs_.block<3, 3>(0, 0) +=
      Eigen::Matrix3d::Identity() * vars_omega;  // omit Jr
  covs_.block<3, 3>(3, 3) += Eigen::Matrix3d::Identity() * vars_p;
  covs_.block<3, 3>(6, 6) += Eigen::Matrix3d::Identity() * vars_v;
  covs_.block<3, 3>(9, 9) += Eigen::Matrix3d::Identity() * vars_ba;
  covs_.block<3, 3>(12, 12) += Eigen::Matrix3d::Identity() * vars_bg;
}

void PreintegratedImuMeasurement::AddMeasurement(const ImuMeasurement& m) {
  // Check if this is the first measurement
  if (!HasStarted()) {
    THROW_CHECK_LE(m.timestamp, t_start_)
        << "The timestamp of the first IMU measurement should not be later "
           "than the start of integration";
    measurements_.insert(m);
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
    measurements_.insert(m);
    return;
  }
  if (last_measurement.timestamp >= t_end_) {
    LOG(WARNING) << "The timestamp of the last measurement has already reached "
                    "t_end. Ignore the current measurement.";
    return;
  }

  // Append measurements
  measurements_.insert(m);

  // Get measurements at the boundaries
  Eigen::Vector3d acc_s = last_measurement.linear_acceleration;
  Eigen::Vector3d gyro_s = last_measurement.angular_velocity;
  Eigen::Vector3d acc_e = m.linear_acceleration;
  Eigen::Vector3d gyro_e = m.angular_velocity;

  // Get dt and update boundaries
  double interval_t_start = std::max(last_measurement.timestamp, t_start_);
  double interval_t_end = std::min(m.timestamp, t_end_);
  double dt = interval_t_end - interval_t_start;
  THROW_CHECK_GT(dt, 0.0);
  const double imu_dt = m.timestamp - last_measurement.timestamp;
  Eigen::Vector3d acc_s_tmp = acc_s;
  Eigen::Vector3d gyro_s_tmp = gyro_s;
  Eigen::Vector3d acc_e_tmp = acc_e;
  Eigen::Vector3d gyro_e_tmp = gyro_e;
  if (interval_t_start > last_measurement.timestamp) {
    const double ratio_s =
        (interval_t_start - last_measurement.timestamp) / imu_dt;
    acc_s_tmp = (1.0 - ratio_s) * acc_s + ratio_s * acc_e;
    gyro_s_tmp = (1.0 - ratio_s) * gyro_s + ratio_s * gyro_e;
  }
  if (interval_t_end < m.timestamp) {
    const double ratio_e =
        (interval_t_end - last_measurement.timestamp) / imu_dt;
    acc_e_tmp = (1.0 - ratio_e) * acc_s + ratio_e * acc_e;
    gyro_e_tmp = (1.0 - ratio_e) * gyro_s + ratio_e * gyro_e;
  }
  acc_s = acc_s_tmp;
  gyro_s = gyro_s_tmp;
  acc_e = acc_e_tmp;
  gyro_e = gyro_e_tmp;
  Eigen::Vector3d acc_true = 0.5 * (acc_s + acc_e) - biases_.head<3>();
  acc_true = acc_rect_mat_inv_ * acc_true;
  Eigen::Vector3d gyro_true = 0.5 * (gyro_s + gyro_e) - biases_.tail<3>();
  gyro_true = gyro_rect_mat_inv_ * gyro_true;

  // Check saturation
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

  // Integration
  integrate(acc_true, gyro_true, dt, acc_noise_density, gyro_noise_density);
}

void PreintegratedImuMeasurement::AddMeasurements(const ImuMeasurements& ms) {
  for (auto it = ms.begin(); it != ms.end(); ++it) {
    AddMeasurement(*it);
  }
}

void PreintegratedImuMeasurement::Finish() {
  // Enforce symmetry and stability
  covs_ = (covs_ + covs_.transpose()) / 2.0;
  covs_ += Eigen::Matrix<double, 15, 15>::Identity() * 1e-18;

  // Factorize
  sqrt_information_ = covs_.inverse().llt().matrixL().transpose();

  // Set flag
  has_finished_ = true;
}

bool PreintegratedImuMeasurement::CheckReintegrate(
    const Eigen::Vector6d& biases) const {
  THROW_CHECK_EQ(HasStarted(), true);
  Eigen::Vector6d diff_biases = biases - biases_;

  // check acc
  double v_norm = diff_biases.head<3>().norm() * delta_t_;
  if (v_norm > options_.reintegrate_vel_norm_thres) return true;

  // check gyro
  double angle_norm = diff_biases.tail<3>().norm() * delta_t_;
  if (angle_norm > options_.reintegrate_angle_norm_thres) return true;

  // else return false
  return false;
}

void PreintegratedImuMeasurement::Reintegrate() {
  Reset();
  ImuMeasurement last_measurement = measurements_[0];
  has_started_ = true;
  for (size_t i = 1; i < measurements_.size(); ++i) {
    // Current measurement
    auto m = measurements_[i];

    // Get measurements at the boundaries
    Eigen::Vector3d acc_s = last_measurement.linear_acceleration;
    Eigen::Vector3d gyro_s = last_measurement.angular_velocity;
    Eigen::Vector3d acc_e = m.linear_acceleration;
    Eigen::Vector3d gyro_e = m.angular_velocity;

    // Get dt and update boundaries
    double interval_t_start = std::max(last_measurement.timestamp, t_start_);
    double interval_t_end = std::min(m.timestamp, t_end_);
    double dt = interval_t_end - interval_t_start;
    THROW_CHECK_GT(dt, 0.0);
    const double imu_dt = m.timestamp - last_measurement.timestamp;
    Eigen::Vector3d acc_s_tmp = acc_s;
    Eigen::Vector3d gyro_s_tmp = gyro_s;
    Eigen::Vector3d acc_e_tmp = acc_e;
    Eigen::Vector3d gyro_e_tmp = gyro_e;
    if (interval_t_start > last_measurement.timestamp) {
      const double ratio_s =
          (interval_t_start - last_measurement.timestamp) / imu_dt;
      acc_s_tmp = (1.0 - ratio_s) * acc_s + ratio_s * acc_e;
      gyro_s_tmp = (1.0 - ratio_s) * gyro_s + ratio_s * gyro_e;
    }
    if (interval_t_end < m.timestamp) {
      const double ratio_e =
          (interval_t_end - last_measurement.timestamp) / imu_dt;
      acc_e_tmp = (1.0 - ratio_e) * acc_s + ratio_e * acc_e;
      gyro_e_tmp = (1.0 - ratio_e) * gyro_s + ratio_e * gyro_e;
    }
    acc_s = acc_s_tmp;
    gyro_s = gyro_s_tmp;
    acc_e = acc_e_tmp;
    gyro_e = gyro_e_tmp;
    Eigen::Vector3d acc_true = 0.5 * (acc_s + acc_e) - biases_.head<3>();
    Eigen::Vector3d gyro_true = 0.5 * (gyro_s + gyro_e) - biases_.tail<3>();

    // Check saturation
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

    // Integration
    integrate(acc_true, gyro_true, dt, acc_noise_density, gyro_noise_density);
    last_measurement = m;
  }
  Finish();
}

void PreintegratedImuMeasurement::Reintegrate(const Eigen::Vector6d& biases) {
  SetBiases(biases);
  Reintegrate();
}

// data interfaces
const double PreintegratedImuMeasurement::DeltaT() const { return delta_t_; }

const Eigen::Quaterniond& PreintegratedImuMeasurement::DeltaR() const {
  return delta_R_ij_;
}

const Eigen::Vector3d& PreintegratedImuMeasurement::DeltaP() const {
  return delta_p_ij_;
}

const Eigen::Vector3d& PreintegratedImuMeasurement::DeltaV() const {
  return delta_v_ij_;
}

const Eigen::Matrix3d PreintegratedImuMeasurement::dR_dbg() const {
  return jacobian_biases_.block<3, 3>(0, 3);
}

const Eigen::Matrix3d PreintegratedImuMeasurement::dp_dba() const {
  return jacobian_biases_.block<3, 3>(3, 0);
}

const Eigen::Matrix3d PreintegratedImuMeasurement::dp_dbg() const {
  return jacobian_biases_.block<3, 3>(3, 3);
}

const Eigen::Matrix3d PreintegratedImuMeasurement::dv_dba() const {
  return jacobian_biases_.block<3, 3>(6, 0);
}

const Eigen::Matrix3d PreintegratedImuMeasurement::dv_dbg() const {
  return jacobian_biases_.block<3, 3>(6, 3);
}

const Eigen::Vector6d& PreintegratedImuMeasurement::Biases() const {
  return biases_;
}

const Eigen::Matrix<double, 15, 15> PreintegratedImuMeasurement::Covariance()
    const {
  return covs_;
}

const Eigen::Matrix<double, 15, 15>
PreintegratedImuMeasurement::SqrtInformation() const {
  return sqrt_information_;
}

const double PreintegratedImuMeasurement::GravityMagnitude() const {
  return calib_.gravity_magnitude;
}

const ImuMeasurements PreintegratedImuMeasurement::Measurements() const {
  return measurements_;
}

}  // namespace colmap

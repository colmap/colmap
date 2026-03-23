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

#include "colmap/util/eigen_matchers.h"
#include "colmap/util/timestamp.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Helper: create a default integrator with a time window [0, T] seconds.
ImuPreintegrator MakeIntegrator(double T_seconds) {
  ImuPreintegrationOptions options;
  ImuCalibration calib;
  timestamp_t t_start = SecondsToTimestamp(0.0);
  timestamp_t t_end = SecondsToTimestamp(T_seconds);
  return ImuPreintegrator(options, calib, t_start, t_end);
}

// Helper: feed N uniform measurements with constant accel and gyro.
void FeedConstant(ImuPreintegrator& integrator,
                  const Eigen::Vector3d& accel,
                  const Eigen::Vector3d& gyro,
                  int N,
                  double dt) {
  for (int i = 0; i <= N; ++i) {
    integrator.FeedImu(ImuMeasurement(SecondsToTimestamp(i * dt), accel, gyro));
  }
}

TEST(ImuPreintegrator, ZeroRotation) {
  const int N = 10;
  const double dt = 0.01;
  auto integrator = MakeIntegrator(N * dt);

  Eigen::Vector3d accel(0, 0, 9.81);
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  FeedConstant(integrator, accel, gyro, N, dt);

  PreintegratedImuData data = integrator.Extract();
  const double T = N * dt;
  EXPECT_NEAR(data.delta_t, T, 1e-12);

  // Identity rotation.
  EXPECT_NEAR(
      data.delta_R.angularDistance(Eigen::Quaterniond::Identity()), 0.0, 1e-12);

  // delta_v = accel * T (no rotation, no bias).
  EXPECT_NEAR(data.delta_v(2), 9.81 * T, 1e-10);

  // delta_p = 0.5 * accel * T^2.
  EXPECT_NEAR(data.delta_p(2), 0.5 * 9.81 * T * T, 1e-10);
}

TEST(ImuPreintegrator, ConstantAcceleration) {
  const int N = 20;
  const double dt = 0.01;
  auto integrator = MakeIntegrator(N * dt);

  Eigen::Vector3d accel(2.0, -1.0, 0.5);
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  FeedConstant(integrator, accel, gyro, N, dt);

  PreintegratedImuData data = integrator.Extract();
  const double T = N * dt;

  // delta_v = accel * T.
  Eigen::Vector3d expected_v = accel * T;
  EXPECT_THAT(data.delta_v, EigenMatrixNear(expected_v, 1e-10));

  // delta_p = 0.5 * accel * T^2.
  Eigen::Vector3d expected_p = 0.5 * accel * T * T;
  EXPECT_THAT(data.delta_p, EigenMatrixNear(expected_p, 1e-10));
}

TEST(ImuPreintegrator, ConstantRotation) {
  const int N = 10;
  const double dt = 0.01;
  auto integrator = MakeIntegrator(N * dt);

  Eigen::Vector3d accel = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro(0.1, 0.0, 0.0);  // rotate around x-axis
  FeedConstant(integrator, accel, gyro, N, dt);

  PreintegratedImuData data = integrator.Extract();
  const double T = N * dt;

  // Expected rotation: Exp(gyro * T).
  Eigen::AngleAxisd expected_aa(gyro.norm() * T, gyro.normalized());
  EXPECT_NEAR(
      data.delta_R.angularDistance(Eigen::Quaterniond(expected_aa)), 0.0, 1e-6);
}

TEST(ImuPreintegrator, Reset) {
  const int N = 5;
  const double dt = 0.01;
  auto integrator = MakeIntegrator(N * dt);

  Eigen::Vector3d accel(1.0, 2.0, 9.81);
  Eigen::Vector3d gyro(0.1, -0.2, 0.05);
  FeedConstant(integrator, accel, gyro, N, dt);

  integrator.Reset();
  PreintegratedImuData data = integrator.Extract();
  EXPECT_NEAR(data.delta_t, 0.0, 1e-15);
  EXPECT_NEAR(data.delta_p.norm(), 0.0, 1e-15);
  EXPECT_NEAR(data.delta_v.norm(), 0.0, 1e-15);
  EXPECT_NEAR(
      data.delta_R.angularDistance(Eigen::Quaterniond::Identity()), 0.0, 1e-15);
}

TEST(ImuPreintegrator, ReintegrateMatchesFresh) {
  const int N = 10;
  const double dt = 0.01;

  // Fresh integration.
  auto integrator_fresh = MakeIntegrator(N * dt);
  Eigen::Vector3d accel(1.0, 0.0, 9.81);
  Eigen::Vector3d gyro(0.0, 0.1, 0.0);
  FeedConstant(integrator_fresh, accel, gyro, N, dt);
  PreintegratedImuData data_fresh = integrator_fresh.Extract();

  // Integrate, then reintegrate with same biases.
  auto integrator = MakeIntegrator(N * dt);
  FeedConstant(integrator, accel, gyro, N, dt);
  integrator.Reintegrate();
  PreintegratedImuData data_reint = integrator.Extract();

  EXPECT_NEAR(data_fresh.delta_t, data_reint.delta_t, 1e-15);
  EXPECT_THAT(data_fresh.delta_p, EigenMatrixNear(data_reint.delta_p, 1e-12));
  EXPECT_THAT(data_fresh.delta_v, EigenMatrixNear(data_reint.delta_v, 1e-12));
  EXPECT_NEAR(
      data_fresh.delta_R.angularDistance(data_reint.delta_R), 0.0, 1e-12);
}

TEST(ImuPreintegrator, ExtractAndUpdateConsistent) {
  const int N = 10;
  const double dt = 0.01;
  auto integrator = MakeIntegrator(N * dt);

  Eigen::Vector3d accel(1.0, -0.5, 9.81);
  Eigen::Vector3d gyro(0.05, 0.1, -0.02);
  FeedConstant(integrator, accel, gyro, N, dt);

  PreintegratedImuData data_extract = integrator.Extract();

  // Reintegrate (same biases) then Update.
  integrator.Reintegrate();
  PreintegratedImuData data_update;
  integrator.Update(&data_update);

  EXPECT_NEAR(data_extract.delta_t, data_update.delta_t, 1e-15);
  EXPECT_THAT(data_extract.delta_p,
              EigenMatrixNear(data_update.delta_p, 1e-12));
  EXPECT_THAT(data_extract.delta_v,
              EigenMatrixNear(data_update.delta_v, 1e-12));
  EXPECT_THAT(data_extract.sqrt_information,
              EigenMatrixNear(data_update.sqrt_information, 1e-10));
}

TEST(ImuPreintegrator, ShouldReintegrate) {
  const int N = 10;
  const double dt = 0.01;
  auto integrator = MakeIntegrator(N * dt);

  Eigen::Vector3d accel(0, 0, 9.81);
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  FeedConstant(integrator, accel, gyro, N, dt);

  // Same biases — should not reintegrate.
  Eigen::Vector6d same_biases = Eigen::Vector6d::Zero();
  EXPECT_FALSE(integrator.ShouldReintegrate(same_biases));

  // Large bias change — should reintegrate.
  Eigen::Vector6d changed_biases = Eigen::Vector6d::Zero();
  changed_biases(0) = 1.0;  // large gyro bias change
  EXPECT_TRUE(integrator.ShouldReintegrate(changed_biases));
}

TEST(ImuPreintegrator, CovariancePositiveDefinite) {
  const int N = 20;
  const double dt = 0.01;
  auto integrator = MakeIntegrator(N * dt);

  Eigen::Vector3d accel(0.5, -0.3, 9.81);
  Eigen::Vector3d gyro(0.1, -0.05, 0.02);
  FeedConstant(integrator, accel, gyro, N, dt);

  PreintegratedImuData data = integrator.Extract();

  // Covariance should be symmetric positive definite after Finalize.
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15>> solver(
      data.covariance);
  EXPECT_TRUE(solver.info() == Eigen::Success);
  EXPECT_GT(solver.eigenvalues().minCoeff(), 0.0);
}

}  // namespace
}  // namespace colmap

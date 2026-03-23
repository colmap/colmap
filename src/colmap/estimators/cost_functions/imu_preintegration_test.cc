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

#include "colmap/estimators/cost_functions/imu_preintegration.h"

#include "colmap/estimators/imu_preintegration.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/util/eigen_matchers.h"
#include "colmap/util/timestamp.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Gravity vector (gravity-aligned world: -Z).
const Eigen::Vector3d kGravity(0, 0, -9.81);

// Build a preintegrated data from constant IMU readings over N steps.
// Returns the data and also outputs the expected world-frame trajectory
// quantities needed to construct zero-residual parameter blocks.
struct TrajectoryGT {
  Rigid3d body_from_world_i;
  Rigid3d body_from_world_j;
  Eigen::Vector3d v_i;
  Eigen::Vector3d v_j;
};

PreintegratedImuData MakeConstantData(const Eigen::Vector3d& accel,
                                      const Eigen::Vector3d& gyro,
                                      int N,
                                      double dt,
                                      TrajectoryGT* gt) {
  ImuPreintegrationOptions options;
  ImuCalibration calib;
  calib.gravity_magnitude = kGravity.norm();
  timestamp_t t_start = SecondsToTimestamp(0.0);
  timestamp_t t_end = SecondsToTimestamp(N * dt);
  ImuPreintegrator integrator(options, calib, t_start, t_end);
  for (int i = 0; i <= N; ++i) {
    integrator.FeedImu(ImuMeasurement(SecondsToTimestamp(i * dt), accel, gyro));
  }
  PreintegratedImuData data = integrator.Extract();

  // Construct ground-truth trajectory consistent with the preintegration.
  // Use a non-trivial initial rotation to exercise rotation residuals.
  const double T = data.delta_t;
  Eigen::Quaterniond q_i =
      Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()));
  Eigen::Vector3d p_i(0.5, -0.2, 1.0);
  Eigen::Matrix3d R_WB_i = q_i.inverse().toRotationMatrix();

  // World-frame velocity at frame i (chosen so residual is zero).
  // From the cost function residual equations:
  //   delta_R = body_from_world_j * world_from_body_i
  //   delta_p = body_from_world_i * (p_j - p_i - v_i*dt - 0.5*g*dt^2)
  //   delta_v = body_from_world_i * (v_j - v_i - g*dt)
  // Solve for j quantities:
  //   body_from_world_j = delta_R * body_from_world_i
  //   v_j = v_i + g*dt + world_from_body_i * delta_v
  //   p_j = p_i + v_i*dt + 0.5*g*dt^2 + world_from_body_i * delta_p
  Eigen::Vector3d v_i(1.0, 0.5, -0.2);
  Eigen::Vector3d v_j = v_i + kGravity * T + R_WB_i * data.delta_v;
  Eigen::Vector3d p_j =
      p_i + v_i * T + 0.5 * kGravity * T * T + R_WB_i * data.delta_p;
  Eigen::Quaterniond q_j = data.delta_R * q_i;

  // Convert world positions to body_from_world (Rigid3d convention):
  //   body_from_world = (q_BW, -q_BW * p_W)
  gt->body_from_world_i = Rigid3d(q_i, -q_i.toRotationMatrix() * p_i);
  gt->body_from_world_j = Rigid3d(q_j, -q_j.toRotationMatrix() * p_j);
  gt->v_i = v_i;
  gt->v_j = v_j;

  return data;
}

// Pack Rigid3d into a 7-element array [qx, qy, qz, qw, tx, ty, tz].
void PackRigid3d(const Rigid3d& rigid, double* out) {
  out[0] = rigid.rotation().x();
  out[1] = rigid.rotation().y();
  out[2] = rigid.rotation().z();
  out[3] = rigid.rotation().w();
  out[4] = rigid.translation().x();
  out[5] = rigid.translation().y();
  out[6] = rigid.translation().z();
}

// Pack IMU state: [vx, vy, vz, bgx, bgy, bgz, bax, bay, baz].
void PackImuState(const Eigen::Vector3d& velocity,
                  const Eigen::Vector3d& gyro_bias,
                  const Eigen::Vector3d& acc_bias,
                  double* data) {
  data[0] = velocity.x();
  data[1] = velocity.y();
  data[2] = velocity.z();
  data[3] = gyro_bias.x();
  data[4] = gyro_bias.y();
  data[5] = gyro_bias.z();
  data[6] = acc_bias.x();
  data[7] = acc_bias.y();
  data[8] = acc_bias.z();
}

TEST(ImuPreintegrationCostFunctor, ZeroResidualAtGroundTruth) {
  const int N = 10;
  const double dt = 0.01;
  Eigen::Vector3d accel(0, 0, 9.81);
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();

  TrajectoryGT gt;
  PreintegratedImuData data = MakeConstantData(accel, gyro, N, dt, &gt);
  data.sqrt_information = Eigen::Matrix<double, 15, 15>::Identity();

  std::unique_ptr<ceres::CostFunction> cost_function(
      ImuPreintegrationCostFunctor::Create(&data, kGravity));

  double body_from_world_i[7], body_from_world_j[7];
  double imu_state_i[9], imu_state_j[9];
  PackRigid3d(gt.body_from_world_i, body_from_world_i);
  PackRigid3d(gt.body_from_world_j, body_from_world_j);
  PackImuState(
      gt.v_i, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), imu_state_i);
  PackImuState(
      gt.v_j, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), imu_state_j);

  double residuals[15];
  const double* parameters[4] = {
      body_from_world_i, imu_state_i, body_from_world_j, imu_state_j};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));

  for (int i = 0; i < 15; ++i) {
    EXPECT_NEAR(residuals[i], 0.0, 1e-8) << "residual[" << i << "]";
  }
}

TEST(ImuPreintegrationCostFunctor, ZeroResidualWithMotion) {
  const int N = 20;
  const double dt = 0.005;
  Eigen::Vector3d accel(0.5, -0.3, 9.81);
  Eigen::Vector3d gyro(0.1, -0.05, 0.02);

  TrajectoryGT gt;
  PreintegratedImuData data = MakeConstantData(accel, gyro, N, dt, &gt);
  data.sqrt_information = Eigen::Matrix<double, 15, 15>::Identity();

  std::unique_ptr<ceres::CostFunction> cost_function(
      ImuPreintegrationCostFunctor::Create(&data, kGravity));

  double body_from_world_i[7], body_from_world_j[7];
  double imu_state_i[9], imu_state_j[9];
  PackRigid3d(gt.body_from_world_i, body_from_world_i);
  PackRigid3d(gt.body_from_world_j, body_from_world_j);
  PackImuState(
      gt.v_i, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), imu_state_i);
  PackImuState(
      gt.v_j, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), imu_state_j);

  double residuals[15];
  const double* parameters[4] = {
      body_from_world_i, imu_state_i, body_from_world_j, imu_state_j};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));

  for (int i = 0; i < 15; ++i) {
    EXPECT_NEAR(residuals[i], 0.0, 1e-6) << "residual[" << i << "]";
  }
}

TEST(VisualCentricImuPreintegrationCostFunctor,
     ZeroResidualIdentityExtrinsics) {
  const int N = 10;
  const double dt = 0.01;
  Eigen::Vector3d accel(0, 0, 9.81);
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();

  TrajectoryGT gt;
  PreintegratedImuData data = MakeConstantData(accel, gyro, N, dt, &gt);
  data.sqrt_information = Eigen::Matrix<double, 15, 15>::Identity();

  std::unique_ptr<ceres::CostFunction> cost_function(
      VisualCentricImuPreintegrationCostFunctor::Create(&data));

  double log_scale[1] = {0.0};
  double gravity_direction[3] = {0, 0, -1};
  double imu_from_cam[7] = {0, 0, 0, 1, 0, 0, 0};
  double i_from_world[7], j_from_world[7];
  double imu_state_i[9], imu_state_j[9];
  PackRigid3d(gt.body_from_world_i, i_from_world);
  PackRigid3d(gt.body_from_world_j, j_from_world);
  PackImuState(
      gt.v_i, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), imu_state_i);
  PackImuState(
      gt.v_j, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), imu_state_j);

  double residuals[15];
  const double* parameters[7] = {log_scale,
                                 gravity_direction,
                                 imu_from_cam,
                                 i_from_world,
                                 imu_state_i,
                                 j_from_world,
                                 imu_state_j};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));

  for (int i = 0; i < 15; ++i) {
    EXPECT_NEAR(residuals[i], 0.0, 1e-8) << "residual[" << i << "]";
  }
}

TEST(ImuPreintegrationCostConsistency, ImuAndVisualCentricMatch) {
  const int N = 15;
  const double dt = 0.005;
  Eigen::Vector3d accel(1.0, -0.5, 9.81);
  Eigen::Vector3d gyro(0.05, 0.1, -0.03);

  TrajectoryGT gt;
  PreintegratedImuData data = MakeConstantData(accel, gyro, N, dt, &gt);
  data.sqrt_information = Eigen::Matrix<double, 15, 15>::Identity();

  gt.v_i += Eigen::Vector3d(0.01, -0.02, 0.005);

  // Body-centric.
  std::unique_ptr<ceres::CostFunction> body_cost(
      ImuPreintegrationCostFunctor::Create(&data, kGravity));
  double body_i[7], body_j[7], state_i[9], state_j[9];
  PackRigid3d(gt.body_from_world_i, body_i);
  PackRigid3d(gt.body_from_world_j, body_j);
  PackImuState(
      gt.v_i, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), state_i);
  PackImuState(
      gt.v_j, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), state_j);
  double body_residuals[15];
  const double* body_params[4] = {body_i, state_i, body_j, state_j};
  EXPECT_TRUE(body_cost->Evaluate(body_params, body_residuals, nullptr));

  // Visual-centric with identity extrinsics and unit scale.
  std::unique_ptr<ceres::CostFunction> visual_cost(
      VisualCentricImuPreintegrationCostFunctor::Create(&data));
  double log_scale[1] = {0.0};
  double gravity_direction[3] = {0, 0, -1};
  double imu_from_cam[7] = {0, 0, 0, 1, 0, 0, 0};
  double visual_residuals[15];
  const double* visual_params[7] = {log_scale,
                                    gravity_direction,
                                    imu_from_cam,
                                    body_i,
                                    state_i,
                                    body_j,
                                    state_j};
  EXPECT_TRUE(visual_cost->Evaluate(visual_params, visual_residuals, nullptr));

  for (int i = 0; i < 15; ++i) {
    EXPECT_NEAR(body_residuals[i], visual_residuals[i], 1e-8)
        << "residual[" << i << "]";
  }
}

class PhysicsConsistencyTest
    : public ::testing::TestWithParam<ImuIntegrationMethod> {};

TEST_P(PhysicsConsistencyTest, CostFunctionMatchesPhysics) {
  // Validate against actual physics: compute the TRUE trajectory via
  // fine-grained Euler integration (independent of preintegrator), then check
  // that the cost function gives zero residual with the physical poses.
  // This catches convention errors (e.g., Exp(+w*dt) vs Exp(-w*dt)) that
  // self-consistent tests (constructing GT from preintegrated data) miss.

  const ImuIntegrationMethod method = GetParam();
  const Eigen::Vector3d gravity(0, 0, -9.81);
  const Eigen::Vector3d gyro(0.1, -0.05, 0.02);  // body angular velocity
  const Eigen::Vector3d accel(0.5, -0.3, 9.81);  // accelerometer reading

  // Initial state: non-trivial orientation.
  Eigen::Quaterniond q_WB_i(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()));
  Eigen::Matrix3d R_WB_i = q_WB_i.toRotationMatrix();
  Eigen::Vector3d p_W_i(0.5, -0.2, 1.0);
  Eigen::Vector3d v_W_i(1.0, 0.5, -0.2);

  // Integration parameters.
  const int N = 20;
  const double dt = 0.005;
  const double T = N * dt;

  //--------------------------------------------------------------------------
  // Step 1: Compute TRUE trajectory via fine-grained Euler integration.
  //--------------------------------------------------------------------------
  Eigen::Matrix3d R_WB = R_WB_i;
  Eigen::Vector3d v_W = v_W_i;
  Eigen::Vector3d p_W = p_W_i;

  // Use very fine steps (1000x finer than IMU) for ground truth.
  const double dt_fine = dt / 1000.0;
  const int N_fine = static_cast<int>(T / dt_fine + 0.5);
  for (int i = 0; i < N_fine; ++i) {
    // World-frame acceleration: R_WB * specific_force + gravity.
    Eigen::Vector3d a_W = R_WB * accel + gravity;
    p_W += v_W * dt_fine + 0.5 * a_W * dt_fine * dt_fine;
    v_W += a_W * dt_fine;
    // R_WB evolves as R_WB(t+dt) = R_WB(t) * Exp(w*dt).
    Eigen::AngleAxisd dR_fine(gyro.norm() * dt_fine, gyro.normalized());
    R_WB = R_WB * dR_fine.toRotationMatrix();
  }

  // Convert to body_from_world (COLMAP convention).
  Eigen::Quaterniond q_BW_i(R_WB_i.transpose());
  Eigen::Quaterniond q_BW_j(R_WB.transpose());
  Eigen::Vector3d t_BW_i = -R_WB_i.transpose() * p_W_i;
  Eigen::Vector3d t_BW_j = -R_WB.transpose() * p_W;

  //--------------------------------------------------------------------------
  // Step 2: Preintegrate IMU measurements.
  //--------------------------------------------------------------------------
  ImuPreintegrationOptions options;
  options.method = method;
  ImuCalibration calib;
  calib.gravity_magnitude = gravity.norm();
  ImuPreintegrator integrator(
      options, calib, SecondsToTimestamp(0.0), SecondsToTimestamp(T));
  for (int i = 0; i <= N; ++i) {
    integrator.FeedImu(ImuMeasurement(SecondsToTimestamp(i * dt), accel, gyro));
  }
  PreintegratedImuData data = integrator.Extract();
  data.sqrt_information = Eigen::Matrix<double, 15, 15>::Identity();

  //--------------------------------------------------------------------------
  // Step 3: Evaluate cost function with TRUE physical poses.
  //--------------------------------------------------------------------------
  std::unique_ptr<ceres::CostFunction> cost_function(
      ImuPreintegrationCostFunctor::Create(&data, gravity));

  double body_from_world_i[7], body_from_world_j[7];
  double imu_state_i[9], imu_state_j[9];
  PackRigid3d(Rigid3d(q_BW_i, t_BW_i), body_from_world_i);
  PackRigid3d(Rigid3d(q_BW_j, t_BW_j), body_from_world_j);
  PackImuState(
      v_W_i, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), imu_state_i);
  PackImuState(
      v_W, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), imu_state_j);

  double residuals[15];
  const double* parameters[4] = {
      body_from_world_i, imu_state_i, body_from_world_j, imu_state_j};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));

  // RK4 with closed-form integrals is very accurate; midpoint has O(dt^2)
  // discretization error per step.
  const double tol = (method == ImuIntegrationMethod::MIDPOINT) ? 1e-3 : 1e-4;
  for (int i = 0; i < 15; ++i) {
    EXPECT_NEAR(residuals[i], 0.0, tol) << "residual[" << i << "]"
                                        << (i < 3   ? " (rotation)"
                                            : i < 6 ? " (position)"
                                            : i < 9 ? " (velocity)"
                                                    : " (bias)");
  }
}

INSTANTIATE_TEST_SUITE_P(ImuPreintegrationCostFunctor,
                         PhysicsConsistencyTest,
                         ::testing::Values(ImuIntegrationMethod::MIDPOINT,
                                           ImuIntegrationMethod::RK4));

}  // namespace
}  // namespace colmap

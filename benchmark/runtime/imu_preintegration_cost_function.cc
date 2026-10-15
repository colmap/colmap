// SPDX-License-Identifier: BSD-3-Clause

// Micro-benchmarks for the IMU preintegration cost functions: AutoDiff vs.
// analytical-Jacobian Evaluate() for the two cost function families
// (body-centric, visual-centric).
//
// Evaluate() reads only the fixed-size preintegrated result (delta_R/p/v, bias
// Jacobians, 15x15 sqrt_info), so its cost is invariant to both the
// number of integrated measurements and the integration method (MIDPOINT/RK4)
// that produced the data -- those only affect the one-off preintegration step,
// not this residual evaluation. The data is therefore built once with fixed
// settings below.
//
// Use the AutoDiff-vs-Analytical ratio to weigh whether the analytical
// Jacobians earn their (substantial) code/maintenance cost. Keep in mind IMU
// residual blocks are O(#frames), vastly outnumbered by O(#observations)
// reprojection blocks, so a per-Evaluate win may not move end-to-end BA time.

#include "colmap/geometry/rigid3.h"
#include "colmap/inertial/preintegration.h"
#include "colmap/inertial/preintegration_cost.h"
#include "colmap/sensor/imu.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/timestamp.h"

#include <memory>

#include <Eigen/Geometry>
#include <benchmark/benchmark.h>
#include <ceres/ceres.h>

using namespace colmap;

namespace {

const Eigen::Vector3d kGravity(0, 0, -9.81);
const Eigen::Vector3d kGyro(0.1, -0.05, 0.02);
const Eigen::Vector3d kAccel(0.5, -0.3, 9.81);

enum class Impl { kAutoDiff, kAnalytical };

// Build a finalized PreintegratedImuData from constant IMU readings. Method and
// count are fixed since Evaluate() is invariant to them (see file header).
PreintegratedImuData MakeImuData() {
  constexpr int kNumMeasurements = 100;
  constexpr double dt = 0.005;
  ImuPreintegrationOptions options;
  options.method = ImuIntegrationMethod::RK4;
  ImuCalibration calib;
  calib.gravity_magnitude = kGravity.norm();
  ImuPreintegrator integrator(options,
                              calib,
                              TimestampFromSeconds(0.0),
                              TimestampFromSeconds(kNumMeasurements * dt));
  for (int i = 0; i <= kNumMeasurements; ++i) {
    integrator.Integrate(
        ImuMeasurement(TimestampFromSeconds(i * dt), kGyro, kAccel));
  }
  return integrator.Extract();
}

// Arbitrary but valid evaluation point (need not be zero-residual; only
// validity and unit quaternions matter for timing).
struct PoseState {
  Rigid3d pose_i = Rigid3d(
      Eigen::Quaterniond(Eigen::AngleAxisd(0.30, Eigen::Vector3d::UnitY())),
      Eigen::Vector3d(0.5, -0.2, 1.0));
  Rigid3d pose_j = Rigid3d(
      Eigen::Quaterniond(Eigen::AngleAxisd(0.35, Eigen::Vector3d::UnitY())),
      Eigen::Vector3d(0.6, -0.15, 1.1));
  double velocity_i[3] = {1.0, 0.5, -0.2};
  double velocity_j[3] = {1.05, 0.45, -0.25};
  // [bias_gyro(3), bias_accel(3)]
  double state_i[6] = {0, 0, 0, 0, 0, 0};
  double state_j[6] = {0, 0, 0, 0, 0, 0};
};

// Body-centric (6 parameter blocks: pose_i[7], velocity_i[3], state_i[6],
// pose_j[7], velocity_j[3], state_j[6]).
void BM_ImuBodyCentric(benchmark::State& state, Impl impl) {
  PreintegratedImuData data = MakeImuData();
  PoseState ps;
  const double* parameters[6] = {ps.pose_i.params.data(),
                                 ps.velocity_i,
                                 ps.state_i,
                                 ps.pose_j.params.data(),
                                 ps.velocity_j,
                                 ps.state_j};
  double residuals[15];
  double jac_pose_i[15 * 7], jac_vel_i[15 * 3], jac_state_i[15 * 6];
  double jac_pose_j[15 * 7], jac_vel_j[15 * 3], jac_state_j[15 * 6];
  double* jacobians[6] = {
      jac_pose_i, jac_vel_i, jac_state_i, jac_pose_j, jac_vel_j, jac_state_j};

  std::unique_ptr<ceres::CostFunction> cost_function =
      impl == Impl::kAutoDiff
          ? std::unique_ptr<ceres::CostFunction>(
                ImuPreintegrationCostFunctor::Create(&data, kGravity))
          : std::unique_ptr<ceres::CostFunction>(
                new AnalyticalImuPreintegrationCostFunction(&data, kGravity));

  for (auto _ : state) {
    cost_function->Evaluate(parameters, residuals, jacobians);
  }
}
BENCHMARK_CAPTURE(BM_ImuBodyCentric, AutoDiff, Impl::kAutoDiff);
BENCHMARK_CAPTURE(BM_ImuBodyCentric, Analytical, Impl::kAnalytical);

// Visual-centric (8 parameter blocks: log_scale[1], gravity_dir[3],
// pose_i[7], velocity_i[3], state_i[6], pose_j[7], velocity_j[3],
// state_j[6]).
void BM_ImuVisualCentric(benchmark::State& state, Impl impl) {
  PreintegratedImuData data = MakeImuData();
  PoseState ps;
  double log_scale[1] = {0.0};
  double gravity_dir[3] = {0, 0, -1};
  const double* parameters[8] = {log_scale,
                                 gravity_dir,
                                 ps.pose_i.params.data(),
                                 ps.velocity_i,
                                 ps.state_i,
                                 ps.pose_j.params.data(),
                                 ps.velocity_j,
                                 ps.state_j};
  double residuals[15];
  double jac_scale[15 * 1], jac_grav[15 * 3];
  double jac_pose_i[15 * 7], jac_vel_i[15 * 3], jac_state_i[15 * 6];
  double jac_pose_j[15 * 7], jac_vel_j[15 * 3], jac_state_j[15 * 6];
  double* jacobians[8] = {jac_scale,
                          jac_grav,
                          jac_pose_i,
                          jac_vel_i,
                          jac_state_i,
                          jac_pose_j,
                          jac_vel_j,
                          jac_state_j};

  std::unique_ptr<ceres::CostFunction> cost_function =
      impl == Impl::kAutoDiff
          ? std::unique_ptr<ceres::CostFunction>(
                VisualCentricImuPreintegrationCostFunctor::Create(&data))
          : std::unique_ptr<ceres::CostFunction>(
                new AnalyticalVisualCentricImuPreintegrationCostFunction(
                    &data));

  for (auto _ : state) {
    cost_function->Evaluate(parameters, residuals, jacobians);
  }
}
BENCHMARK_CAPTURE(BM_ImuVisualCentric, AutoDiff, Impl::kAutoDiff);
BENCHMARK_CAPTURE(BM_ImuVisualCentric, Analytical, Impl::kAnalytical);

}  // namespace

BENCHMARK_MAIN();

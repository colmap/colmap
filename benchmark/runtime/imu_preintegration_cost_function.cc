// Micro-benchmarks for the IMU preintegration cost functions: AutoDiff vs.
// analytical-Jacobian Evaluate() for the two cost function families
// (body-centric, visual-centric).
//
// Evaluate() reads only the fixed-size preintegrated result (delta_R/p/v, bias
// Jacobians, 15x15 sqrt_information), so its cost is invariant to both the
// number of integrated measurements and the integration method (MIDPOINT/RK4)
// that produced the data -- those only affect the one-off preintegration step,
// not this residual evaluation. The data is therefore built once with fixed
// settings below.
//
// Use the AutoDiff-vs-Analytical ratio to weigh whether the analytical
// Jacobians earn their (substantial) code/maintenance cost. Keep in mind IMU
// residual blocks are O(#frames), vastly outnumbered by O(#observations)
// reprojection blocks, so a per-Evaluate win may not move end-to-end BA time.

#include "colmap/estimators/imu_preintegration.h"
#include "colmap/estimators/imu_preintegration_cost.h"
#include "colmap/geometry/rigid3.h"
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
    integrator.FeedImu(
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
  // [velocity(3), bias_gyro(3), bias_accel(3)]
  double state_i[9] = {1.0, 0.5, -0.2, 0, 0, 0, 0, 0, 0};
  double state_j[9] = {1.05, 0.45, -0.25, 0, 0, 0, 0, 0, 0};
};

// Body-centric (4 parameter blocks: pose_i[7], state_i[9], pose_j[7],
// state_j[9]).
void BM_ImuBodyCentric(benchmark::State& state, Impl impl) {
  PreintegratedImuData data = MakeImuData();
  PoseState ps;
  const double* parameters[4] = {
      ps.pose_i.params.data(), ps.state_i, ps.pose_j.params.data(), ps.state_j};
  double residuals[15];
  double jac_pose_i[15 * 7], jac_state_i[15 * 9];
  double jac_pose_j[15 * 7], jac_state_j[15 * 9];
  double* jacobians[4] = {jac_pose_i, jac_state_i, jac_pose_j, jac_state_j};

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

// Visual-centric (7 parameter blocks: log_scale[1], gravity_dir[3],
// imu_from_cam[7], pose_i[7], state_i[9], pose_j[7], state_j[9]).
void BM_ImuVisualCentric(benchmark::State& state, Impl impl) {
  PreintegratedImuData data = MakeImuData();
  PoseState ps;
  double log_scale[1] = {0.0};
  double gravity_dir[3] = {0, 0, -1};
  double imu_from_cam[7] = {0, 0, 0, 1, 0, 0, 0};
  const double* parameters[7] = {log_scale,
                                 gravity_dir,
                                 imu_from_cam,
                                 ps.pose_i.params.data(),
                                 ps.state_i,
                                 ps.pose_j.params.data(),
                                 ps.state_j};
  double residuals[15];
  double jac_scale[15 * 1], jac_grav[15 * 3], jac_ic[15 * 7];
  double jac_pose_i[15 * 7], jac_state_i[15 * 9];
  double jac_pose_j[15 * 7], jac_state_j[15 * 9];
  double* jacobians[7] = {jac_scale,
                          jac_grav,
                          jac_ic,
                          jac_pose_i,
                          jac_state_i,
                          jac_pose_j,
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

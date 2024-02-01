#include "colmap/estimators/cost_functions.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/sensor/models.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <benchmark/benchmark.h>
#include <ceres/ceres.h>

using namespace colmap;
using camera_model = SimpleRadialCameraModel;

struct ReprojErrorData {
  Rigid3d cam_from_world;
  Eigen::Vector3d point3D;
  Eigen::Vector2d point2D;
  std::vector<double> camera_params;
};

static ReprojErrorData CreateReprojErrorData() {
  ReprojErrorData data{
      Rigid3d(Eigen::Quaterniond(0.9, 0.1, 0.1, 0.1), Eigen::Vector3d::Zero()),
      Eigen::Vector3d(1, 2, 10),
      Eigen::Vector2d(0.1, 0.2),
      {1, 0, 0, 0.1},
  };
  CHECK_EQ(data.camera_params.size(), camera_model::num_params);
  return std::move(data);
}

static void BM_ReprojErrorCostFunction(benchmark::State& state) {
  ReprojErrorData data = CreateReprojErrorData();
  const double* parameters[] = {data.cam_from_world.rotation.coeffs().data(),
                                data.cam_from_world.translation.data(),
                                data.point3D.data(),
                                data.camera_params.data()};
  double residuals[2];
  double jacobian_q[2 * 4];
  double jacobian_t[2 * 3];
  double jacobian_p[2 * 3];
  double jacobian_params[2 * camera_model::num_params];
  double* jacobians[] = {jacobian_q, jacobian_t, jacobian_p, jacobian_params};
  std::unique_ptr<ceres::CostFunction> cost_function(
      ReprojErrorCostFunction<camera_model>::Create(data.point2D));

  for (auto _ : state) {
    cost_function->Evaluate(parameters, residuals, jacobians);
  }
}
BENCHMARK(BM_ReprojErrorCostFunction);

static void BM_ReprojErrorConstantPoseCostFunction(benchmark::State& state) {
  ReprojErrorData data = CreateReprojErrorData();
  const double* parameters[] = {data.point3D.data(), data.camera_params.data()};
  double residuals[2];
  double jacobian_p[2 * 3];
  double jacobian_params[2 * camera_model::num_params];
  double* jacobians[] = {jacobian_p, jacobian_params};
  std::unique_ptr<ceres::CostFunction> cost_function(
      ReprojErrorConstantPoseCostFunction<camera_model>::Create(
          data.cam_from_world, data.point2D));

  for (auto _ : state) {
    cost_function->Evaluate(parameters, residuals, jacobians);
  }
}
BENCHMARK(BM_ReprojErrorConstantPoseCostFunction);

static void BM_ReprojErrorConstantPoint3DCostFunction(benchmark::State& state) {
  ReprojErrorData data = CreateReprojErrorData();
  const double* parameters[] = {data.cam_from_world.rotation.coeffs().data(),
                                data.cam_from_world.translation.data(),
                                data.camera_params.data()};
  double residuals[2];
  double jacobian_q[2 * 4];
  double jacobian_t[2 * 3];
  double jacobian_params[2 * camera_model::num_params];
  double* jacobians[] = {jacobian_q, jacobian_t, jacobian_params};
  std::unique_ptr<ceres::CostFunction> cost_function(
      ReprojErrorConstantPoint3DCostFunction<camera_model>::Create(
          data.point2D, data.point3D));

  for (auto _ : state) {
    cost_function->Evaluate(parameters, residuals, jacobians);
  }
}
BENCHMARK(BM_ReprojErrorConstantPoint3DCostFunction);

BENCHMARK_MAIN();

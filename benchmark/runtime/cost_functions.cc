#include "colmap/estimators/cost_functions/reprojection_error.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/sensor/models.h"
#include "colmap/util/eigen_alignment.h"

#include <memory>
#include <vector>

#include <benchmark/benchmark.h>
#include <ceres/ceres.h>

using namespace colmap;

namespace {

struct ReprojErrorData {
  Rigid3d cam_from_world;
  Eigen::Vector3d point3D;
  Eigen::Vector2d point2D;
  std::vector<double> camera_params;
};

// Nominal, in-frame camera parameters for each model. The 3D point is chosen so
// that its normalized coordinates are small and valid for all models.
template <typename CameraModel>
std::vector<double> NominalCameraParams();
template <>
std::vector<double> NominalCameraParams<SimplePinholeCameraModel>() {
  return {1000, 320, 240};
}
template <>
std::vector<double> NominalCameraParams<PinholeCameraModel>() {
  return {1000, 1000, 320, 240};
}
template <>
std::vector<double> NominalCameraParams<SimpleRadialCameraModel>() {
  return {1000, 320, 240, 0.01};
}
template <>
std::vector<double> NominalCameraParams<RadialCameraModel>() {
  return {1000, 320, 240, 0.01, 0.001};
}
template <>
std::vector<double> NominalCameraParams<OpenCVCameraModel>() {
  return {1000, 1000, 320, 240, 0.01, 0.001, 0.0001, 0.0001};
}
template <>
std::vector<double> NominalCameraParams<FullOpenCVCameraModel>() {
  return {1000,
          1000,
          320,
          240,
          0.01,
          0.001,
          0.0001,
          0.0001,
          0.001,
          0.0005,
          -0.0005,
          0.0001};
}
template <>
std::vector<double> NominalCameraParams<FOVCameraModel>() {
  return {1000, 1000, 320, 240, 0.5};
}
template <>
std::vector<double> NominalCameraParams<SimpleRadialFisheyeCameraModel>() {
  return {1000, 320, 240, 0.01};
}
template <>
std::vector<double> NominalCameraParams<RadialFisheyeCameraModel>() {
  return {1000, 320, 240, 0.01, 0.001};
}
template <>
std::vector<double> NominalCameraParams<OpenCVFisheyeCameraModel>() {
  return {1000, 1000, 320, 240, 0.01, 0.001, 0.0001, 0.0001};
}
template <>
std::vector<double> NominalCameraParams<ThinPrismFisheyeCameraModel>() {
  return {1000,
          1000,
          320,
          240,
          0.01,
          0.001,
          0.0001,
          0.0001,
          0.001,
          0.0005,
          0.0001,
          0.0001};
}
template <>
std::vector<double> NominalCameraParams<RadTanThinPrismFisheyeModel>() {
  return {1000,
          1000,
          320,
          240,
          0.01,
          0.001,
          0.0001,
          0.00001,
          0.000001,
          0.0000001,
          0.0001,
          0.0001,
          0.0001,
          0.00005,
          0.0001,
          0.00005};
}
template <>
std::vector<double> NominalCameraParams<SimpleDivisionCameraModel>() {
  return {1000, 320, 240, 0.01};
}
template <>
std::vector<double> NominalCameraParams<DivisionCameraModel>() {
  return {1000, 1000, 320, 240, 0.01};
}
template <>
std::vector<double> NominalCameraParams<SimpleFisheyeCameraModel>() {
  return {1000, 320, 240};
}
template <>
std::vector<double> NominalCameraParams<FisheyeCameraModel>() {
  return {1000, 1000, 320, 240};
}
template <>
std::vector<double> NominalCameraParams<EUCMCameraModel>() {
  return {1000, 1000, 320, 240, 0.5, 1.0};
}
template <>
std::vector<double> NominalCameraParams<EquirectangularCameraModel>() {
  return {640, 480};
}

template <typename CameraModel>
ReprojErrorData CreateReprojErrorData() {
  ReprojErrorData data{
      Rigid3d(Eigen::Quaterniond(0.9, 0.1, 0.1, 0.1).normalized(),
              Eigen::Vector3d(0.1, 0.2, 0.3)),
      Eigen::Vector3d(1, 2, 10),
      Eigen::Vector2d(320.1, 240.2),
      NominalCameraParams<CameraModel>(),
  };
  CHECK_EQ(data.camera_params.size(), CameraModel::num_params);
  return data;
}

// Fully-variable reprojection error (point, pose, calibration).
// The autodiff variant is obtained directly from the functor; the analytic
// variant is obtained through the production dispatch, which routes to the
// hand-written Jacobian for models that implement ImgFromCamWithJac().
template <typename CameraModel>
void BM_ReprojError(benchmark::State& state, bool analytic) {
  ReprojErrorData data = CreateReprojErrorData<CameraModel>();
  std::unique_ptr<ceres::CostFunction> cost_function(
      analytic ? CreateCameraCostFunction<ReprojErrorCostFunctor>(
                     CameraModel::model_id, data.point2D)
               : ReprojErrorCostFunctor<CameraModel>::Create(data.point2D));

  const double* parameters[3] = {data.point3D.data(),
                                 data.cam_from_world.params.data(),
                                 data.camera_params.data()};
  double residuals[2];
  double jacobian_point[2 * 3];
  double jacobian_pose[2 * 7];
  double jacobian_params[2 * CameraModel::num_params];
  double* jacobians[3] = {jacobian_point, jacobian_pose, jacobian_params};

  for (auto _ : state) {
    cost_function->Evaluate(parameters, residuals, jacobians);
  }
}

// Fixed-pose reprojection error (point, calibration), the local-BA hot path.
template <typename CameraModel>
void BM_ReprojErrorConstantPose(benchmark::State& state, bool analytic) {
  ReprojErrorData data = CreateReprojErrorData<CameraModel>();
  std::unique_ptr<ceres::CostFunction> cost_function(
      analytic ? CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor>(
                     CameraModel::model_id, data.point2D, data.cam_from_world)
               : ReprojErrorConstantPoseCostFunctor<CameraModel>::Create(
                     data.point2D, data.cam_from_world));

  const double* parameters[2] = {data.point3D.data(),
                                 data.camera_params.data()};
  double residuals[2];
  double jacobian_point[2 * 3];
  double jacobian_params[2 * CameraModel::num_params];
  double* jacobians[2] = {jacobian_point, jacobian_params};

  for (auto _ : state) {
    cost_function->Evaluate(parameters, residuals, jacobians);
  }
}

}  // namespace

#define REGISTER_MODEL(Model)                                                \
  BENCHMARK_CAPTURE(BM_ReprojError<Model>, Model##_AutoDiff, false);         \
  BENCHMARK_CAPTURE(BM_ReprojError<Model>, Model##_Analytic, true);          \
  BENCHMARK_CAPTURE(                                                         \
      BM_ReprojErrorConstantPose<Model>, Model##_ConstPose_AutoDiff, false); \
  BENCHMARK_CAPTURE(                                                         \
      BM_ReprojErrorConstantPose<Model>, Model##_ConstPose_Analytic, true);

REGISTER_MODEL(SimplePinholeCameraModel)
REGISTER_MODEL(PinholeCameraModel)
REGISTER_MODEL(SimpleRadialCameraModel)
REGISTER_MODEL(RadialCameraModel)
REGISTER_MODEL(OpenCVCameraModel)
REGISTER_MODEL(FullOpenCVCameraModel)
REGISTER_MODEL(FOVCameraModel)
REGISTER_MODEL(SimpleRadialFisheyeCameraModel)
REGISTER_MODEL(RadialFisheyeCameraModel)
REGISTER_MODEL(OpenCVFisheyeCameraModel)
REGISTER_MODEL(ThinPrismFisheyeCameraModel)
REGISTER_MODEL(RadTanThinPrismFisheyeModel)
REGISTER_MODEL(SimpleDivisionCameraModel)
REGISTER_MODEL(DivisionCameraModel)
REGISTER_MODEL(SimpleFisheyeCameraModel)
REGISTER_MODEL(FisheyeCameraModel)
REGISTER_MODEL(EUCMCameraModel)
REGISTER_MODEL(EquirectangularCameraModel)

BENCHMARK_MAIN();

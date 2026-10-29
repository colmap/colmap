// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/inertial/preintegration_cost.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindImuPreintegrationCosts(py::module& m) {
  m.def(
      "ImuPreintegrationCost",
      [](PreintegratedImuData& data, const Eigen::Vector3d& gravity) {
        return ImuPreintegrationCostFunctor::Create(&data, gravity);
      },
      "preintegrated_imu_data"_a,
      "gravity"_a,
      py::keep_alive<0, 1>(),
      "IMU preintegration cost function (body-centric, 4 parameter blocks). "
      "The data object must outlive the cost function.");

  m.def(
      "AnalyticalImuPreintegrationCost",
      [](PreintegratedImuData& data, const Eigen::Vector3d& gravity) {
        return std::unique_ptr<ceres::CostFunction>(
            new AnalyticalImuPreintegrationCostFunction(&data, gravity));
      },
      "preintegrated_imu_data"_a,
      "gravity"_a,
      py::keep_alive<0, 1>(),
      "IMU preintegration cost function with analytical Jacobians "
      "(body-centric, 4 parameter blocks). "
      "The data object must outlive the cost function.");

  m.def(
      "VisualCentricImuPreintegrationCost",
      [](PreintegratedImuData& data, const bool imu_from_cam_is_metric) {
        return VisualCentricImuPreintegrationCostFunctor::Create(
            &data, imu_from_cam_is_metric);
      },
      "preintegrated_imu_data"_a,
      "imu_from_cam_is_metric"_a = false,
      py::keep_alive<0, 1>(),
      "IMU preintegration cost function for post-hoc SfM refinement "
      "(7 parameter blocks: scale, gravity, extrinsics, poses, states). "
      "If imu_from_cam_is_metric, the extrinsics translation is metric and not "
      "scaled. "
      "The data object must outlive the cost function.");

  m.def(
      "AnalyticalVisualCentricImuPreintegrationCost",
      [](PreintegratedImuData& data, const bool imu_from_cam_is_metric) {
        return std::unique_ptr<ceres::CostFunction>(
            new AnalyticalVisualCentricImuPreintegrationCostFunction(
                &data, imu_from_cam_is_metric));
      },
      "preintegrated_imu_data"_a,
      "imu_from_cam_is_metric"_a = false,
      py::keep_alive<0, 1>(),
      "IMU preintegration cost function with analytical Jacobians for "
      "post-hoc SfM refinement (7 parameter blocks). "
      "If imu_from_cam_is_metric, the extrinsics translation is metric and not "
      "scaled. "
      "The data object must outlive the cost function.");

  m.def(
      "InertialRotationCost",
      [](PreintegratedImuData& data, const py::object& imu_from_cam) {
        if (py::isinstance<Rigid3d>(imu_from_cam)) {
          return InertialRotationCostFunctor::Create(
              &data, imu_from_cam.cast<const Rigid3d&>());
        }
        if (py::isinstance<pycolmap::Rotation3dWrapper>(imu_from_cam)) {
          return InertialRotationCostFunctor::Create(
              &data, imu_from_cam.cast<pycolmap::Rotation3dWrapper&>().map());
        }
        return InertialRotationCostFunctor::Create(
            &data, imu_from_cam.cast<Eigen::Quaterniond>());
      },
      "preintegrated_imu_data"_a,
      "imu_from_cam"_a,
      py::keep_alive<0, 1>(),
      "Inertial rotation cost function for rotation averaging "
      "(4 parameter blocks: i_from_world_aa[3], i_imu_state[9], "
      "j_from_world_aa[3], j_imu_state[9]). "
      "The imu_from_cam argument can be Rigid3d or Rotation3d. "
      "The data object must outlive the cost function.");

  m.def(
      "InertialGlobalPositioningCost",
      [](PreintegratedImuData& data,
         const Rigid3d& imu_from_cam,
         const Eigen::Quaterniond& i_from_world_q,
         const Eigen::Quaterniond& j_from_world_q) {
        return InertialGlobalPositioningCostFunctor::Create(
            &data, imu_from_cam, i_from_world_q, j_from_world_q);
      },
      "preintegrated_imu_data"_a,
      "imu_from_cam"_a,
      "i_from_world_q"_a,
      "j_from_world_q"_a,
      py::keep_alive<0, 1>(),
      "Inertial position and velocity cost function for global positioning "
      "(6 parameter blocks: log_scale[1], gravity_direction[3], "
      "i_center[3], i_imu_state[9], j_center[3], j_imu_state[9]). "
      "The data object must outlive the cost function.");
}

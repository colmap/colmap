// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/inertial/preintegration_cost.h"

#include "pycolmap/helpers.h"

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
      "IMU preintegration cost function (body-centric, 6 parameter blocks). "
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
      "(body-centric, 6 parameter blocks). "
      "The data object must outlive the cost function.");

  m.def(
      "VisualCentricImuPreintegrationCost",
      [](PreintegratedImuData& data) {
        return VisualCentricImuPreintegrationCostFunctor::Create(&data);
      },
      "preintegrated_imu_data"_a,
      py::keep_alive<0, 1>(),
      "IMU preintegration cost function for post-hoc SfM refinement "
      "(8 parameter blocks: scale, gravity, pose i, vel i, state i, pose j, "
      "vel j, state j). "
      "The data object must outlive the cost function.");

  m.def(
      "AnalyticalVisualCentricImuPreintegrationCost",
      [](PreintegratedImuData& data) {
        return std::unique_ptr<ceres::CostFunction>(
            new AnalyticalVisualCentricImuPreintegrationCostFunction(&data));
      },
      "preintegrated_imu_data"_a,
      py::keep_alive<0, 1>(),
      "IMU preintegration cost function with analytical Jacobians for "
      "post-hoc SfM refinement (8 parameter blocks: scale, gravity, pose i, "
      "vel i, state i, pose j, vel j, state j). "
      "The data object must outlive the cost function.");
}

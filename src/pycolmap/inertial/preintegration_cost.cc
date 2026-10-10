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
      [](PreintegratedImuData& data,
         const Eigen::Quaterniond& q_iori_i,
         const Eigen::Quaterniond& q_iori_j) {
        return VisualCentricImuPreintegrationCostFunctor::Create(
            &data, q_iori_i, q_iori_j);
      },
      "preintegrated_imu_data"_a,
      "q_iori_i"_a = Eigen::Quaterniond::Identity(),
      "q_iori_j"_a = Eigen::Quaterniond::Identity(),
      py::keep_alive<0, 1>(),
      "IMU preintegration cost function for post-hoc SfM refinement "
      "(7 parameter blocks: scale, gravity, extrinsics, poses, states). "
      "The data object must outlive the cost function.");

  m.def(
      "AnalyticalVisualCentricImuPreintegrationCost",
      [](PreintegratedImuData& data,
         const Eigen::Quaterniond& q_iori_i,
         const Eigen::Quaterniond& q_iori_j) {
        return std::unique_ptr<ceres::CostFunction>(
            new AnalyticalVisualCentricImuPreintegrationCostFunction(
                &data, q_iori_i, q_iori_j));
      },
      "preintegrated_imu_data"_a,
      "q_iori_i"_a = Eigen::Quaterniond::Identity(),
      "q_iori_j"_a = Eigen::Quaterniond::Identity(),
      py::keep_alive<0, 1>(),
      "IMU preintegration cost function with analytical Jacobians for "
      "post-hoc SfM refinement (7 parameter blocks). "
      "The data object must outlive the cost function.");
}

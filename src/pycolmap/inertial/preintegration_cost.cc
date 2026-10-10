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
         const Eigen::Quaterniond& q_iori_j,
         const bool metric_imu_from_cam) {
        return VisualCentricImuPreintegrationCostFunctor::Create(
            &data, q_iori_i, q_iori_j, metric_imu_from_cam);
      },
      "preintegrated_imu_data"_a,
      "q_iori_i"_a = Eigen::Quaterniond::Identity(),
      "q_iori_j"_a = Eigen::Quaterniond::Identity(),
      "metric_imu_from_cam"_a = false,
      py::keep_alive<0, 1>(),
      "IMU preintegration cost function for post-hoc SfM refinement "
      "(7 parameter blocks: scale, gravity, extrinsics, poses, states). "
      "The data object must outlive the cost function.");

  m.def(
      "AnalyticalVisualCentricImuPreintegrationCost",
      [](PreintegratedImuData& data,
         const Eigen::Quaterniond& q_iori_i,
         const Eigen::Quaterniond& q_iori_j,
         const bool metric_imu_from_cam) {
        return std::unique_ptr<ceres::CostFunction>(
            new AnalyticalVisualCentricImuPreintegrationCostFunction(
                &data, q_iori_i, q_iori_j, metric_imu_from_cam));
      },
      "preintegrated_imu_data"_a,
      "q_iori_i"_a = Eigen::Quaterniond::Identity(),
      "q_iori_j"_a = Eigen::Quaterniond::Identity(),
      "metric_imu_from_cam"_a = false,
      py::keep_alive<0, 1>(),
      "IMU preintegration cost function with analytical Jacobians for "
      "post-hoc SfM refinement (7 parameter blocks). "
      "The data object must outlive the cost function.");

  m.def(
      "InertialRotationCost",
      [](PreintegratedImuData& data,
         const py::object& imu_from_cam,
         const Eigen::Quaterniond& q_iori_i,
         const Eigen::Quaterniond& q_iori_j) {
        if (py::isinstance<Rigid3d>(imu_from_cam)) {
          return InertialRotationCostFunctor::Create(
              &data,
              imu_from_cam.cast<const Rigid3d&>(),
              Eigen::Matrix<double, 6, 6>::Zero(),
              q_iori_i,
              q_iori_j);
        }
        if (py::isinstance<pycolmap::Rotation3dWrapper>(imu_from_cam)) {
          return InertialRotationCostFunctor::Create(
              &data,
              imu_from_cam.cast<pycolmap::Rotation3dWrapper&>().map(),
              Eigen::Matrix<double, 6, 6>::Zero(),
              q_iori_i,
              q_iori_j);
        }
        return InertialRotationCostFunctor::Create(
            &data,
            imu_from_cam.cast<Eigen::Quaterniond>(),
            Eigen::Matrix<double, 6, 6>::Zero(),
            q_iori_i,
            q_iori_j);
      },
      "preintegrated_imu_data"_a,
      "imu_from_cam"_a,
      "q_iori_i"_a = Eigen::Quaterniond::Identity(),
      "q_iori_j"_a = Eigen::Quaterniond::Identity(),
      py::keep_alive<0, 1>(),
      "Inertial rotation cost function for rotation averaging "
      "(4 parameter blocks: i_from_world_aa[3], i_imu_state[9], "
      "j_from_world_aa[3], j_imu_state[9]). "
      "The imu_from_cam argument can be Rigid3d or Rotation3d. "
      "The data object must outlive the cost function.");

  m.def(
      "InertialRotationCost",
      [](PreintegratedImuData& data,
         const py::object& imu_from_cam,
         const Eigen::Matrix<double, 6, 6>& sqrt_info,
         const Eigen::Quaterniond& q_iori_i,
         const Eigen::Quaterniond& q_iori_j) {
        if (py::isinstance<Rigid3d>(imu_from_cam)) {
          return InertialRotationCostFunctor::Create(
              &data,
              imu_from_cam.cast<const Rigid3d&>(),
              sqrt_info,
              q_iori_i,
              q_iori_j);
        }
        if (py::isinstance<pycolmap::Rotation3dWrapper>(imu_from_cam)) {
          return InertialRotationCostFunctor::Create(
              &data,
              imu_from_cam.cast<pycolmap::Rotation3dWrapper&>().map(),
              sqrt_info,
              q_iori_i,
              q_iori_j);
        }
        return InertialRotationCostFunctor::Create(
            &data,
            imu_from_cam.cast<Eigen::Quaterniond>(),
            sqrt_info,
            q_iori_i,
            q_iori_j);
      },
      "preintegrated_imu_data"_a,
      "imu_from_cam"_a,
      "sqrt_info"_a,
      "q_iori_i"_a = Eigen::Quaterniond::Identity(),
      "q_iori_j"_a = Eigen::Quaterniond::Identity(),
      py::keep_alive<0, 1>(),
      "Inertial rotation cost function for rotation averaging with custom "
      "sqrt_info matrix "
      "(4 parameter blocks: i_from_world_aa[3], i_imu_state[9], "
      "j_from_world_aa[3], j_imu_state[9]). "
      "The imu_from_cam argument can be Rigid3d or Rotation3d. "
      "The data object must outlive the cost function.");

  m.def(
      "InertialGlobalPositioningCost",
      [](PreintegratedImuData& data,
         const Rigid3d& imu_from_cam,
         const Eigen::Quaterniond& i_from_world_q,
         const Eigen::Quaterniond& j_from_world_q,
         const bool metric_imu_from_cam) {
        return InertialGlobalPositioningCostFunctor::Create(
            &data,
            imu_from_cam,
            i_from_world_q,
            j_from_world_q,
            Eigen::Matrix<double, 9, 9>::Zero(),
            metric_imu_from_cam);
      },
      "preintegrated_imu_data"_a,
      "imu_from_cam"_a,
      "i_from_world_q"_a,
      "j_from_world_q"_a,
      "metric_imu_from_cam"_a = false,
      py::keep_alive<0, 1>(),
      "Inertial position and velocity cost function for global positioning "
      "(6 parameter blocks: log_scale[1], gravity_direction[3], "
      "i_center[3], i_imu_state[9], j_center[3], j_imu_state[9]). "
      "The data object must outlive the cost function.");
}

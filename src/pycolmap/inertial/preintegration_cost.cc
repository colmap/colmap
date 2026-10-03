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
      [](PreintegratedImuData& data) {
        return VisualCentricImuPreintegrationCostFunctor::Create(&data);
      },
      "preintegrated_imu_data"_a,
      py::keep_alive<0, 1>(),
      "IMU preintegration cost function for post-hoc SfM refinement "
      "(7 parameter blocks: scale, gravity, extrinsics, poses, states). "
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
      "post-hoc SfM refinement (7 parameter blocks). "
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

  m.def(
      "BiasPriorCost",
      [](const Eigen::Vector3d& prior_bias, int bias_offset) {
        return BiasPriorCostFunctor<9>::Create(prior_bias, bias_offset);
      },
      "prior_bias"_a,
      "bias_offset"_a = 3,
      "3-DoF error on a 3D slice of 9D IMU state at given offset.");
  m.def(
      "BiasPriorCost",
      [](double stddev, const Eigen::Vector3d& prior_bias, int bias_offset) {
        return BiasPriorCostFunctor<9>::Create(prior_bias, stddev, bias_offset);
      },
      "prior_stddev"_a,
      "prior_bias"_a,
      "bias_offset"_a = 3,
      "3-DoF error on a 3D slice of 9D IMU state with isotropic standard "
      "deviation.");
  m.def(
      "BiasPriorCost",
      [](const Eigen::Vector3d& prior_bias, double stddev, int bias_offset) {
        return BiasPriorCostFunctor<9>::Create(prior_bias, stddev, bias_offset);
      },
      "prior_bias"_a,
      "prior_stddev"_a,
      "bias_offset"_a = 3,
      "3-DoF error on a 3D slice of 9D IMU state with isotropic standard "
      "deviation.");
  m.def(
      "BiasPriorCost",
      [](const Eigen::Matrix3d& cov,
         const Eigen::Vector3d& prior_bias,
         int bias_offset) {
        return BiasPriorCostFunctor<9>::Create(prior_bias, cov, bias_offset);
      },
      "prior_cov"_a,
      "prior_bias"_a,
      "bias_offset"_a = 3,
      "3-DoF error on a 3D slice of 9D IMU state with prior covariance.");
  m.def(
      "BiasPriorCost",
      [](const Eigen::Vector3d& prior_bias,
         const Eigen::Matrix3d& cov,
         int bias_offset) {
        return BiasPriorCostFunctor<9>::Create(prior_bias, cov, bias_offset);
      },
      "prior_bias"_a,
      "prior_cov"_a,
      "bias_offset"_a = 3,
      "3-DoF error on a 3D slice of 9D IMU state with prior covariance.");
  m.def(
      "BiasPriorCost",
      [](const Eigen::Vector3d& stddev_vec,
         const Eigen::Vector3d& prior_bias,
         int bias_offset) {
        return ScaleWeightedCostFunctor<BiasPriorCostFunctor<9>>::Create(
            stddev_vec, prior_bias, bias_offset);
      },
      "prior_stddev"_a,
      "prior_bias"_a,
      "bias_offset"_a = 3,
      "3-DoF error on a 3D slice of 9D IMU state with per-axis standard "
      "deviations.");

  m.def(
      "GyroBiasPriorCost",
      [](const Eigen::Vector3d& prior_bias) {
        return BiasPriorCostFunctor<9>::CreateGyro(prior_bias);
      },
      "prior_bias"_a,
      "3-DoF error on IMU gyro bias (slice [3:6] of 9D IMU state).");
  m.def(
      "GyroBiasPriorCost",
      [](double stddev, const Eigen::Vector3d& prior_bias) {
        return BiasPriorCostFunctor<9>::CreateGyro(prior_bias, stddev);
      },
      "prior_stddev"_a,
      "prior_bias"_a,
      "3-DoF error on IMU gyro bias with isotropic standard deviation.");
  m.def(
      "GyroBiasPriorCost",
      [](const Eigen::Vector3d& prior_bias, double stddev) {
        return BiasPriorCostFunctor<9>::CreateGyro(prior_bias, stddev);
      },
      "prior_bias"_a,
      "prior_stddev"_a,
      "3-DoF error on IMU gyro bias with isotropic standard deviation.");
  m.def(
      "GyroBiasPriorCost",
      [](const Eigen::Matrix3d& cov, const Eigen::Vector3d& prior_bias) {
        return BiasPriorCostFunctor<9>::CreateGyro(prior_bias, cov);
      },
      "prior_cov"_a,
      "prior_bias"_a,
      "3-DoF error on IMU gyro bias with prior covariance.");
  m.def(
      "GyroBiasPriorCost",
      [](const Eigen::Vector3d& prior_bias, const Eigen::Matrix3d& cov) {
        return BiasPriorCostFunctor<9>::CreateGyro(prior_bias, cov);
      },
      "prior_bias"_a,
      "prior_cov"_a,
      "3-DoF error on IMU gyro bias with prior covariance.");
  m.def(
      "GyroBiasPriorCost",
      [](const Eigen::Vector3d& stddev_vec, const Eigen::Vector3d& prior_bias) {
        return ScaleWeightedCostFunctor<BiasPriorCostFunctor<9>>::Create(
            stddev_vec, prior_bias, 3);
      },
      "prior_stddev"_a,
      "prior_bias"_a,
      "3-DoF error on IMU gyro bias with per-axis standard deviations.");

  m.def(
      "AccelBiasPriorCost",
      [](const Eigen::Vector3d& prior_bias) {
        return BiasPriorCostFunctor<9>::CreateAccel(prior_bias);
      },
      "prior_bias"_a,
      "3-DoF error on IMU accel bias (slice [6:9] of 9D IMU state).");
  m.def(
      "AccelBiasPriorCost",
      [](double stddev, const Eigen::Vector3d& prior_bias) {
        return BiasPriorCostFunctor<9>::CreateAccel(prior_bias, stddev);
      },
      "prior_stddev"_a,
      "prior_bias"_a,
      "3-DoF error on IMU accel bias with isotropic standard deviation.");
  m.def(
      "AccelBiasPriorCost",
      [](const Eigen::Vector3d& prior_bias, double stddev) {
        return BiasPriorCostFunctor<9>::CreateAccel(prior_bias, stddev);
      },
      "prior_bias"_a,
      "prior_stddev"_a,
      "3-DoF error on IMU accel bias with isotropic standard deviation.");
  m.def(
      "AccelBiasPriorCost",
      [](const Eigen::Matrix3d& cov, const Eigen::Vector3d& prior_bias) {
        return BiasPriorCostFunctor<9>::CreateAccel(prior_bias, cov);
      },
      "prior_cov"_a,
      "prior_bias"_a,
      "3-DoF error on IMU accel bias with prior covariance.");
  m.def(
      "AccelBiasPriorCost",
      [](const Eigen::Vector3d& prior_bias, const Eigen::Matrix3d& cov) {
        return BiasPriorCostFunctor<9>::CreateAccel(prior_bias, cov);
      },
      "prior_bias"_a,
      "prior_cov"_a,
      "3-DoF error on IMU accel bias with prior covariance.");
  m.def(
      "AccelBiasPriorCost",
      [](const Eigen::Vector3d& stddev_vec, const Eigen::Vector3d& prior_bias) {
        return ScaleWeightedCostFunctor<BiasPriorCostFunctor<9>>::Create(
            stddev_vec, prior_bias, 6);
      },
      "prior_stddev"_a,
      "prior_bias"_a,
      "3-DoF error on IMU accel bias with per-axis standard deviations.");
}

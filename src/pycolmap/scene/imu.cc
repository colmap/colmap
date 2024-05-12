#include "colmap/sensor/imu.h"

#include "colmap/scene/imu.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/utils.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindImu(py::module& m) {
  py::class_<ImuCalibration>(m, "ImuCalibration")
      .def(py::init<>())
      .def_readwrite("acc_noise_density", &ImuCalibration::acc_noise_density)
      .def_readwrite("gyro_noise_density", &ImuCalibration::gyro_noise_density)
      .def_readwrite("acc_bias_random_walk_sigma",
                     &ImuCalibration::acc_bias_random_walk_sigma)
      .def_readwrite("gyro_bias_random_walk_sigma",
                     &ImuCalibration::gyro_bias_random_walk_sigma)
      .def_readwrite("acc_saturation_max", &ImuCalibration::acc_saturation_max)
      .def_readwrite("gyro_saturation_max",
                     &ImuCalibration::gyro_saturation_max)
      .def_readwrite("gravity_magnitude", &ImuCalibration::gravity_magnitude)
      .def_readwrite("imu_rate", &ImuCalibration::imu_rate);

  py::class_<ImuMeasurement>(m, "ImuMeasurement")
      .def(py::init<const double,
                    const Eigen::Vector3d&,
                    const Eigen::Vector3d&>())
      .def_readwrite("timestamp", &ImuMeasurement::timestamp)
      .def_readwrite("linear_acceleration",
                     &ImuMeasurement::linear_acceleration)
      .def_readwrite("angular_velocity", &ImuMeasurement::angular_velocity);

  py::class_<Imu>(m, "Imu")
      .def(py::init<>())
      .def_readwrite("imu_id", &Imu::camera_id)
      .def_readwrite("camera_id", &Imu::camera_id)
      .def_readwrite("cam_to_imu", &Imu::cam_to_imu);

  py::class_<ImuState>(m, "ImuState")
      .def(py::init<>())
      .def_property_readonly("data", &ImuState::Data)
      .def_property_readonly("velocity", &ImuState::Velocity)
      .def_property_readonly("velocity_ptr", &ImuState::VelocityPtr)
      .def_property_readonly("acc_bias", &ImuState::AccBias)
      .def_property_readonly("acc_bias_ptr", &ImuState::AccBiasPtr)
      .def_property_readonly("gyro_bias", &ImuState::GyroBias)
      .def_property_readonly("gyro_bias_ptr", &ImuState::GyroBiasPtr);
}

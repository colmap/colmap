#include "colmap/scene/imu.h"

#include "colmap/sensor/imu.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/scene/types.h"
#include "pycolmap/utils.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindImu(py::module& m) {
  py::classh_ext<ImuCalibration> PyImuCalibration(m, "ImuCalibration");
  PyImuCalibration.def(py::init<>())
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
  MakeDataclass(PyImuCalibration);

  py::classh_ext<ImuMeasurement> PyImuMeasurement(m, "ImuMeasurement");
  PyImuMeasurement.def(py::init<>())
      .def(py::init<timestamp_t,
                    const Eigen::Vector3d&,
                    const Eigen::Vector3d&>())
      .def_readwrite("timestamp", &ImuMeasurement::timestamp)
      .def_readwrite("accel", &ImuMeasurement::accel)
      .def_readwrite("gyro", &ImuMeasurement::gyro)
      .def("__repr__", [](const ImuMeasurement& m) {
        std::ostringstream ss;
        ss << m;
        return ss.str();
      });
  MakeDataclass(PyImuMeasurement);

  py::bind_vector<ImuMeasurements>(m, "ImuMeasurements");

  m.def("get_measurements_contain_edge",
        &GetMeasurementsContainEdge,
        "measurements"_a,
        "t1"_a,
        "t2"_a);

  py::classh<Imu>(m, "Imu")
      .def(py::init<>())
      .def_readwrite("imu_id", &Imu::imu_id)
      .def_readwrite("camera_id", &Imu::camera_id)
      .def_readwrite("imu_from_cam", &Imu::imu_from_cam)
      .def("__repr__", [](const Imu& s) {
        std::ostringstream ss;
        ss << s;
        return ss.str();
      });

  py::classh<ImuState>(m, "ImuState")
      .def(py::init<>())
      .def("set_velocity", &ImuState::SetVelocity)
      .def("set_acc_bias", &ImuState::SetAccBias)
      .def("set_gyro_bias", &ImuState::SetGyroBias)
      .def_property(
          "data",
          py::overload_cast<>(&ImuState::Data),
          [](ImuState& self, const Eigen::Matrix<double, 9, 1>& data) {
            self.Data() = data;
          })
      .def_property_readonly("velocity", &ImuState::Velocity)
      .def_property_readonly("velocity_ptr", &ImuState::VelocityPtr)
      .def_property_readonly("acc_bias", &ImuState::AccBias)
      .def_property_readonly("acc_bias_ptr", &ImuState::AccBiasPtr)
      .def_property_readonly("gyro_bias", &ImuState::GyroBias)
      .def_property_readonly("gyro_bias_ptr", &ImuState::GyroBiasPtr)
      .def("__repr__", [](const ImuState& s) {
        std::ostringstream ss;
        ss << s;
        return ss.str();
      });
}

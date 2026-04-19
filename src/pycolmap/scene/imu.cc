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
  py::classh<ImuCalibration> PyImuCalibration(m, "ImuCalibration");
  PyImuCalibration.def(py::init<>())
      .def_readwrite("accel_noise_density",
                     &ImuCalibration::accel_noise_density)
      .def_readwrite("gyro_noise_density", &ImuCalibration::gyro_noise_density)
      .def_readwrite("bias_accel_random_walk_sigma",
                     &ImuCalibration::bias_accel_random_walk_sigma)
      .def_readwrite("bias_gyro_random_walk_sigma",
                     &ImuCalibration::bias_gyro_random_walk_sigma)
      .def_readwrite("accel_saturation_max",
                     &ImuCalibration::accel_saturation_max)
      .def_readwrite("gyro_saturation_max",
                     &ImuCalibration::gyro_saturation_max)
      .def_readwrite("gravity_magnitude", &ImuCalibration::gravity_magnitude)
      .def_readwrite("accel_rectification",
                     &ImuCalibration::accel_rectification)
      .def_readwrite("gyro_rectification", &ImuCalibration::gyro_rectification)
      .def_readwrite("imu_rate", &ImuCalibration::imu_rate);
  MakeDataclass(PyImuCalibration);

  py::classh_ext<ImuMeasurement> PyImuMeasurement(m, "ImuMeasurement");
  PyImuMeasurement.def(py::init<>())
      .def(py::init<timestamp_t,
                    const Eigen::Vector3d&,
                    const Eigen::Vector3d&>(),
           "timestamp"_a,
           "gyro"_a,
           "accel"_a)
      .def_readwrite("timestamp", &ImuMeasurement::timestamp)
      .def_readwrite("gyro", &ImuMeasurement::gyro)
      .def_readwrite("accel", &ImuMeasurement::accel)
      .def("__repr__", [](const ImuMeasurement& m) {
        std::ostringstream ss;
        ss << m;
        return ss.str();
      });
  MakeDataclass(PyImuMeasurement);

  py::classh<ImuMeasurements>(m, "ImuMeasurements")
      .def(py::init<>())
      .def("insert",
           py::overload_cast<const ImuMeasurement&>(&ImuMeasurements::Insert),
           "measurement"_a)
      .def("insert",
           py::overload_cast<const std::vector<ImuMeasurement>&>(
               &ImuMeasurements::Insert),
           "measurements"_a)
      .def("insert",
           py::overload_cast<const ImuMeasurements&>(&ImuMeasurements::Insert),
           "measurements"_a)
      .def("insert_sorted",
           &ImuMeasurements::InsertSorted,
           "sorted_measurements"_a)
      .def("clear", &ImuMeasurements::Clear)
      .def("empty", &ImuMeasurements::Empty)
      .def("__len__", &ImuMeasurements::Size)
      .def("__getitem__", &ImuMeasurements::operator[])
      .def("extract_measurements_contain_edge",
           &ImuMeasurements::ExtractMeasurementsContainEdge,
           "t1"_a,
           "t2"_a)
      .def(
          "__iter__",
          [](const ImuMeasurements& ms) {
            return py::make_iterator(ms.begin(), ms.end());
          },
          py::keep_alive<0, 1>())
      .def("__repr__", [](const ImuMeasurements& ms) {
        return "ImuMeasurements(size=" + std::to_string(ms.Size()) + ")";
      });

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
      .def(py::init<const Eigen::Vector3d&,
                    const Eigen::Vector3d&,
                    const Eigen::Vector3d&>(),
           "velocity"_a,
           "bias_gyro"_a,
           "bias_accel"_a)
      .def_property(
          "params",
          [](ImuState& self) -> Eigen::Matrix<double, 9, 1>& {
            return self.params;
          },
          [](ImuState& self, const Eigen::Matrix<double, 9, 1>& params) {
            self.params = params;
          })
      .def_property(
          "velocity",
          [](py::object self) {
            ImuState& state = self.cast<ImuState&>();
            return py::array_t<double>(
                {3}, {sizeof(double)}, state.params.data(), self);
          },
          [](ImuState& self, const Eigen::Vector3d& v) { self.velocity() = v; })
      .def_property(
          "bias_gyro",
          [](py::object self) {
            ImuState& state = self.cast<ImuState&>();
            return py::array_t<double>(
                {3}, {sizeof(double)}, state.params.data() + 3, self);
          },
          [](ImuState& self, const Eigen::Vector3d& bg) {
            self.bias_gyro() = bg;
          })
      .def_property(
          "bias_accel",
          [](py::object self) {
            ImuState& state = self.cast<ImuState&>();
            return py::array_t<double>(
                {3}, {sizeof(double)}, state.params.data() + 6, self);
          },
          [](ImuState& self, const Eigen::Vector3d& ba) {
            self.bias_accel() = ba;
          })
      .def("__repr__", [](const ImuState& s) {
        std::ostringstream ss;
        ss << s;
        return ss.str();
      });
}

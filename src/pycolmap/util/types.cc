#include "colmap/feature/types.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindUtilTypes(py::module& m) {
  auto PySensorType = py::enum_<SensorType>(m, "SensorType")
                          .value("INVALID", SensorType::INVALID)
                          .value("CAMERA", SensorType::CAMERA)
                          .value("IMU", SensorType::IMU);
  AddStringToEnumConstructor(PySensorType);

  auto PySensorT = py::class_<sensor_t>(m, "sensor_t")
                       .def(py::init<>())
                       .def_readwrite("type", &sensor_t::type)
                       .def_readwrite("id", &sensor_t::id);
  MakeDataclass(PySensorT);

  auto PyDataT = py::class_<data_t>(m, "data_t")
                     .def(py::init<>())
                     .def_readwrite("sensor_id", &data_t::sensor_id)
                     .def_readwrite("id", &data_t::id);
  MakeDataclass(PyDataT);
}

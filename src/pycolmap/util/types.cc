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

  auto PySensorT = py::classh<sensor_t>(m, "sensor_t")
                       .def(py::init<>())
                       .def(py::init<SensorType, uint32_t>(), "type"_a, "id"_a)
                       .def_readwrite("type", &sensor_t::type)
                       .def_readwrite("id", &sensor_t::id);
  MakeDataclass(PySensorT);

  auto PyDataT = py::classh<data_t>(m, "data_t")
                     .def(py::init<>())
                     .def(py::init<sensor_t, uint32_t>(), "sensor_id"_a, "id"_a)
                     .def_readwrite("sensor_id", &data_t::sensor_id)
                     .def_readwrite("id", &data_t::id);
  MakeDataclass(PyDataT);

  m.def("image_pair_to_pair_id",
        &ImagePairToPairId,
        "image_id1"_a,
        "image_id2"_a);
  m.def("pair_id_to_image_pair", &PairIdToImagePair, "pair_id"_a);
  m.def("should_swap_image_pair",
        &ShouldSwapImagePair,
        "image_id1"_a,
        "image_id2"_a);
}

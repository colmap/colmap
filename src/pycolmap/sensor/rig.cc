#include "colmap/sensor/rig.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/scene/types.h"

#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindSensorRig(py::module& m) {
  py::classh<Rig> PyRig(m, "Rig");
  PyRig.def(py::init<>())
      .def_property("rig_id",
                    &Rig::RigId,
                    &Rig::SetRigId,
                    "Unique identifier of the rig.")
      .def("add_ref_sensor",
           &Rig::AddRefSensor,
           "sensor_id"
           "Add reference sensor.")
      .def("add_sensor",
           &Rig::AddSensor,
           "sensor_id"
           "Add non-reference sensor.")
      .def("has_sensor",
           &Rig::HasSensor,
           "Whether the rig has a specific sensor.")
      .def("is_ref_sensor",
           &Rig::IsRefSensor,
           "Check whether the given sensor is the reference sensor.")
      .def("num_sensors", &Rig::NumSensors, "The number of sensors in the rig.")
      .def_property_readonly("ref_sensor_id",
                             &Rig::RefSensorId,
                             "The reference sensor's identifier.")
      .def("sensor_ids",
           &Rig::SensorIds,
           "Get all sensor ids (including the reference sensor) in the rig.")
      .def("sensor_from_rig",
           py::overload_cast<sensor_t>(&Rig::MaybeSensorFromRig),
           "The the transformation from rig to the sensor.")
      .def("set_sensor_from_rig",
           py::overload_cast<sensor_t, const std::optional<Rigid3d>&>(
               &Rig::SetSensorFromRig),
           "Set the sensor_from_rig transformation.")
      .def_property_readonly(
          "non_ref_sensors",
          py::overload_cast<>(&Rig::NonRefSensors),
          py::return_value_policy::reference_internal,
          "Access all sensors in the rig except for reference sensor");
  MakeDataclass(PyRig);

  py::bind_map<RigMap>(m, "RigMap");
}

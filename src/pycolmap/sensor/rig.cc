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
  py::class_<Rig, std::shared_ptr<Rig>> PyRig(m, "Rig");
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
      .def(
          "sensor_from_rig",
          [](Rig& self, sensor_t sensor_id) -> py::typing::Optional<Rigid3d> {
            if (const std::optional<Rigid3d> sensor_from_rig =
                    self.MaybeSensorFromRig(sensor_id);
                sensor_from_rig.has_value()) {
              return py::cast(*sensor_from_rig);
            } else {
              return py::none();
            }
          },
          "The pose of the frame, defined as the transformation from world to "
          "rig space.")
      .def_property_readonly(
          "sensors",
          py::overload_cast<>(&Rig::Sensors),
          py::return_value_policy::reference_internal,
          "Access all sensors in the rig except for reference sensor");
  MakeDataclass(PyRig);

  py::bind_map<RigMap>(m, "RigMap");
}

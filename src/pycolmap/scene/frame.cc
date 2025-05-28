#include "colmap/scene/frame.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/scene/types.h"

#include <memory>
#include <optional>
#include <sstream>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindFrame(py::module& m) {
  py::class_<Frame, std::shared_ptr<Frame>> PyFrame(m, "Frame");
  PyFrame.def(py::init<>())
      .def_property("frame_id",
                    &Frame::FrameId,
                    &Frame::SetFrameId,
                    "Unique identifier of the frame.")
      .def_property("rig_id",
                    &Frame::RigId,
                    &Frame::SetRigId,
                    "Unique identifier of the rig.")
      .def("add_data_id", &Frame::AddDataId, "Associate data with frame.")
      .def("has_data",
           &Frame::HasDataId,
           "Check whether frame has associated data.")
      .def_property_readonly(
          "data_ids",
          [](const Frame& self) { return self.DataIds(); },
          "The associated data.")
      .def_property(
          "rig",
          [](Frame& self) -> py::typing::Optional<Rig> {
            if (self.HasRigPtr()) {
              return py::cast(self.RigPtr());
            } else {
              return py::none();
            }
          },
          &Frame::SetRigPtr,
          "The associated rig object.")
      .def("reset_rig_ptr",
           &Frame::ResetRigPtr,
           "Make the rig pointer a nullptr.")
      .def_property(
          "rig_from_world",
          [](Frame& self) -> py::typing::Optional<Rigid3d> {
            if (self.HasPose()) {
              return py::cast(self.RigFromWorld());
            } else {
              return py::none();
            }
          },
          [](Frame& self, const Rigid3d& rig_from_world) {
            self.SetRigFromWorld(rig_from_world);
          },
          "The pose of the frame, defined as the transformation from world to "
          "rig space.")
      .def("has_pose", &Frame::HasPose, "Whether the frame has a valid pose.")
      .def("reset_pose", &Frame::ResetPose, "Invalidate the pose of the frame.")
      .def("sensor_from_world",
           &Frame::SensorFromWorld,
           "sensor_id"_a,
           "The transformation from the world to a specific sensor.");
  MakeDataclass(PyFrame);

  py::bind_map<FrameMap>(m, "FrameMap");
}

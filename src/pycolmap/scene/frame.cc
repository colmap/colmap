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
  py::classh<Frame> PyFrame(m, "Frame");
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
      .def("num_data_ids",
           &Frame::NumDataIds,
           "Number of associated data items in frame.")
      .def("has_data",
           &Frame::HasDataId,
           "Check whether frame has associated data.")
      .def_property_readonly(
          "data_ids",
          [](const Frame& self) { return self.DataIds(); },
          "The associated data.")
      // Cannot have the same name as the property "data_ids" above due to
      // pybind11 limitations.
      .def(
          "data_ids_by_sensor",
          [](const Frame& self, SensorType type) {
            const auto data_ids = self.DataIds(type);
            return std::vector<data_t>(data_ids.begin(), data_ids.end());
          },
          "type"_a,
          "The associated data for a given sensor type.")
      .def_property_readonly(
          "image_ids",
          [](const Frame& self) {
            const auto image_ids = self.ImageIds();
            return std::vector<data_t>(image_ids.begin(), image_ids.end());
          },
          "The associated image data.")
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
          py::overload_cast<>(&Frame::MaybeRigFromWorld),
          py::overload_cast<const std::optional<Rigid3d>&>(
              &Frame::SetRigFromWorld),
          "The pose of the frame, defined as the transformation from world to "
          "rig space.")
      .def("has_pose", &Frame::HasPose, "Whether the frame has a valid pose.")
      .def("reset_pose", &Frame::ResetPose, "Invalidate the pose of the frame.")
      .def("sensor_from_world",
           &Frame::SensorFromWorld,
           "sensor_id"_a,
           "The transformation from the world to a specific sensor.")
      .def("set_cam_from_world",
           &Frame::SetCamFromWorld,
           "camera_id"_a,
           "cam_from_world"_a,
           "Set the world to frame from the given camera from world "
           "transformation.");
  MakeDataclass(PyFrame);

  py::bind_map<FrameMap>(m, "FrameMap");
}

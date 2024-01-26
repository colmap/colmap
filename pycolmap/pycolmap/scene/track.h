#include "colmap/scene/track.h"
#include "colmap/util/misc.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"
#include "pycolmap/log_exceptions.h"

#include <memory>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindTrack(py::module& m) {
  py::class_<TrackElement, std::shared_ptr<TrackElement>> PyTrackElement(
      m, "TrackElement");
  PyTrackElement.def(py::init<>())
      .def(py::init<image_t, point2D_t>())
      .def_readwrite("image_id", &TrackElement::image_id)
      .def_readwrite("point2D_idx", &TrackElement::point2D_idx)
      .def("__repr__", [](const TrackElement& self) {
        return "TrackElement(image_id=" + std::to_string(self.image_id) +
               ", point2D_idx=" + std::to_string(self.point2D_idx) + ")";
      });
  MakeDataclass(PyTrackElement);

  py::class_<Track, std::shared_ptr<Track>> PyTrack(m, "Track");
  PyTrack.def(py::init<>())
      .def(py::init([](const std::vector<TrackElement>& elements) {
        auto track = std::make_shared<Track>();
        track->AddElements(elements);
        return track;
      }))
      .def("length", &Track::Length, "Track Length.")
      .def("add_element",
           py::overload_cast<image_t, point2D_t>(&Track::AddElement),
           "Add observation (image_id, point2D_idx) to track.")
      .def("delete_element",
           py::overload_cast<image_t, point2D_t>(&Track::DeleteElement),
           "Delete observation (image_id, point2D_idx) from track.")
      .def("append", py::overload_cast<const TrackElement&>(&Track::AddElement))
      .def(
          "add_element",
          py::overload_cast<const image_t, const point2D_t>(&Track::AddElement))
      .def("add_elements", &Track::AddElements, "Add TrackElement list.")
      .def(
          "remove",
          [](Track& self, const size_t idx) {
            THROW_CHECK_LT(idx, self.Elements().size());
            self.DeleteElement(idx);
          },
          "Remove TrackElement at index.")
      .def_property("elements",
                    py::overload_cast<>(&Track::Elements),
                    &Track::SetElements)
      .def("remove",
           py::overload_cast<const image_t, const point2D_t>(
               &Track::DeleteElement),
           "Remove TrackElement with (image_id,point2D_idx).")
      .def("__repr__", [](const Track& self) {
        return "Track(length=" + std::to_string(self.Length()) + ")";
      });
  MakeDataclass(PyTrack);
}

#include "colmap/scene/two_view_geometry.h"

#include "colmap/scene/camera.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/utils.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindTwoViewGeometryScene(py::module& m) {
  py::enum_<TwoViewGeometry::ConfigurationType> PyTwoViewGeometryConfiguration(
      m, "TwoViewGeometryConfiguration");
  PyTwoViewGeometryConfiguration.value("UNDEFINED", TwoViewGeometry::UNDEFINED)
      .value("DEGENERATE", TwoViewGeometry::DEGENERATE)
      .value("CALIBRATED", TwoViewGeometry::CALIBRATED)
      .value("UNCALIBRATED", TwoViewGeometry::UNCALIBRATED)
      .value("PLANAR", TwoViewGeometry::PLANAR)
      .value("PANORAMIC", TwoViewGeometry::PANORAMIC)
      .value("PLANAR_OR_PANORAMIC", TwoViewGeometry::PLANAR_OR_PANORAMIC)
      .value("WATERMARK", TwoViewGeometry::WATERMARK)
      .value("MULTIPLE", TwoViewGeometry::MULTIPLE);
  AddStringToEnumConstructor(PyTwoViewGeometryConfiguration);

  py::class_<TwoViewGeometry> PyTwoViewGeometry(m, "TwoViewGeometry");
  PyTwoViewGeometry.def(py::init<>())
      .def_readwrite("config", &TwoViewGeometry::config)
      .def_readwrite("E", &TwoViewGeometry::E)
      .def_readwrite("F", &TwoViewGeometry::F)
      .def_readwrite("H", &TwoViewGeometry::H)
      .def_readwrite("cam2_from_cam1", &TwoViewGeometry::cam2_from_cam1)
      .def_property(
          "inlier_matches",
          [](const TwoViewGeometry& self) {
            return FeatureMatchesToMatrix(self.inlier_matches);
          },
          [](TwoViewGeometry& self, const PyFeatureMatches& matrix) {
            self.inlier_matches = FeatureMatchesFromMatrix(matrix);
          })
      .def_readwrite("tri_angle", &TwoViewGeometry::tri_angle)
      .def("invert", &TwoViewGeometry::Invert);
  MakeDataclass(PyTwoViewGeometry);
}

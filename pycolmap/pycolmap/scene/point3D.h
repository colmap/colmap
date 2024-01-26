#include "colmap/scene/point3d.h"
#include "colmap/util/misc.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"
#include "pycolmap/log_exceptions.h"
#include "pycolmap/pybind11_extension.h"

#include <memory>
#include <sstream>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
namespace py = pybind11;

using Point3DMap = std::unordered_map<point3D_t, Point3D>;
PYBIND11_MAKE_OPAQUE(Point3DMap);

void BindPoint3D(py::module& m) {
  py::bind_map_fix<Point3DMap>(m, "MapPoint3DIdToPoint3D")
      .def("__repr__", [](const Point3DMap& self) {
        return "MapPoint3DIdToPoint3D(num_points3D=" +
               std::to_string(self.size()) + ")";
      });

  py::class_ext_<Point3D, std::shared_ptr<Point3D>> PyPoint3D(m, "Point3D");
  PyPoint3D.def(py::init<>())
      .def_readwrite("xyz", &Point3D::xyz)
      .def_readwrite("color", &Point3D::color)
      .def_readwrite("error", &Point3D::error)
      .def_readwrite("track", &Point3D::track)
      .def("__repr__", [](const Point3D& self) {
        std::stringstream ss;
        ss << "Point3D(xyz=[" << self.xyz.format(vec_fmt) << "], color=["
           << self.color.format(vec_fmt) << "], error=" << self.error
           << ", track=Track(length=" << self.track.Length() << "))";
        return ss.str();
      });
  MakeDataclass(PyPoint3D);
}

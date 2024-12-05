#include "colmap/scene/point2d.h"

#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/scene/types.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

Point2D MakePoint2D(const Eigen::Vector2d& xy,
                    point3D_t point3D_id = kInvalidPoint3DId,
                    float weight = 1.0f) {
  Point2D point;
  point.xy = xy;
  point.point3D_id = point3D_id;
  point.weight = weight;
  return point;
}

void BindPoint2D(py::module& m) {
  py::classh_ext<Point2D> PyPoint2D(m, "Point2D");
  PyPoint2D.def(py::init<>())
      .def(py::init(&MakePoint2D),
           "xy"_a,
           py::arg_v(
               "point3D_id", kInvalidPoint3DId, "pycolmap.INVALID_POINT3D_ID"),
           py::arg_v("weight", 1.0f, "pycolmap.DEFAULT_POINT2D_WEIGHT"))
      .def_readwrite("xy", &Point2D::xy)
      .def("x", [](const Point2D& self) -> double { return self.xy[0]; })
      .def("y", [](const Point2D& self) -> double { return self.xy[1]; })
      .def_readwrite("point3D_id", &Point2D::point3D_id)
      .def_readwrite("weight", &Point2D::weight)
      .def("has_point3D", &Point2D::HasPoint3D);
  MakeDataclass(PyPoint2D);

  py::bind_vector<Point2DVector>(m, "Point2DList");
}

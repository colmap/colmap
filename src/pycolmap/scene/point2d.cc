#include "colmap/scene/point2d.h"

#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/scene/types.h"

#include <memory>
#include <sstream>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

std::string PrintPoint2D(const Point2D& p2D) {
  std::stringstream ss;
  ss << "Point2D(xy=[" << p2D.xy.format(vec_fmt) << "], point3D_id="
     << (p2D.HasPoint3D() ? std::to_string(p2D.point3D_id) : "Invalid") << ")";
  return ss.str();
}

void BindPoint2D(py::module& m) {
  py::class_ext_<Point2D, std::shared_ptr<Point2D>> PyPoint2D(m, "Point2D");
  PyPoint2D.def(py::init<>())
      .def(py::init<const Eigen::Vector2d&, size_t>(),
           "xy"_a,
           "point3D_id"_a = kInvalidPoint3DId)
      .def_readwrite("xy", &Point2D::xy)
      .def_readwrite("point3D_id", &Point2D::point3D_id)
      .def("has_point3D", &Point2D::HasPoint3D)
      .def("__repr__", &PrintPoint2D);
  MakeDataclass(PyPoint2D);

  py::bind_vector<Point2DVector>(m, "ListPoint2D")
      .def("__repr__", [](const Point2DVector& self) {
        std::string repr = "[";
        bool is_first = true;
        for (auto& p2D : self) {
          if (!is_first) {
            repr += ", ";
          }
          is_first = false;
          repr += PrintPoint2D(p2D);
        }
        repr += "]";
        return repr;
      });
}

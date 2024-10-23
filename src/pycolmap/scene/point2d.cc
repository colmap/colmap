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

void BindPoint2D(py::module& m) {
  py::class_ext_<Point2D, std::shared_ptr<Point2D>> PyPoint2D(m, "Point2D");
  PyPoint2D.def(py::init<>())
      .def(py::init<const Eigen::Vector2d&, size_t>(),
           "xy"_a,
           "point3D_id"_a = kInvalidPoint3DId)
      .def_readwrite("xy", &Point2D::xy)
      .def_readwrite("point3D_id", &Point2D::point3D_id)
      .def("has_point3D", &Point2D::HasPoint3D);
  MakeDataclass(PyPoint2D);

  py::bind_vector<Point2DVector>(m, "ListPoint2D");
}

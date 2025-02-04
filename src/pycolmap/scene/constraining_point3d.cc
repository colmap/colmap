#include "colmap/scene/constraining_point3d.h"

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

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

ConstrainingPoint3D MakeConstrainingPoint3D(const Eigen::Vector3d& xyz) {
  ConstrainingPoint3D point;
  point.xyz = xyz;
  return point;
}

void BindConstrainingPoint3D(py::module& m) {
  py::classh_ext<ConstrainingPoint3D> PyConstrainingPoint3D(
      m, "ConstrainingPoint3D");
  PyConstrainingPoint3D.def(py::init<>())
      .def(py::init(&MakeConstrainingPoint3D), "xyz"_a)
      .def_readwrite("xyz", &ConstrainingPoint3D::xyz);
  MakeDataclass(PyConstrainingPoint3D);

  py::bind_map<std::unordered_map<point3D_t, ConstrainingPoint3D>>(
      m, "MapPoint3DIdToConstrainingPoint3D");
}
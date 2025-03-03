#include "colmap/geometry/triangulation.h"

#include "colmap/util/logging.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindTriangulation(py::module& m) {
  m.def(
      "TriangulatePoint",
      [](const Eigen::Matrix3x4d& cam1_from_world,
         const Eigen::Matrix3x4d& cam2_from_world,
         const Eigen::Vector2d& point1,
         const Eigen::Vector2d& point2)
          -> py::typing::Optional<Eigen::Vector3d> {
        Eigen::Vector3d xyz;
        if (TriangulatePoint(
                cam1_from_world, cam2_from_world, point1, point2, &xyz)) {
          return py::cast(xyz);
        } else {
          return py::none();
        }
      },
      "cam1_from_world"_a,
      "cam2_from_world"_a,
      "point1"_a,
      "point2"_a,
      "Triangulate point from two-view observation.");
  m.def("CalculateTriangulationAngle",
        &CalculateTriangulationAngle,
        "proj_center1"_a,
        "proj_center2"_a,
        "point3D"_a,
        "Calculate triangulation angle in radians.");
}

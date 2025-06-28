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
      "triangulate_point",
      [](const Eigen::Matrix3x4d& cam1_from_world,
         const Eigen::Matrix3x4d& cam2_from_world,
         const Eigen::Vector2d& cam_point1,
         const Eigen::Vector2d& cam_point2)
          -> py::typing::Optional<Eigen::Vector3d> {
        Eigen::Vector3d point3D;
        if (TriangulatePoint(cam1_from_world,
                             cam2_from_world,
                             cam_point1,
                             cam_point2,
                             &point3D)) {
          return py::cast(point3D);
        } else {
          return py::none();
        }
      },
      "cam1_from_world"_a,
      "cam2_from_world"_a,
      "cam_point1"_a,
      "cam_point2"_a,
      "Triangulate point in world from two-view observation.");
  m.def("calculate_triangulation_angle",
        &CalculateTriangulationAngle,
        "proj_center1"_a,
        "proj_center2"_a,
        "point3D"_a,
        "Calculate triangulation angle in radians.");
  DefDeprecation(m, "TriangulatePoint", "triangulate_point");
  DefDeprecation(
      m, "CalculateTriangulationAngle", "calculate_triangulation_angle");

  m.def(
      "triangulate_mid_point",
      [](const Rigid3d& cam2_from_cam1,
         const Eigen::Vector3d& cam_ray1,
         const Eigen::Vector3d& cam_ray2)
          -> py::typing::Optional<Eigen::Vector3d> {
        Eigen::Vector3d point3D_in_cam1;
        if (TriangulateMidPoint(
                cam2_from_cam1, cam_ray1, cam_ray2, &point3D_in_cam1)) {
          return py::cast(point3D_in_cam1);
        } else {
          return py::none();
        }
      },
      "cam2_from_cam1"_a,
      "cam_ray1"_a,
      "cam_ray2"_a,
      "Triangulate mid-point in first camera from two-view observation.");
}

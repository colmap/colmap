#pragma once

#include "colmap/geometry/homography_matrix.h"
#include "colmap/util/logging.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

py::dict PyPoseFromHomographyMatrix(
    const Eigen::Matrix3d& H,
    const Eigen::Matrix3d& K1,
    const Eigen::Matrix3d& K2,
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2) {
  py::gil_scoped_release release;
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  Eigen::Vector3d n;
  std::vector<Eigen::Vector3d> points3D;
  PoseFromHomographyMatrix(H, K1, K2, points1, points2, &R, &t, &n, &points3D);
  py::gil_scoped_acquire acquire;
  return py::dict("R"_a = R, "t"_a = t, "n"_a = n, "points3D"_a = points3D);
}

void BindHomographyGeometry(py::module& m) {
  m.def("homography_decomposition",
        &PyPoseFromHomographyMatrix,
        "H"_a,
        "K1"_a,
        "K2"_a,
        "points1"_a,
        "points2"_a,
        "Analytical Homography Decomposition.");
}

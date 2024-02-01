#pragma once

#include "colmap/estimators/triangulation.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/image.h"
#include "colmap/util/logging.h"

#include "pycolmap/helpers.h"
#include "pycolmap/utils.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

py::dict PyEstimateTriangulation(
    const std::vector<TriangulationEstimator::PointData>& point_data,
    const std::vector<Image>& images,
    const std::vector<Camera>& cameras,
    const EstimateTriangulationOptions& options) {
  THROW_CHECK_EQ(images.size(), cameras.size());
  THROW_CHECK_EQ(images.size(), point_data.size());
  py::object failure = py::none();
  py::gil_scoped_release release;

  std::vector<TriangulationEstimator::PoseData> pose_data;
  pose_data.reserve(images.size());
  for (size_t i = 0; i < images.size(); ++i) {
    pose_data.emplace_back(images[i].CamFromWorld().ToMatrix(),
                           images[i].ProjectionCenter(),
                           &cameras[i]);
  }
  Eigen::Vector3d xyz;
  std::vector<char> inlier_mask;
  if (!EstimateTriangulation(
          options, point_data, pose_data, &inlier_mask, &xyz)) {
    return failure;
  }

  py::gil_scoped_acquire acquire;
  return py::dict("xyz"_a = xyz, "inliers"_a = ToPythonMask(inlier_mask));
}

void BindTriangulationEstimator(py::module& m) {
  auto PyRANSACOptions = m.attr("RANSACOptions");

  py::class_<TriangulationEstimator::PointData>(m, "PointData")
      .def(py::init<const Eigen::Vector2d&, const Eigen::Vector2d&>());

  py::class_<EstimateTriangulationOptions> PyTriangulationOptions(
      m, "EstimateTriangulationOptions");
  PyTriangulationOptions
      .def(py::init<>([PyRANSACOptions]() {
        EstimateTriangulationOptions options;
        // init through Python to obtain the new defaults defined in  __init__
        options.ransac_options = PyRANSACOptions().cast<RANSACOptions>();
        return options;
      }))
      .def_readwrite("min_tri_angle",
                     &EstimateTriangulationOptions::min_tri_angle)
      .def_readwrite("ransac", &EstimateTriangulationOptions::ransac_options);
  MakeDataclass(PyTriangulationOptions);
  auto triangulation_options =
      PyTriangulationOptions().cast<EstimateTriangulationOptions>();

  m.def("estimate_triangulation",
        &PyEstimateTriangulation,
        "point_data"_a,
        "images"_a,
        "cameras"_a,
        "opions"_a = triangulation_options,
        "Robustly estimate 3D point from observations in multiple views using "
        "RANSAC");
}

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

py::object PyEstimateTriangulation(
    const std::vector<TriangulationEstimator::PointData>& point_data,
    const std::vector<Image>& images,
    const std::vector<Camera>& cameras,
    const EstimateTriangulationOptions& options) {
  py::gil_scoped_release release;
  THROW_CHECK_EQ(images.size(), cameras.size());
  THROW_CHECK_EQ(images.size(), point_data.size());

  std::vector<TriangulationEstimator::PoseData> pose_data;
  pose_data.reserve(images.size());
  for (size_t i = 0; i < images.size(); ++i) {
    pose_data.emplace_back(images[i].CamFromWorld().ToMatrix(),
                           images[i].ProjectionCenter(),
                           &cameras[i]);
  }
  Eigen::Vector3d xyz;
  std::vector<char> inlier_mask;
  const bool success =
      EstimateTriangulation(options, point_data, pose_data, &inlier_mask, &xyz);

  py::gil_scoped_acquire acquire;
  if (success) {
    return py::dict("xyz"_a = xyz, "inliers"_a = ToPythonMask(inlier_mask));
  } else {
    return py::none();
  }
}

void BindTriangulationEstimator(py::module& m) {
  auto PyRANSACOptions = m.attr("RANSACOptions");

  py::class_<TriangulationEstimator::PointData>(m, "PointData")
      .def(py::init<const Eigen::Vector2d&, const Eigen::Vector2d&>());

  using ResType = TriangulationEstimator::ResidualType;
  auto PyResType =
      py::enum_<ResType>(m, "TriangulationResidualType")
          .value("ANGULAR_ERROR", ResType::ANGULAR_ERROR)
          .value("REPROJECTION_ERROR", ResType::REPROJECTION_ERROR);
  AddStringToEnumConstructor(PyResType);

  using Options = EstimateTriangulationOptions;
  py::class_<Options> PyTriangulationOptions(m, "EstimateTriangulationOptions");
  PyTriangulationOptions
      .def(py::init<>([PyRANSACOptions]() {
        Options options;
        // init through Python to obtain the new defaults defined in  __init__
        options.ransac_options = PyRANSACOptions().cast<RANSACOptions>();
        return options;
      }))
      .def_readwrite("min_tri_angle",
                     &Options::min_tri_angle,
                     "Minimum triangulation angle in radians.")
      .def_readwrite(
          "residual_type", &Options::residual_type, "Employed residual type.")
      .def_readwrite("ransac", &Options::ransac_options, "RANSAC options.");
  MakeDataclass(PyTriangulationOptions);
  auto triangulation_options = PyTriangulationOptions().cast<Options>();

  m.def("estimate_triangulation",
        &PyEstimateTriangulation,
        "point_data"_a,
        "images"_a,
        "cameras"_a,
        "options"_a = triangulation_options,
        "Robustly estimate 3D point from observations in multiple views using "
        "RANSAC");
}

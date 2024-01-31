#pragma once

#include "colmap/estimators/two_view_geometry.h"
#include "colmap/estimators/utils.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/logging.h"

#include "pycolmap/helpers.h"
#include "pycolmap/utils.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

// TODO(sarlinpe): Consider changing the COLMAP type.
typedef Eigen::Matrix<uint32_t, Eigen::Dynamic, 2, Eigen::RowMajor>
    PyFeatureMatches;

PyFeatureMatches FeatureMatchesToMatrix(const FeatureMatches& matches) {
  PyFeatureMatches matrix(matches.size(), 2);
  for (size_t i = 0; i < matches.size(); i++) {
    matrix(i, 0) = matches[i].point2D_idx1;
    matrix(i, 1) = matches[i].point2D_idx2;
  }
  return matrix;
}

FeatureMatches FeatureMatchesFromMatrix(const PyFeatureMatches& matrix) {
  FeatureMatches matches(matrix.rows());
  for (size_t i = 0; i < matches.size(); i++) {
    matches[i].point2D_idx1 = matrix(i, 0);
    matches[i].point2D_idx2 = matrix(i, 1);
  }
  return matches;
}

void BindTwoViewGeometryEstimator(py::module& m) {
  py::class_<TwoViewGeometryOptions> PyTwoViewGeometryOptions(
      m, "TwoViewGeometryOptions");
  PyTwoViewGeometryOptions.def(py::init<>())
      .def_readwrite("min_num_inliers",
                     &TwoViewGeometryOptions::min_num_inliers)
      .def_readwrite("min_E_F_inlier_ratio",
                     &TwoViewGeometryOptions::min_E_F_inlier_ratio)
      .def_readwrite("max_H_inlier_ratio",
                     &TwoViewGeometryOptions::max_H_inlier_ratio)
      .def_readwrite("watermark_min_inlier_ratio",
                     &TwoViewGeometryOptions::watermark_min_inlier_ratio)
      .def_readwrite("watermark_border_size",
                     &TwoViewGeometryOptions::watermark_border_size)
      .def_readwrite("detect_watermark",
                     &TwoViewGeometryOptions::detect_watermark)
      .def_readwrite("multiple_ignore_watermark",
                     &TwoViewGeometryOptions::multiple_ignore_watermark)
      .def_readwrite("force_H_use", &TwoViewGeometryOptions::force_H_use)
      .def_readwrite("compute_relative_pose",
                     &TwoViewGeometryOptions::compute_relative_pose)
      .def_readwrite("multiple_models",
                     &TwoViewGeometryOptions::multiple_models)
      .def_readwrite("ransac", &TwoViewGeometryOptions::ransac_options);
  MakeDataclass(PyTwoViewGeometryOptions);
  auto tvg_options = PyTwoViewGeometryOptions().cast<TwoViewGeometryOptions>();

  py::enum_<TwoViewGeometry::ConfigurationType>(m,
                                                "TwoViewGeometryConfiguration")
      .value("UNDEFINED", TwoViewGeometry::UNDEFINED)
      .value("DEGENERATE", TwoViewGeometry::DEGENERATE)
      .value("CALIBRATED", TwoViewGeometry::CALIBRATED)
      .value("UNCALIBRATED", TwoViewGeometry::UNCALIBRATED)
      .value("PLANAR", TwoViewGeometry::PLANAR)
      .value("PANORAMIC", TwoViewGeometry::PANORAMIC)
      .value("PLANAR_OR_PANORAMIC", TwoViewGeometry::PLANAR_OR_PANORAMIC)
      .value("WATERMARK", TwoViewGeometry::WATERMARK)
      .value("MULTIPLE", TwoViewGeometry::MULTIPLE);

  py::class_<TwoViewGeometry> PyTwoViewGeometry(m, "TwoViewGeometry");
  PyTwoViewGeometry.def(py::init<>())
      .def_readonly("config", &TwoViewGeometry::config)
      .def_readonly("E", &TwoViewGeometry::E)
      .def_readonly("F", &TwoViewGeometry::F)
      .def_readonly("H", &TwoViewGeometry::H)
      .def_readonly("cam2_from_cam1", &TwoViewGeometry::cam2_from_cam1)
      .def_property_readonly(
          "inlier_matches",
          [](const TwoViewGeometry& self) {
            return FeatureMatchesToMatrix(self.inlier_matches);
          })
      .def_readonly("tri_angle", &TwoViewGeometry::tri_angle)
      .def("invert", &TwoViewGeometry::Invert);
  MakeDataclass(PyTwoViewGeometry);

  m.def(
      "estimate_calibrated_two_view_geometry",
      [](const Camera& camera1,
         const std::vector<Eigen::Vector2d>& points1,
         const Camera& camera2,
         const std::vector<Eigen::Vector2d>& points2,
         const PyFeatureMatches* matches_ptr,
         const TwoViewGeometryOptions& options) {
        py::gil_scoped_release release;
        FeatureMatches matches;
        if (matches_ptr != nullptr) {
          matches = FeatureMatchesFromMatrix(*matches_ptr);
        } else {
          THROW_CHECK_EQ(points1.size(), points2.size());
          matches.reserve(points1.size());
          for (size_t i = 0; i < points1.size(); i++) {
            matches.emplace_back(i, i);
          }
        }
        return EstimateCalibratedTwoViewGeometry(
            camera1, points1, camera2, points2, matches, options);
      },
      "camera1"_a,
      "points1"_a,
      "camera2"_a,
      "points2"_a,
      "matches"_a = py::none(),
      "options"_a = tvg_options);

  m.def(
      "estimate_two_view_geometry",
      [](const Camera& camera1,
         const std::vector<Eigen::Vector2d>& points1,
         const Camera& camera2,
         const std::vector<Eigen::Vector2d>& points2,
         const PyFeatureMatches* matches_ptr,
         const TwoViewGeometryOptions& options) {
        py::gil_scoped_release release;
        FeatureMatches matches;
        if (matches_ptr != nullptr) {
          matches = FeatureMatchesFromMatrix(*matches_ptr);
        } else {
          THROW_CHECK_EQ(points1.size(), points2.size());
          matches.reserve(points1.size());
          for (size_t i = 0; i < points1.size(); i++) {
            matches.emplace_back(i, i);
          }
        }
        return EstimateTwoViewGeometry(
            camera1, points1, camera2, points2, matches, options);
      },
      "camera1"_a,
      "points1"_a,
      "camera2"_a,
      "points2"_a,
      "matches"_a = py::none(),
      "options"_a = tvg_options);

  m.def("estimate_two_view_geometry_pose",
        &EstimateTwoViewGeometryPose,
        "camera1"_a,
        "points1"_a,
        "camera2"_a,
        "points2"_a,
        "geometry"_a);

  m.def(
      "squared_sampson_error",
      [](const std::vector<Eigen::Vector2d>& points1,
         const std::vector<Eigen::Vector2d>& points2,
         const Eigen::Matrix3d& E) {
        std::vector<double> residuals;
        ComputeSquaredSampsonError(points1, points2, E, &residuals);
        return residuals;
      },
      "points2D1"_a,
      "points2D2"_a,
      "E"_a,
      "Calculate the squared Sampson error for a given essential or "
      "fundamental matrix.",
      py::call_guard<py::gil_scoped_release>());
}

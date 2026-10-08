// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/two_view_geometry.h"

#include "colmap/feature/types.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/homography_matrix.h"
#include "colmap/geometry/normalization.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/logging.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/utils.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindTwoViewGeometryEstimator(py::module& m) {
  py::classh<TwoViewGeometryOptions> PyTwoViewGeometryOptions(
      m, "TwoViewGeometryOptions");
  PyTwoViewGeometryOptions.def(py::init<>())
      .def_readwrite("min_num_inliers",
                     &TwoViewGeometryOptions::min_num_inliers)
      .def_readwrite("min_inlier_ratio",
                     &TwoViewGeometryOptions::min_inlier_ratio)
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
      .def_readwrite("watermark_detection_max_error",
                     &TwoViewGeometryOptions::watermark_detection_max_error)
      .def_readwrite("filter_stationary_matches",
                     &TwoViewGeometryOptions::filter_stationary_matches)
      .def_readwrite("stationary_matches_max_error",
                     &TwoViewGeometryOptions::stationary_matches_max_error)
      .def_readwrite("force_H_use", &TwoViewGeometryOptions::force_H_use)
      .def_readwrite("use_degensac", &TwoViewGeometryOptions::use_degensac)
      .def_readwrite("use_sampson_refinement",
                     &TwoViewGeometryOptions::use_sampson_refinement)
      .def_readwrite("compute_relative_pose",
                     &TwoViewGeometryOptions::compute_relative_pose)
      .def_readwrite("multiple_models",
                     &TwoViewGeometryOptions::multiple_models)
      .def_readwrite("ransac", &TwoViewGeometryOptions::ransac_options);
  MakeDataclass(PyTwoViewGeometryOptions);

  m.def(
      "estimate_calibrated_two_view_geometry",
      [](const Camera& camera1,
         const std::vector<Eigen::Vector2d>& points1,
         const Camera& camera2,
         const std::vector<Eigen::Vector2d>& points2,
         const FeatureMatchesMatrix* matches_ptr,
         const TwoViewGeometryOptions& options) {
        py::gil_scoped_release release;
        FeatureMatches matches;
        if (matches_ptr != nullptr) {
          matches = MatchesFromMatrix(*matches_ptr);
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
      py::arg_v(
          "options", TwoViewGeometryOptions(), "TwoViewGeometryOptions()"));

  m.def(
      "estimate_two_view_geometry",
      [](const Camera& camera1,
         const std::vector<Eigen::Vector2d>& points1,
         const Camera& camera2,
         const std::vector<Eigen::Vector2d>& points2,
         const FeatureMatchesMatrix* matches_ptr,
         const TwoViewGeometryOptions& options) {
        py::gil_scoped_release release;
        FeatureMatches matches;
        if (matches_ptr != nullptr) {
          matches = MatchesFromMatrix(*matches_ptr);
        } else {
          THROW_CHECK_EQ(points1.size(), points2.size());
          matches.reserve(points1.size());
          for (size_t i = 0; i < points1.size(); i++) {
            matches.emplace_back(i, i);
          }
        }
        return EstimateTwoViewGeometry(
            camera1, points1, camera2, points2, std::move(matches), options);
      },
      "camera1"_a,
      "points1"_a,
      "camera2"_a,
      "points2"_a,
      "matches"_a = py::none(),
      py::arg_v(
          "options", TwoViewGeometryOptions(), "TwoViewGeometryOptions()"));

  m.def("estimate_two_view_geometry_pose",
        &EstimateTwoViewGeometryPose,
        "camera1"_a,
        "points1"_a,
        "camera2"_a,
        "points2"_a,
        "geometry"_a);

  m.def(
      "compute_squared_sampson_error",
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

  auto PyTwoViewPoseCovarianceOptions =
      py::classh<TwoViewPoseCovarianceOptions>(m,
                                               "TwoViewPoseCovarianceOptions")
          .def(py::init<>())
          .def_readwrite("num_threads",
                         &TwoViewPoseCovarianceOptions::num_threads)
          .def_readwrite("min_sigma_obs_px",
                         &TwoViewPoseCovarianceOptions::min_sigma_obs_px)
          .def_readwrite("min_rotation_eigenvalue",
                         &TwoViewPoseCovarianceOptions::min_rotation_eigenvalue)
          .def_readwrite(
              "min_translation_eigenvalue",
              &TwoViewPoseCovarianceOptions::min_translation_eigenvalue)
          .def_readwrite(
              "min_translation_rel_eigenvalue",
              &TwoViewPoseCovarianceOptions::min_translation_rel_eigenvalue)
          .def_readwrite("max_rotation_cov_cond",
                         &TwoViewPoseCovarianceOptions::max_rotation_cov_cond)
          .def_readwrite(
              "rotation_sigma_floor_rad",
              &TwoViewPoseCovarianceOptions::rotation_sigma_floor_rad)
          .def_readwrite(
              "fallback_rotation_sigma_rad",
              &TwoViewPoseCovarianceOptions::fallback_rotation_sigma_rad);
  MakeDataclass(PyTwoViewPoseCovarianceOptions);

  py::classh<TwoViewPoseCovariance>(m, "TwoViewPoseCovariance")
      .def(py::init<>())
      .def_readwrite("cov_rot", &TwoViewPoseCovariance::cov_rot)
      .def_readwrite("cov_trans_tangent",
                     &TwoViewPoseCovariance::cov_trans_tangent)
      .def_readwrite("sigma_obs_px", &TwoViewPoseCovariance::sigma_obs_px)
      .def_readwrite("num_inliers", &TwoViewPoseCovariance::num_inliers)
      .def_readwrite("is_rotation_degenerate",
                     &TwoViewPoseCovariance::is_rotation_degenerate)
      .def_readwrite("is_translation_degenerate",
                     &TwoViewPoseCovariance::is_translation_degenerate);

  m.def("estimate_two_view_pose_covariance",
        &EstimateTwoViewPoseCovariance,
        "camera1"_a,
        "points1"_a,
        "camera2"_a,
        "points2"_a,
        "geometry"_a,
        py::arg_v("options",
                  TwoViewPoseCovarianceOptions(),
                  "TwoViewPoseCovarianceOptions()"),
        "Estimate the covariance of the relative pose of a two-view geometry "
        "from its inlier correspondences.",
        py::call_guard<py::gil_scoped_release>());

  m.def("estimate_pose_graph_covariances",
        &EstimatePoseGraphCovariances,
        "database_cache"_a,
        "pose_graph"_a,
        py::arg_v("options",
                  TwoViewPoseCovarianceOptions(),
                  "TwoViewPoseCovarianceOptions()"),
        "Populate the rotation covariance of all valid pose graph edges that "
        "do not have one yet, from their inlier correspondences.",
        py::call_guard<py::gil_scoped_release>());
}

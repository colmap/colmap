// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/two_view_pose_covariance.h"

#include "colmap/scene/camera.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/two_view_geometry.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindTwoViewPoseCovarianceEstimator(py::module& m) {
  auto PyTwoViewPoseCovarianceOptions =
      py::classh<TwoViewPoseCovarianceOptions>(m,
                                               "TwoViewPoseCovarianceOptions")
          .def(py::init<>())
          .def_readwrite("num_threads",
                         &TwoViewPoseCovarianceOptions::num_threads)
          .def_readwrite("min_sigma_obs_px",
                         &TwoViewPoseCovarianceOptions::min_sigma_obs_px)
          .def_readwrite(
              "max_rotation_sigma_deg",
              &TwoViewPoseCovarianceOptions::max_rotation_sigma_deg,
              "Maximum standard deviation (degrees) of the relative rotation "
              "along its least constrained axis, beyond which no rotation "
              "covariance is returned.");
  MakeDataclass(PyTwoViewPoseCovarianceOptions);

  py::classh<TwoViewPoseCovariance>(m, "TwoViewPoseCovariance")
      .def(py::init<>())
      .def_readwrite("cov_rot", &TwoViewPoseCovariance::cov_rot)
      .def_readwrite("cov_trans_tangent",
                     &TwoViewPoseCovariance::cov_trans_tangent)
      .def_readwrite("sigma_obs_px", &TwoViewPoseCovariance::sigma_obs_px)
      .def_readwrite("num_inliers", &TwoViewPoseCovariance::num_inliers);

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

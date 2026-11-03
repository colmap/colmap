// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/two_view_rotation_covariance.h"

#include "colmap/scene/camera.h"
#include "colmap/scene/two_view_geometry.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindTwoViewRotationCovarianceEstimator(py::module& m) {
  auto PyTwoViewRotationCovarianceOptions =
      py::classh<TwoViewRotationCovarianceOptions>(
          m, "TwoViewRotationCovarianceOptions")
          .def(py::init<>())
          .def_readwrite("point2D_stddev_px",
                         &TwoViewRotationCovarianceOptions::point2D_stddev_px,
                         "Standard deviation of the 2D image point "
                         "observations in pixels.")
          .def_readwrite(
              "max_rotation_sigma_deg",
              &TwoViewRotationCovarianceOptions::max_rotation_sigma_deg,
              "Maximum standard deviation (degrees) of the relative rotation "
              "along any axis. Poorly or not constrained axes are capped to "
              "it.")
          .def("check", &TwoViewRotationCovarianceOptions::Check);
  MakeDataclass(PyTwoViewRotationCovarianceOptions);

  m.def(
      "estimate_two_view_rotation_covariance",
      &EstimateTwoViewRotationCovariance,
      "camera1"_a,
      "points1"_a,
      "camera2"_a,
      "points2"_a,
      "geometry"_a,
      py::arg_v("options",
                TwoViewRotationCovarianceOptions(),
                "TwoViewRotationCovarianceOptions()"),
      "Estimate the covariance of the relative rotation of a two-view geometry "
      "from its inlier correspondences.",
      py::call_guard<py::gil_scoped_release>());
}

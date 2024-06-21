#include "colmap/estimators/covariance.h"

#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindCovarianceEstimator(py::module& m) {
  m.def("get_covariance_for_pose_inverse",
        &GetCovarianceForPoseInverse,
        py::arg("covar"),
        py::arg("rigid3"));

  m.def(
      "estimate_pose_covariance_from_ba_ceres_backend",
      &BundleAdjustmentCovarianceEstimator::EstimatePoseCovarianceCeresBackend,
      py::arg("problem"),
      py::arg("reconstruction"));

  m.def("estimate_pose_covariance_from_ba",
        &BundleAdjustmentCovarianceEstimator::EstimatePoseCovariance,
        py::arg("problem"),
        py::arg("reconstruction"));
}

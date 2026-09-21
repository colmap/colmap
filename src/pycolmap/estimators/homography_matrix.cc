// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/solvers/homography_matrix.h"

#include "colmap/math/random.h"
#include "colmap/optim/loransac.h"
#include "colmap/optim/support_measurement.h"
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

py::typing::Optional<py::dict> PyEstimateHomographyMatrix(
    const std::vector<Eigen::Vector2d>& points2D1,
    const std::vector<Eigen::Vector2d>& points2D2,
    const RANSACOptions& options,
    bool cheirality_check) {
  py::gil_scoped_release release;
  THROW_CHECK_EQ(points2D1.size(), points2D2.size());
  const auto report_to_dict =
      [](const auto& report) -> py::typing::Optional<py::dict> {
    py::gil_scoped_acquire acquire;
    if (!report.success) {
      return py::none();
    }
    return py::dict("H"_a = report.model,
                    "num_inliers"_a = report.support.num_inliers,
                    "inlier_mask"_a = ToPythonMask(report.inlier_mask));
  };
  if (cheirality_check) {
    LORANSAC<HomographyMatrixCheiralityEstimator,
             HomographyMatrixEstimator,
             MEstimatorSupportMeasurer>
        ransac(options);
    return report_to_dict(ransac.Estimate(points2D1, points2D2));
  }
  LORANSAC<HomographyMatrixEstimator,
           HomographyMatrixEstimator,
           MEstimatorSupportMeasurer>
      ransac(options);
  return report_to_dict(ransac.Estimate(points2D1, points2D2));
}

void BindHomographyMatrixEstimator(py::module& m) {
  auto est_options = m.attr("RANSACOptions")().cast<RANSACOptions>();

  m.def("estimate_homography_matrix",
        &PyEstimateHomographyMatrix,
        "points2D1"_a,
        "points2D2"_a,
        py::arg_v("estimation_options", est_options, "RANSACOptions()"),
        "cheirality_check"_a = false,
        "Robustly estimate homography matrix using LO-RANSAC. If "
        "cheirality_check is true, minimal samples that flip orientation are "
        "rejected without solving, which assumes an orientation-preserving "
        "homography.");
}

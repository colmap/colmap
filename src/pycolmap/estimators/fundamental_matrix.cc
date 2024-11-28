#include "colmap/estimators/fundamental_matrix.h"

#include "colmap/math/random.h"
#include "colmap/optim/loransac.h"
#include "colmap/scene/camera.h"
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

py::typing::Optional<py::dict> PyEstimateFundamentalMatrix(
    const std::vector<Eigen::Vector2d>& points2D1,
    const std::vector<Eigen::Vector2d>& points2D2,
    const RANSACOptions& options) {
  py::gil_scoped_release release;
  THROW_CHECK_EQ(points2D1.size(), points2D2.size());
  LORANSAC<FundamentalMatrixSevenPointEstimator,
           FundamentalMatrixEightPointEstimator>
      ransac(options);
  const auto report = ransac.Estimate(points2D1, points2D2);
  py::gil_scoped_acquire acquire;
  if (!report.success) {
    return py::none();
  }

  return py::dict("F"_a = report.model,
                  "num_inliers"_a = report.support.num_inliers,
                  "inlier_mask"_a = ToPythonMask(report.inlier_mask));
}

void BindFundamentalMatrixEstimator(py::module& m) {
  auto ransac_options = m.attr("RANSACOptions")().cast<RANSACOptions>();

  m.def("estimate_fundamental_matrix",
        &PyEstimateFundamentalMatrix,
        "points2D1"_a,
        "points2D2"_a,
        py::arg_v("estimation_options", ransac_options, "RANSACOptions()"),
        "Robustly estimate fundamental matrix with LO-RANSAC.");
  DefDeprecation(
      m, "fundamental_matrix_estimation", "estimate_fundamental_matrix");
}

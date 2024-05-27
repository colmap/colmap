#include "colmap/estimators/homography_matrix.h"

#include "colmap/math/random.h"
#include "colmap/optim/loransac.h"
#include "colmap/util/logging.h"

#include "pycolmap/pybind11_extension.h"
#include "pycolmap/utils.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

py::object PyEstimateHomographyMatrix(
    const std::vector<Eigen::Vector2d>& points2D1,
    const std::vector<Eigen::Vector2d>& points2D2,
    const RANSACOptions& options) {
  py::gil_scoped_release release;
  THROW_CHECK_EQ(points2D1.size(), points2D2.size());
  LORANSAC<HomographyMatrixEstimator, HomographyMatrixEstimator> H_ransac(
      options);
  const auto report = H_ransac.Estimate(points2D1, points2D2);
  py::gil_scoped_acquire acquire;
  if (!report.success) {
    return py::none();
  }
  const Eigen::Matrix3d H = report.model;
  return py::dict("H"_a = H,
                  "num_inliers"_a = report.support.num_inliers,
                  "inliers"_a = ToPythonMask(report.inlier_mask));
}

void BindHomographyMatrixEstimator(py::module& m) {
  auto est_options = m.attr("RANSACOptions")().cast<RANSACOptions>();

  m.def("homography_matrix_estimation",
        &PyEstimateHomographyMatrix,
        "points2D1"_a,
        "points2D2"_a,
        "estimation_options"_a = est_options,
        "LORANSAC + 4-point DLT algorithm.");
}

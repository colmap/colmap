#include "colmap/estimators/affine_transform.h"

#include "colmap/math/random.h"
#include "colmap/optim/loransac.h"
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

void BindAffineTransformEstimator(py::module& m) {
  auto ransac_options = m.attr("RANSACOptions")().cast<RANSACOptions>();

  m.def(
      "estimate_affine2d",
      [](const std::vector<Eigen::Vector2d>& src,
         const std::vector<Eigen::Vector2d>& tgt)
          -> py::typing::Optional<Eigen::Matrix2x3d> {
        py::gil_scoped_release release;
        Eigen::Matrix2x3d tgt_from_src;
        const bool success = EstimateAffine2d(src, tgt, tgt_from_src);
        py::gil_scoped_acquire acquire;
        if (success) {
          return py::cast(tgt_from_src);
        } else {
          return py::none();
        }
      },
      "src"_a,
      "tgt"_a,
      "Estimate the 2D affine transform tgt_from_src.");

  m.def(
      "estimate_affine2d_robust",
      [](const std::vector<Eigen::Vector2d>& src,
         const std::vector<Eigen::Vector2d>& tgt,
         const RANSACOptions& options) -> py::typing::Optional<py::dict> {
        py::gil_scoped_release release;
        Eigen::Matrix2x3d tgt_from_src;
        const auto report =
            EstimateAffine2dRobust(src, tgt, options, tgt_from_src);
        py::gil_scoped_acquire acquire;
        if (!report.success) {
          return py::none();
        }
        return py::dict("tgt_from_src"_a = tgt_from_src,
                        "num_inliers"_a = report.support.num_inliers,
                        "inlier_mask"_a = ToPythonMask(report.inlier_mask));
      },
      "src"_a,
      "tgt"_a,
      py::arg_v("estimation_options", ransac_options, "RANSACOptions()"),
      "Robustly estimate the 2D affine transform tgt_from_src using "
      "LO-RANSAC.");
}

#include "colmap/estimators/similarity_transform.h"

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

void BindSimilarityTransformEstimator(py::module& m) {
  auto est_options = m.attr("RANSACOptions")().cast<RANSACOptions>();

  m.def(
      "estimate_sim3d",
      [](const std::vector<Eigen::Vector3d>& src,
         const std::vector<Eigen::Vector3d>& tgt)
          -> py::typing::Optional<Sim3d> {
        py::gil_scoped_release release;
        Sim3d tgt_from_src;
        const bool success = EstimateSim3d(src, tgt, tgt_from_src);
        py::gil_scoped_acquire acquire;
        if (success) {
          return py::cast(tgt_from_src);
        } else {
          return py::none();
        }
      },
      "points3D_src"_a,
      "points3D_tgt"_a,
      "Estimate the 3D similarity transform tgt_T_src.");

  m.def(
      "estimate_sim3d_robust",
      [](const std::vector<Eigen::Vector3d>& src,
         const std::vector<Eigen::Vector3d>& tgt,
         const RANSACOptions& options) -> py::typing::Optional<Sim3d> {
        py::gil_scoped_release release;
        Sim3d tgt_from_src;
        const auto report =
            EstimateSim3dRobust(src, tgt, options, tgt_from_src);
        py::gil_scoped_acquire acquire;
        if (!report.success) {
          return py::none();
        }
        return py::dict("tgt_from_src"_a = Sim3d::FromMatrix(report.model),
                        "num_inliers"_a = report.support.num_inliers,
                        "inliers"_a = ToPythonMask(report.inlier_mask));
      },
      "points3D_src"_a,
      "points3D_tgt"_a,
      py::arg_v("estimation_options", est_options, "RANSACOptions()"),
      "Estimate the 3D similarity transform in a robust way with LORANSAC.");
}

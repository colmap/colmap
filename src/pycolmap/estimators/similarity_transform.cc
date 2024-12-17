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
  auto ransac_options = m.attr("RANSACOptions")().cast<RANSACOptions>();

  m.def(
      "estimate_rigid3d",
      [](const std::vector<Eigen::Vector3d>& src,
         const std::vector<Eigen::Vector3d>& tgt)
          -> py::typing::Optional<Rigid3d> {
        py::gil_scoped_release release;
        Rigid3d tgt_from_src;
        const bool success = EstimateRigid3d(src, tgt, tgt_from_src);
        py::gil_scoped_acquire acquire;
        if (success) {
          return py::cast(tgt_from_src);
        } else {
          return py::none();
        }
      },
      "src"_a,
      "tgt"_a,
      "Estimate the 3D rigid transform tgt_from_src.");

  m.def(
      "estimate_rigid3d_robust",
      [](const std::vector<Eigen::Vector3d>& src,
         const std::vector<Eigen::Vector3d>& tgt,
         const RANSACOptions& options) -> py::typing::Optional<Rigid3d> {
        py::gil_scoped_release release;
        Rigid3d tgt_from_src;
        const auto report =
            EstimateRigid3dRobust(src, tgt, options, tgt_from_src);
        py::gil_scoped_acquire acquire;
        if (!report.success) {
          return py::none();
        }
        return py::dict("tgt_from_src"_a = Rigid3d::FromMatrix(report.model),
                        "num_inliers"_a = report.support.num_inliers,
                        "inlier_mask"_a = ToPythonMask(report.inlier_mask));
      },
      "src"_a,
      "tgt"_a,
      py::arg_v("estimation_options", ransac_options, "RANSACOptions()"),
      "Robustly estimate the 3D rigid transform tgt_from_src using LO-RANSAC.");

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
      "src"_a,
      "tgt"_a,
      "Estimate the 3D similarity transform tgt_from_src.");

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
                        "inlier_mask"_a = ToPythonMask(report.inlier_mask));
      },
      "src"_a,
      "tgt"_a,
      py::arg_v("estimation_options", ransac_options, "RANSACOptions()"),
      "Robustly estimate the 3D similarity transform tgt_from_src using "
      "LO-RANSAC.");
}

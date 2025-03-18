#include "colmap/estimators/generalized_pose.h"
#include "colmap/estimators/pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/random.h"
#include "colmap/optim/ransac.h"
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

py::typing::Optional<py::dict> PyEstimateAndRefineGeneralizedAbsolutePose(
    const std::vector<Eigen::Vector2d>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const std::vector<size_t>& camera_idxs,
    const std::vector<Rigid3d>& cams_from_rig,
    std::vector<Camera>& cameras,
    const RANSACOptions& ransac_options,
    const AbsolutePoseRefinementOptions& refinement_options,
    const bool return_covariance) {
  py::gil_scoped_release release;
  Rigid3d rig_from_world;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  if (!EstimateGeneralizedAbsolutePose(ransac_options,
                                       points2D,
                                       points3D,
                                       camera_idxs,
                                       cams_from_rig,
                                       cameras,
                                       &rig_from_world,
                                       &num_inliers,
                                       &inlier_mask)) {
    py::gil_scoped_acquire acquire;
    return py::none();
  }

  Eigen::Matrix<double, 6, 6> covariance;
  if (!RefineGeneralizedAbsolutePose(
          refinement_options,
          inlier_mask,
          points2D,
          points3D,
          camera_idxs,
          cams_from_rig,
          &rig_from_world,
          &cameras,
          return_covariance ? &covariance : nullptr)) {
    py::gil_scoped_acquire acquire;
    return py::none();
  }

  py::gil_scoped_acquire acquire;
  py::dict success_dict("rig_from_world"_a = rig_from_world,
                        "num_inliers"_a = num_inliers,
                        "inlier_mask"_a = ToPythonMask(inlier_mask));
  if (return_covariance) success_dict["covariance"] = covariance;
  return success_dict;
}

void BindGeneralizedAbsolutePoseEstimator(py::module& m) {
  m.def("estimate_and_refine_generalized_absolute_pose",
        &PyEstimateAndRefineGeneralizedAbsolutePose,
        "points2D"_a,
        "points3D"_a,
        "camera_idxs"_a,
        "cams_from_rig"_a,
        "cameras"_a,
        py::arg_v("estimation_options",
                  AbsolutePoseEstimationOptions().ransac_options,
                  "AbsolutePoseEstimationOptions().ransac"),
        py::arg_v("refinement_options",
                  AbsolutePoseRefinementOptions(),
                  "AbsolutePoseRefinementOptions()"),
        "return_covariance"_a = false,
        "Robustly estimate generalized absolute pose using LO-RANSAC"
        "followed by non-linear refinement.");
  DefDeprecation(m,
                 "rig_absolute_pose_estimation",
                 "estimate_and_refine_generalized_absolute_pose");
}

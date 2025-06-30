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

py::typing::Optional<py::dict> PyEstimateGeneralizedAbsolutePose(
    const std::vector<Eigen::Vector2d>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const std::vector<size_t>& camera_idxs,
    const std::vector<Rigid3d>& cams_from_rig,
    const std::vector<Camera>& cameras,
    const RANSACOptions& estimation_options) {
  py::gil_scoped_release release;
  Rigid3d rig_from_world;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  if (!EstimateGeneralizedAbsolutePose(estimation_options,
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

  py::gil_scoped_acquire acquire;
  return py::dict("rig_from_world"_a = rig_from_world,
                  "num_inliers"_a = num_inliers,
                  "inlier_mask"_a = ToPythonMask(inlier_mask));
}

py::typing::Optional<py::dict> PyRefineGeneralizedAbsolutePose(
    const Rigid3d& init_rig_from_world,
    const std::vector<Eigen::Vector2d>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const PyInlierMask& inlier_mask,
    const std::vector<size_t>& camera_idxs,
    const std::vector<Rigid3d>& cams_from_rig,
    std::vector<Camera>& cameras,
    const AbsolutePoseRefinementOptions& refinement_options,
    const bool return_covariance) {
  py::gil_scoped_release release;
  Rigid3d refined_rig_from_world = init_rig_from_world;
  std::vector<char> inlier_mask_char(inlier_mask.size());
  Eigen::Map<Eigen::Matrix<char, Eigen::Dynamic, 1>>(
      inlier_mask_char.data(), inlier_mask.size()) = inlier_mask.cast<char>();
  Eigen::Matrix<double, 6, 6> covariance;
  if (!RefineGeneralizedAbsolutePose(
          refinement_options,
          inlier_mask_char,
          points2D,
          points3D,
          camera_idxs,
          cams_from_rig,
          &refined_rig_from_world,
          &cameras,
          return_covariance ? &covariance : nullptr)) {
    py::gil_scoped_acquire acquire;
    return py::none();
  }
  py::gil_scoped_acquire acquire;
  py::dict result("rig_from_world"_a = refined_rig_from_world);
  if (return_covariance) result["covariance"] = covariance;
  return result;
}

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
  py::dict dict("rig_from_world"_a = rig_from_world,
                "num_inliers"_a = num_inliers,
                "inlier_mask"_a = ToPythonMask(inlier_mask));
  if (return_covariance) dict["covariance"] = covariance;
  return dict;
}

py::typing::Optional<py::dict> PyEstimateGeneralizedRelativePose(
    const std::vector<Eigen::Vector2d>& points2D1,
    const std::vector<Eigen::Vector2d>& points2D2,
    const std::vector<size_t>& camera_idxs1,
    const std::vector<size_t>& camera_idxs2,
    const std::vector<Rigid3d>& cams_from_rig,
    const std::vector<Camera>& cameras,
    const RANSACOptions& estimation_options) {
  py::gil_scoped_release release;
  std::optional<Rigid3d> rig2_from_rig1;
  std::optional<Rigid3d> pano2_from_pano1;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  if (!EstimateGeneralizedRelativePose(estimation_options,
                                       points2D1,
                                       points2D2,
                                       camera_idxs1,
                                       camera_idxs2,
                                       cams_from_rig,
                                       cameras,
                                       &rig2_from_rig1,
                                       &pano2_from_pano1,
                                       &num_inliers,
                                       &inlier_mask)) {
    py::gil_scoped_acquire acquire;
    return py::none();
  }

  py::gil_scoped_acquire acquire;
  py::dict dict("num_inliers"_a = num_inliers,
                "inlier_mask"_a = ToPythonMask(inlier_mask));
  if (rig2_from_rig1) dict["rig2_from_rig1"] = *rig2_from_rig1;
  if (pano2_from_pano1) dict["pano2_from_pano1"] = *pano2_from_pano1;
  return dict;
}

void BindGeneralizedAbsolutePoseEstimator(py::module& m) {
  m.def("estimate_generalized_absolute_pose",
        &PyEstimateGeneralizedAbsolutePose,
        "points2D"_a,
        "points3D"_a,
        "camera_idxs"_a,
        "cams_from_rig"_a,
        "cameras"_a,
        py::arg_v("estimation_options", RANSACOptions(), "RANSACOptions()"));

  m.def("refine_generalized_absolute_pose",
        &PyRefineGeneralizedAbsolutePose,
        "rig_from_world"_a,
        "points2D"_a,
        "points3D"_a,
        "inlier_mask"_a,
        "camera_idxs"_a,
        "cams_from_rig"_a,
        "cameras"_a,
        py::arg_v("refinement_options",
                  AbsolutePoseRefinementOptions(),
                  "AbsolutePoseRefinementOptions()"),
        "return_covariance"_a = false,
        "Robustly estimate generalized absolute pose using LO-RANSAC"
        "followed by non-linear refinement.");

  m.def("estimate_and_refine_generalized_absolute_pose",
        &PyEstimateAndRefineGeneralizedAbsolutePose,
        "points2D"_a,
        "points3D"_a,
        "camera_idxs"_a,
        "cams_from_rig"_a,
        "cameras"_a,
        py::arg_v("estimation_options", RANSACOptions(), "RANSACOptions()"),
        py::arg_v("refinement_options",
                  AbsolutePoseRefinementOptions(),
                  "AbsolutePoseRefinementOptions()"),
        "return_covariance"_a = false,
        "Robustly estimate generalized absolute pose using LO-RANSAC"
        "followed by non-linear refinement.");
  DefDeprecation(m,
                 "rig_absolute_pose_estimation",
                 "estimate_and_refine_generalized_absolute_pose");

  m.def("estimate_generalized_relative_pose",
        &PyEstimateGeneralizedRelativePose,
        "points2D1"_a,
        "points2D2"_a,
        "camera_idxs1"_a,
        "camera_idxs2"_a,
        "cams_from_rig"_a,
        "cameras"_a,
        py::arg_v("estimation_options", RANSACOptions(), "RANSACOptions()"));
}

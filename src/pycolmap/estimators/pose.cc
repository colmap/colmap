#include "colmap/estimators/pose.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/math/random.h"
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

py::typing::Optional<py::dict> PyEstimateAbsolutePose(
    const std::vector<Eigen::Vector2d>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    Camera& camera,
    const AbsolutePoseEstimationOptions& estimation_options) {
  py::gil_scoped_release release;
  Rigid3d cam_from_world;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  if (!EstimateAbsolutePose(estimation_options,
                            points2D,
                            points3D,
                            &cam_from_world,
                            &camera,
                            &num_inliers,
                            &inlier_mask)) {
    py::gil_scoped_acquire acquire;
    return py::none();
  }

  py::gil_scoped_acquire acquire;
  return py::dict("cam_from_world"_a = cam_from_world,
                  "num_inliers"_a = num_inliers,
                  "inlier_mask"_a = ToPythonMask(inlier_mask));
}

py::typing::Optional<py::dict> PyRefineAbsolutePose(
    const Rigid3d& init_cam_from_world,
    const std::vector<Eigen::Vector2d>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const PyInlierMask& inlier_mask,
    Camera& camera,
    const AbsolutePoseRefinementOptions& refinement_options,
    const bool return_covariance) {
  py::gil_scoped_release release;
  Rigid3d refined_cam_from_world = init_cam_from_world;
  std::vector<char> inlier_mask_char(inlier_mask.size());
  Eigen::Map<Eigen::Matrix<char, Eigen::Dynamic, 1>>(
      inlier_mask_char.data(), inlier_mask.size()) = inlier_mask.cast<char>();
  Eigen::Matrix<double, 6, 6> covariance;
  if (!RefineAbsolutePose(refinement_options,
                          inlier_mask_char,
                          points2D,
                          points3D,
                          &refined_cam_from_world,
                          &camera,
                          return_covariance ? &covariance : nullptr)) {
    py::gil_scoped_acquire acquire;
    return py::none();
  }
  py::gil_scoped_acquire acquire;
  py::dict result("cam_from_world"_a = refined_cam_from_world);
  if (return_covariance) result["covariance"] = covariance;
  return result;
}

py::typing::Optional<py::dict> PyEstimateAndRefineAbsolutePose(
    const std::vector<Eigen::Vector2d>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    Camera& camera,
    const AbsolutePoseEstimationOptions& estimation_options,
    const AbsolutePoseRefinementOptions& refinement_options,
    const bool return_covariance) {
  py::gil_scoped_release release;
  Rigid3d cam_from_world;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  if (!EstimateAbsolutePose(estimation_options,
                            points2D,
                            points3D,
                            &cam_from_world,
                            &camera,
                            &num_inliers,
                            &inlier_mask)) {
    py::gil_scoped_acquire acquire;
    return py::none();
  }

  Eigen::Matrix<double, 6, 6> covariance;
  if (!RefineAbsolutePose(refinement_options,
                          inlier_mask,
                          points2D,
                          points3D,
                          &cam_from_world,
                          &camera,
                          return_covariance ? &covariance : nullptr)) {
    py::gil_scoped_acquire acquire;
    return py::none();
  }

  py::gil_scoped_acquire acquire;
  py::dict result("cam_from_world"_a = cam_from_world,
                  "num_inliers"_a = num_inliers,
                  "inlier_mask"_a = ToPythonMask(inlier_mask));
  if (return_covariance) result["covariance"] = covariance;
  return result;
}

py::typing::Optional<py::dict> PyEstimateRelativePose(
    const std::vector<Eigen::Vector3d>& cam_rays1,
    const std::vector<Eigen::Vector3d>& cam_rays2,
    const RANSACOptions& estimation_options) {
  py::gil_scoped_release release;
  Rigid3d cam2_from_cam1;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  if (!EstimateRelativePose(estimation_options,
                            cam_rays1,
                            cam_rays2,
                            &cam2_from_cam1,
                            &num_inliers,
                            &inlier_mask)) {
    py::gil_scoped_acquire acquire;
    return py::none();
  }

  py::gil_scoped_acquire acquire;
  return py::dict("cam2_from_cam1"_a = cam2_from_cam1,
                  "num_inliers"_a = num_inliers,
                  "inlier_mask"_a = ToPythonMask(inlier_mask));
}

py::typing::Optional<py::dict> PyRefineRelativePose(
    const Rigid3d& init_cam2_from_cam1,
    const std::vector<Eigen::Vector3d>& cam_rays1,
    const std::vector<Eigen::Vector3d>& cam_rays2,
    const PyInlierMask& inlier_mask,
    const ceres::Solver::Options& refinement_options) {
  py::gil_scoped_release release;
  Rigid3d refined_cam2_from_cam1 = init_cam2_from_cam1;
  std::vector<char> inlier_mask_char(inlier_mask.size());
  Eigen::Map<Eigen::Matrix<char, Eigen::Dynamic, 1>>(
      inlier_mask_char.data(), inlier_mask.size()) = inlier_mask.cast<char>();
  if (!RefineRelativePose(refinement_options,
                          inlier_mask_char,
                          cam_rays1,
                          cam_rays2,
                          &refined_cam2_from_cam1)) {
    py::gil_scoped_acquire acquire;
    return py::none();
  }
  py::gil_scoped_acquire acquire;
  py::dict result("cam2_from_cam1"_a = refined_cam2_from_cam1);
  return result;
}

void BindAbsolutePoseEstimator(py::module& m) {
  auto PyRANSACOptions = m.attr("RANSACOptions");
  py::class_<AbsolutePoseEstimationOptions> PyEstimationOptions(
      m, "AbsolutePoseEstimationOptions");
  PyEstimationOptions.def(py::init<>())
      .def_readwrite("estimate_focal_length",
                     &AbsolutePoseEstimationOptions::estimate_focal_length)
      .def_readwrite("ransac", &AbsolutePoseEstimationOptions::ransac_options);
  MakeDataclass(PyEstimationOptions);

  py::class_<AbsolutePoseRefinementOptions> PyRefinementOptions(
      m, "AbsolutePoseRefinementOptions");
  PyRefinementOptions.def(py::init<>())
      .def_readwrite("gradient_tolerance",
                     &AbsolutePoseRefinementOptions::gradient_tolerance)
      .def_readwrite("max_num_iterations",
                     &AbsolutePoseRefinementOptions::max_num_iterations)
      .def_readwrite("loss_function_scale",
                     &AbsolutePoseRefinementOptions::loss_function_scale)
      .def_readwrite("refine_focal_length",
                     &AbsolutePoseRefinementOptions::refine_focal_length)
      .def_readwrite("refine_extra_params",
                     &AbsolutePoseRefinementOptions::refine_extra_params)
      .def_readwrite("print_summary",
                     &AbsolutePoseRefinementOptions::print_summary);
  MakeDataclass(PyRefinementOptions);

  m.def("estimate_absolute_pose",
        &PyEstimateAbsolutePose,
        "points2D"_a,
        "points3D"_a,
        "camera"_a,
        py::arg_v("estimation_options",
                  AbsolutePoseEstimationOptions(),
                  "AbsolutePoseEstimationOptions()"),
        "Robustly estimate absolute pose using LO-RANSAC "
        "without non-linear refinement.");

  m.def("refine_absolute_pose",
        &PyRefineAbsolutePose,
        "cam_from_world"_a,
        "points2D"_a,
        "points3D"_a,
        "inlier_mask"_a,
        "camera"_a,
        py::arg_v("refinement_options",
                  AbsolutePoseRefinementOptions(),
                  "AbsolutePoseRefinementOptions()"),
        "return_covariance"_a = false,
        "Non-linear refinement of absolute pose.");

  m.def("estimate_and_refine_absolute_pose",
        &PyEstimateAndRefineAbsolutePose,
        "points2D"_a,
        "points3D"_a,
        "camera"_a,
        py::arg_v("estimation_options",
                  AbsolutePoseEstimationOptions(),
                  "AbsolutePoseEstimationOptions()"),
        py::arg_v("refinement_options",
                  AbsolutePoseRefinementOptions(),
                  "AbsolutePoseRefinementOptions()"),
        "return_covariance"_a = false,
        "Robust absolute pose estimation with LO-RANSAC "
        "followed by non-linear refinement.");
  DefDeprecation(
      m, "absolute_pose_estimation", "estimate_and_refine_absolute_pose");

  m.def("estimate_relative_pose",
        &PyEstimateRelativePose,
        "cam_rays1"_a,
        "cam_rays2"_a,
        py::arg_v("options", RANSACOptions(), "RANSACOptions()"),
        "Robustly estimate relative pose using LO-RANSAC "
        "without non-linear refinement.");
  m.def("refine_relative_pose",
        &PyRefineRelativePose,
        "cam2_from_cam1"_a,
        "cam_rays1"_a,
        "cam_rays2"_a,
        "inlier_mask"_a,
        py::arg_v("options", ceres::Solver::Options()),
        "Non-linear refinement of relative pose.");
}

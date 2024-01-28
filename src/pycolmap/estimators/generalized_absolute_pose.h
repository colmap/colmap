#pragma once

#include "colmap/estimators/generalized_pose.h"
#include "colmap/estimators/pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/random.h"
#include "colmap/optim/ransac.h"
#include "colmap/scene/camera.h"

#include "pycolmap/log_exceptions.h"
#include "pycolmap/utils.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

py::object PyEstimateAndRefineGeneralizedAbsolutePose(
    const std::vector<Eigen::Vector2d>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const std::vector<size_t>& camera_idxs,
    const std::vector<Rigid3d>& cams_from_rig,
    std::vector<Camera>& cameras,
    const RANSACOptions& ransac_options,
    const AbsolutePoseRefinementOptions& refinement_options,
    const bool return_covariance) {
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  THROW_CHECK_EQ(points2D.size(), camera_idxs.size());
  THROW_CHECK_EQ(cams_from_rig.size(), cameras.size());
  THROW_CHECK_GE(*std::min_element(camera_idxs.begin(), camera_idxs.end()), 0);
  THROW_CHECK_LT(*std::max_element(camera_idxs.begin(), camera_idxs.end()),
                 cameras.size());

  py::object failure = py::none();
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
    return failure;
  }

  // Absolute pose refinement.
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
    return failure;
  }

  py::gil_scoped_acquire acquire;
  py::dict success_dict("rig_from_world"_a = rig_from_world,
                        "num_inliers"_a = num_inliers,
                        "inliers"_a = ToPythonMask(inlier_mask));
  if (return_covariance) success_dict["covariance"] = covariance;
  return success_dict;
}

void BindGeneralizedAbsolutePoseEstimator(py::module& m) {
  auto est_options = m.attr("RANSACOptions")().cast<RANSACOptions>();
  auto ref_options = m.attr("AbsolutePoseRefinementOptions")()
                         .cast<AbsolutePoseRefinementOptions>();

  m.def(
      "rig_absolute_pose_estimation",
      &PyEstimateAndRefineGeneralizedAbsolutePose,
      "points2D"_a,
      "points3D"_a,
      "cameras"_a,
      "camera_idxs"_a,
      "cams_from_rig"_a,
      "estimation_options"_a = est_options,
      "refinement_options"_a = ref_options,
      "return_covariance"_a = false,
      "Absolute pose estimation with non-linear refinement for a multi-camera "
      "rig.");
}

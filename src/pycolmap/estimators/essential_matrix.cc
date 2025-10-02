#include "colmap/estimators/essential_matrix.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
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

py::typing::Optional<py::dict> PyEstimateAndDecomposeEssentialMatrix(
    const std::vector<Eigen::Vector2d>& points2D1,
    const std::vector<Eigen::Vector2d>& points2D2,
    const Camera& camera1,
    const Camera& camera2,
    const RANSACOptions& options) {
  py::gil_scoped_release release;

  THROW_CHECK_EQ(points2D1.size(), points2D2.size());
  const size_t num_points2D = points2D1.size();

  std::vector<Eigen::Vector3d> cam_rays1(num_points2D);
  std::vector<Eigen::Vector3d> cam_rays2(num_points2D);
  for (size_t point2D_idx = 0; point2D_idx < num_points2D; ++point2D_idx) {
    if (const std::optional<Eigen::Vector2d> cam_point1 =
            camera1.CamFromImg(points2D1[point2D_idx]);
        cam_point1) {
      cam_rays1[point2D_idx] = cam_point1->homogeneous().normalized();
    } else {
      cam_rays1[point2D_idx].setZero();
    }
    if (const std::optional<Eigen::Vector2d> cam_point2 =
            camera2.CamFromImg(points2D2[point2D_idx]);
        cam_point2) {
      cam_rays2[point2D_idx] = cam_point2->homogeneous().normalized();
    } else {
      cam_rays2[point2D_idx].setZero();
    }
  }

  const double max_error_px = options.max_error;
  RANSACOptions ransac_options(options);
  ransac_options.max_error = 0.5 * (max_error_px / camera1.MeanFocalLength() +
                                    max_error_px / camera2.MeanFocalLength());

  LORANSAC<EssentialMatrixFivePointEstimator, EssentialMatrixFivePointEstimator>
      ransac(ransac_options);

  // Essential matrix estimation.
  const auto report = ransac.Estimate(cam_rays1, cam_rays2);

  if (!report.success) {
    py::gil_scoped_acquire acquire;
    return py::none();
  }

  // Pose from essential matrix.
  std::vector<Eigen::Vector3d> inlier_cam_rays1;
  inlier_cam_rays1.reserve(report.support.num_inliers);
  std::vector<Eigen::Vector3d> inlier_cam_rays2;
  inlier_cam_rays1.reserve(report.support.num_inliers);
  for (size_t point2D_idx = 0; point2D_idx < num_points2D; ++point2D_idx) {
    if (report.inlier_mask[point2D_idx]) {
      inlier_cam_rays1.push_back(cam_rays1[point2D_idx]);
      inlier_cam_rays2.push_back(cam_rays2[point2D_idx]);
    }
  }

  Rigid3d cam2_from_cam1;
  std::vector<Eigen::Vector3d> inlier_points3D;
  PoseFromEssentialMatrix(report.model,
                          inlier_cam_rays1,
                          inlier_cam_rays2,
                          &cam2_from_cam1,
                          &inlier_points3D);

  py::gil_scoped_acquire acquire;
  return py::dict("E"_a = report.model,
                  "cam2_from_cam1"_a = cam2_from_cam1,
                  "num_inliers"_a = report.support.num_inliers,
                  "inlier_mask"_a = ToPythonMask(report.inlier_mask),
                  "inlier_points3D"_a = inlier_points3D);
}

void BindEssentialMatrixEstimator(py::module& m) {
  auto ransac_options = m.attr("RANSACOptions")().cast<RANSACOptions>();

  m.def("estimate_essential_matrix",
        &PyEstimateAndDecomposeEssentialMatrix,
        "points2D1"_a,
        "points2D2"_a,
        "camera1"_a,
        "camera2"_a,
        py::arg_v("estimation_options", ransac_options, "RANSACOptions()"),
        "Robustly estimate essential matrix with LO-RANSAC and decompose it "
        "using the cheirality check.");
  DefDeprecation(m, "essential_matrix_estimation", "estimate_essential_matrix");
}

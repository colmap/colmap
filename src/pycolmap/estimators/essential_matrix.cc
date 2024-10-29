#include "colmap/estimators/essential_matrix.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/random.h"
#include "colmap/optim/loransac.h"
#include "colmap/scene/camera.h"
#include "colmap/util/logging.h"

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
    Camera& camera1,
    Camera& camera2,
    const RANSACOptions& options) {
  py::gil_scoped_release release;
  THROW_CHECK_EQ(points2D1.size(), points2D2.size());

  std::vector<Eigen::Vector2d> world_points2D1;
  points2D1_normalized.reserve(points2D1.size());
  std::vector<Eigen::Vector2d> points2D2_normalized;
  points2D2_normalized.reserve(points2D2.size());
  for (size_t i = 0; i < points2D1.size(); ++i) {
    const std::optional<Eigen::Vector2d> point2D1_normalized =
        camera1.CamFromImg(points2D1[i]);
    const std::optional<Eigen::Vector2d> point2D2_normalized =
        camera2.CamFromImg(points2D2[i]);
    if (point2D1_normalized && point2D2_normalized) {
      points2D1_normalized.push_back(*point2D1_normalized);
      points2D2_normalized.push_back(*point2D2_normalized);
    }
  }

  const double max_error_px = options.max_error;
  const double max_error = 0.5 * (max_error_px / camera1.MeanFocalLength() +
                                  max_error_px / camera2.MeanFocalLength());
  RANSACOptions ransac_options(options);
  ransac_options.max_error = max_error;

  LORANSAC<EssentialMatrixFivePointEstimator, EssentialMatrixFivePointEstimator>
      ransac(ransac_options);

  // Essential matrix estimation.
  const auto report =
      ransac.Estimate(points2D1_normalized, points2D2_normalized);

  if (!report.success) {
    py::gil_scoped_acquire acquire;
    return py::none();
  }

  // Recover data from report.
  const Eigen::Matrix3d E = report.model;
  const size_t num_inliers = report.support.num_inliers;

  // Pose from essential matrix.
  std::vector<Eigen::Vector2d> inlier_points2D1_normalized;
  inlier_points2D1_normalized.reserve(report.support.num_inliers);
  std::vector<Eigen::Vector2d> inlier_points2D2_normalized;
  inlier_points2D1_normalized.reserve(report.support.num_inliers);
  for (size_t i = 0; i < report.inlier_mask.size(); ++i) {
    if (report.inlier_mask[i]) {
      inlier_points2D1_normalized.push_back(points2D1_normalized[i]);
      inlier_points2D2_normalized.push_back(points2D2_normalized[i]);
    }
  }

  Rigid3d cam2_from_cam1;
  std::vector<Eigen::Vector3d> points3D;
  PoseFromEssentialMatrix(report.model,
                          inlier_world_points2D1,
                          inlier_world_points2D2,
                          &cam2_from_cam1,
                          &points3D);

  py::gil_scoped_acquire acquire;
  return py::dict("E"_a = report.model,
                  "cam2_from_cam1"_a = cam2_from_cam1,
                  "num_inliers"_a = report.support.num_inliers,
                  "inliers"_a = ToPythonMask(report.inlier_mask));
}

void BindEssentialMatrixEstimator(py::module& m) {
  auto est_options = m.attr("RANSACOptions")().cast<RANSACOptions>();

  m.def("essential_matrix_estimation",
        &PyEstimateAndDecomposeEssentialMatrix,
        "points2D1"_a,
        "points2D2"_a,
        "camera1"_a,
        "camera2"_a,
        py::arg_v("estimation_options", est_options, "RANSACOptions()"),
        "LORANSAC + 5-point algorithm.");
}

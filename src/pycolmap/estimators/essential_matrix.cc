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
  for (size_t idx = 0; idx < points2D1.size(); ++idx) {
    world_points2D1.push_back(camera1.CamFromImg(points2D1[idx]));
  }
  std::vector<Eigen::Vector2d> world_points2D2;
  for (size_t idx = 0; idx < points2D2.size(); ++idx) {
    world_points2D2.push_back(camera2.CamFromImg(points2D2[idx]));
  }

  const double max_error_px = options.max_error;
  const double max_error = 0.5 * (max_error_px / camera1.MeanFocalLength() +
                                  max_error_px / camera2.MeanFocalLength());
  RANSACOptions ransac_options(options);
  ransac_options.max_error = max_error;

  LORANSAC<EssentialMatrixFivePointEstimator, EssentialMatrixFivePointEstimator>
      ransac(ransac_options);

  // Essential matrix estimation.
  const auto report = ransac.Estimate(world_points2D1, world_points2D2);

  if (!report.success) {
    py::gil_scoped_acquire acquire;
    return py::none();
  }

  // Recover data from report.
  const Eigen::Matrix3d E = report.model;
  const size_t num_inliers = report.support.num_inliers;
  const auto& inlier_mask = report.inlier_mask;

  // Pose from essential matrix.
  std::vector<Eigen::Vector2d> inlier_world_points2D1;
  std::vector<Eigen::Vector2d> inlier_world_points2D2;

  for (size_t idx = 0; idx < inlier_mask.size(); ++idx) {
    if (inlier_mask[idx]) {
      inlier_world_points2D1.push_back(world_points2D1[idx]);
      inlier_world_points2D2.push_back(world_points2D2[idx]);
    }
  }

  Rigid3d cam2_from_cam1;
  Eigen::Matrix3d cam2_from_cam1_rot_mat;
  std::vector<Eigen::Vector3d> points3D;
  PoseFromEssentialMatrix(E,
                          inlier_world_points2D1,
                          inlier_world_points2D2,
                          &cam2_from_cam1_rot_mat,
                          &cam2_from_cam1.translation,
                          &points3D);
  cam2_from_cam1.rotation = Eigen::Quaterniond(cam2_from_cam1_rot_mat);

  py::gil_scoped_acquire acquire;
  return py::dict("E"_a = E,
                  "cam2_from_cam1"_a = cam2_from_cam1,
                  "num_inliers"_a = num_inliers,
                  "inliers"_a = ToPythonMask(inlier_mask));
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

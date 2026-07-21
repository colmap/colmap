#include "colmap/estimators/solvers/essential_matrix.h"

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

  // Unproject to rays + per-ray Jacobians (for the pixel-unit tangent Sampson
  // score). End users pass camera + 2D points; Jacobians are never part of the
  // interface. Unprojectable points are zeroed -> infinite residual ->
  // rejected.
  std::vector<CamRayWithJac> cam_rays_with_jac1(num_points2D);
  std::vector<CamRayWithJac> cam_rays_with_jac2(num_points2D);
  for (size_t point2D_idx = 0; point2D_idx < num_points2D; ++point2D_idx) {
    cam_rays_with_jac1[point2D_idx] =
        camera1.CamRayFromImgWithJac(points2D1[point2D_idx])
            .value_or(CamRayWithJac::Zero());
    cam_rays_with_jac2[point2D_idx] =
        camera2.CamRayFromImgWithJac(points2D2[point2D_idx])
            .value_or(CamRayWithJac::Zero());
  }

  // The tangent Sampson residual is in pixels, so the pixel threshold is used
  // directly (no CamFromImgThreshold / focal-length conversion).
  LORANSAC<EssentialMatrixTangentSampsonEstimator,
           EssentialMatrixTangentSampsonEstimator>
      ransac(options);

  // Essential matrix estimation.
  const auto report = ransac.Estimate(cam_rays_with_jac1, cam_rays_with_jac2);

  if (!report.success) {
    py::gil_scoped_acquire acquire;
    return py::none();
  }

  // Pose from essential matrix.
  std::vector<Eigen::Vector3d> inlier_cam_rays1;
  inlier_cam_rays1.reserve(report.support.num_inliers);
  std::vector<Eigen::Vector3d> inlier_cam_rays2;
  inlier_cam_rays2.reserve(report.support.num_inliers);
  for (size_t point2D_idx = 0; point2D_idx < num_points2D; ++point2D_idx) {
    if (report.inlier_mask[point2D_idx]) {
      inlier_cam_rays1.push_back(cam_rays_with_jac1[point2D_idx].ray);
      inlier_cam_rays2.push_back(cam_rays_with_jac2[point2D_idx].ray);
    }
  }

  Rigid3d cam2_from_cam1;
  std::vector<int> valid_indices;
  PoseFromEssentialMatrix(report.model,
                          inlier_cam_rays1,
                          inlier_cam_rays2,
                          &cam2_from_cam1,
                          &valid_indices);

  py::gil_scoped_acquire acquire;
  return py::dict("E"_a = report.model,
                  "cam2_from_cam1"_a = cam2_from_cam1,
                  "num_inliers"_a = report.support.num_inliers,
                  "inlier_mask"_a = ToPythonMask(report.inlier_mask));
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
}

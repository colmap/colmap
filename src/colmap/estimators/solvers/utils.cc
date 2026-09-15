// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/solvers/utils.h"

#include "colmap/util/logging.h"

#include <Eigen/QR>
#include <Eigen/SVD>

namespace colmap {

Eigen::Matrix3d SolveEpipolarConstraintMatrix(
    const Eigen::Matrix<double, Eigen::Dynamic, 9>& A) {
  THROW_CHECK_GE(A.rows(), 8);

  // Solve for the nullspace of the constraint matrix.
  Eigen::Matrix3d Q;
  if (A.rows() == 8) {
    Eigen::Matrix<double, 9, 9> QQ =
        A.transpose().householderQr().householderQ();
    Q = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
        QQ.col(8).data());
  } else {
    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
        A, Eigen::ComputeFullV);
    Q = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
        svd.matrixV().col(8).data());
  }

  // Enforce rank at most 2.
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      Q, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d singular_values = svd.singularValues();
  singular_values(2) = 0.0;
  return svd.matrixU() * singular_values.asDiagonal() *
         svd.matrixV().transpose();
}

std::vector<Eigen::Vector3d> CalibratedRays(
    const std::vector<Eigen::Vector2d>& points, const double focal) {
  const double inv_f = 1.0 / focal;
  std::vector<Eigen::Vector3d> rays(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    rays[i] = Eigen::Vector3d(points[i].x() * inv_f, points[i].y() * inv_f, 1.0)
                  .normalized();
  }
  return rays;
}

std::vector<Eigen::Vector3d> RaysFromCamRaysWithJac(
    const std::vector<CamRayWithJac>& cam_rays_with_jac) {
  std::vector<Eigen::Vector3d> rays(cam_rays_with_jac.size());
  for (size_t i = 0; i < cam_rays_with_jac.size(); ++i) {
    rays[i] = cam_rays_with_jac[i].ray;
  }
  return rays;
}

}  // namespace colmap

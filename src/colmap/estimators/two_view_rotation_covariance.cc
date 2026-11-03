// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/two_view_rotation_covariance.h"

#include "colmap/estimators/cost_functions/tiny_manifold.h"
#include "colmap/estimators/cost_functions/tiny_sampson_error.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

namespace colmap {

namespace {

// Relative tolerance below which an eigenvalue of an information matrix is
// considered numerically zero.
constexpr double kMinRelativeEigenvalue = 1e-12;

// Minimum ratio between the smallest and largest variance of a rotation
// covariance, so that storing it in single precision preserves
// positive-definiteness.
constexpr double kMinRelativeVariance = 1e-5;

// Information matrix of the relative rotation under a pure-rotation
// (zero-baseline) model, in which each correspondence constrains the 2D
// tangent plane orthogonal to ray2 via c_i = B2^T (R21 * ray1).
std::optional<Eigen::Matrix3d> EstimatePanoramicRotationInformation(
    const std::vector<CamRayWithJac>& rays1,
    const std::vector<CamRayWithJac>& rays2,
    const Eigen::Quaterniond& cam2_from_cam1_rotation,
    double point2D_stddev_px) {
  const Eigen::Matrix3d R21 = cam2_from_cam1_rotation.toRotationMatrix();
  Eigen::Matrix3d unscaled_information = Eigen::Matrix3d::Zero();
  size_t num_valid = 0;

  for (size_t i = 0; i < rays1.size(); ++i) {
    const Eigen::Vector3d& x1 = rays1[i].ray;
    const Eigen::Matrix3x2d& J1 = rays1[i].jacobian;
    const Eigen::Vector3d& x2 = rays2[i].ray;
    const Eigen::Matrix3x2d& J2 = rays2[i].jacobian;

    Eigen::Matrix<double, 3, 2, Eigen::RowMajor> B2;
    SphereManifold<3>().PlusJacobian(x2.data(), B2.data());

    // Whiten the residual Jacobian with the covariance propagated from the
    // image noise.
    const Eigen::Matrix3x2d R21_J1 = R21 * J1;
    const Eigen::Matrix2d C_i =
        B2.transpose() * (R21_J1 * R21_J1.transpose() + J2 * J2.transpose()) *
        B2;
    const Eigen::LLT<Eigen::Matrix2d> llt(C_i);
    if (llt.info() != Eigen::Success) {
      continue;
    }

    // d(R21 * Exp(delta) * x1)/d(delta) = -R21 * [x1]_x.
    const Eigen::Matrix<double, 2, 3> J_c =
        -B2.transpose() * R21 * CrossProductMatrix(x1);
    const Eigen::Matrix<double, 2, 3> J_R_i = llt.matrixL().solve(J_c);

    unscaled_information.noalias() += J_R_i.transpose() * J_R_i;
    ++num_valid;
  }

  if (num_valid < 3) {
    return std::nullopt;
  }
  return unscaled_information / (point2D_stddev_px * point2D_stddev_px);
}

// Information matrix of the relative pose from the tangent Sampson errors,
// w.r.t. the right perturbation of the rotation and the perturbation of the
// translation direction on the tangent space of S^2.
Eigen::Matrix<double, 5, 5> EstimateRelativePoseInformation(
    const std::vector<CamRayWithJac>& rays1,
    const std::vector<CamRayWithJac>& rays2,
    const Rigid3d& cam2_from_cam1,
    double point2D_stddev_px) {
  using RelativePoseManifold =
      ProductManifold<EigenQuaternionManifold, SphereManifold<3>>;

  const RelPoseParams params = RelPoseParamsFromRigid3d(cam2_from_cam1);

  const size_t num_inliers = rays1.size();
  const TinyTangentSampsonErrorCostFunctor cost_fn(rays1, rays2);
  Eigen::VectorXd residuals(num_inliers);
  Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::ColMajor> J_amb(num_inliers,
                                                                  7);
  cost_fn(params.data(), residuals.data(), J_amb.data());

  Eigen::Matrix<double, 7, 5, Eigen::RowMajor> plus_jac;
  RelativePoseManifold().PlusJacobian(params.data(), plus_jac.data());
  const Eigen::Matrix<double, Eigen::Dynamic, 5> J_tan = J_amb * plus_jac;

  return (J_tan.transpose() * J_tan) / (point2D_stddev_px * point2D_stddev_px);
}

// Marginalizes the translation direction out of the relative pose information
// with the Schur complement, excluding only numerically null directions of its
// information.
std::optional<Eigen::Matrix3d> MarginalizeTranslationDirection(
    const Eigen::Matrix<double, 5, 5>& information) {
  const Eigen::Matrix3d Lambda_RR = information.topLeftCorner<3, 3>();
  const Eigen::Matrix<double, 3, 2> Lambda_Rt =
      information.topRightCorner<3, 2>();
  const Eigen::Matrix2d Lambda_tt = information.bottomRightCorner<2, 2>();

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> trans_eig(Lambda_tt);
  if (trans_eig.info() != Eigen::Success) {
    return std::nullopt;
  }
  const Eigen::Vector2d& trans_evals = trans_eig.eigenvalues();
  const Eigen::Matrix2d& trans_evecs = trans_eig.eigenvectors();
  const double trans_thresh =
      kMinRelativeEigenvalue * std::max(0.0, trans_evals(1));

  Eigen::Matrix2d Lambda_tt_pinv = Eigen::Matrix2d::Zero();
  for (int j = 0; j < 2; ++j) {
    if (trans_evals(j) > trans_thresh) {
      Lambda_tt_pinv.noalias() +=
          (1.0 / trans_evals(j)) *
          (trans_evecs.col(j) * trans_evecs.col(j).transpose());
    }
  }

  return Lambda_RR - Lambda_Rt * Lambda_tt_pinv * Lambda_Rt.transpose();
}

// Inverts the rotation information into a covariance. The standard deviation
// along poorly or unconstrained axes, e.g., of geometrically degenerate
// configurations, is capped to max_sigma_deg, such that their edges are never
// more certain than well-conditioned ones. Returns nullopt if the
// eigendecomposition fails.
std::optional<Eigen::Matrix3d> RotationCovarianceFromInformation(
    const Eigen::Matrix3d& information, double max_sigma_deg) {
  const double max_sigma = DegToRad(max_sigma_deg);
  const double max_variance = max_sigma * max_sigma;
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(
      0.5 * (information + information.transpose()));
  if (eig.info() != Eigen::Success) {
    return std::nullopt;
  }

  Eigen::Vector3d variances;
  for (int i = 0; i < 3; ++i) {
    const double eigenvalue = eig.eigenvalues()(i);
    variances(i) =
        eigenvalue * max_variance > 1.0 ? 1.0 / eigenvalue : max_variance;
  }
  variances = variances.cwiseMax(kMinRelativeVariance * variances.maxCoeff());

  const Eigen::Matrix3d cov = eig.eigenvectors() * variances.asDiagonal() *
                              eig.eigenvectors().transpose();
  return 0.5 * (cov + cov.transpose());
}

}  // namespace

bool TwoViewRotationCovarianceOptions::Check() const {
  CHECK_OPTION(std::isfinite(point2D_stddev_px));
  CHECK_OPTION_GT(point2D_stddev_px, 0.0);
  CHECK_OPTION(std::isfinite(max_rotation_sigma_deg));
  CHECK_OPTION_GT(max_rotation_sigma_deg, 0.0);
  return true;
}

std::optional<Eigen::Matrix3d> EstimateTwoViewRotationCovariance(
    const Camera& camera1,
    const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2,
    const std::vector<Eigen::Vector2d>& points2,
    const TwoViewGeometry& geometry,
    const TwoViewRotationCovarianceOptions& options) {
  THROW_CHECK(options.Check());
  if (!geometry.cam2_from_cam1.has_value()) {
    return std::nullopt;
  }
  const Rigid3d& cam2_from_cam1 = *geometry.cam2_from_cam1;

  // Use the intrinsics estimated with the two-view geometry, if any.
  const auto [rays1, rays2] = ExtractMatchedCamRaysWithJac(
      geometry.camera1.has_value() ? *geometry.camera1 : camera1,
      points1,
      geometry.camera2.has_value() ? *geometry.camera2 : camera2,
      points2,
      geometry.inlier_matches,
      /*skip_invalid=*/true);

  const bool is_panoramic =
      geometry.config == TwoViewGeometry::ConfigurationType::PANORAMIC ||
      cam2_from_cam1.translation().squaredNorm() < 1e-12;
  const size_t num_inliers = rays1.size();
  if (num_inliers < (is_panoramic ? 3 : 5)) {
    return std::nullopt;
  }

  std::optional<Eigen::Matrix3d> rotation_information;
  if (is_panoramic) {
    rotation_information = EstimatePanoramicRotationInformation(
        rays1, rays2, cam2_from_cam1.rotation(), options.point2D_stddev_px);
  } else {
    rotation_information =
        MarginalizeTranslationDirection(EstimateRelativePoseInformation(
            rays1, rays2, cam2_from_cam1, options.point2D_stddev_px));
  }
  if (!rotation_information.has_value()) {
    return std::nullopt;
  }

  return RotationCovarianceFromInformation(*rotation_information,
                                           options.max_rotation_sigma_deg);
}

}  // namespace colmap

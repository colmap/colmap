// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/estimators/cost_functions/quaternion_utils.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/geometry/pose.h"

#include <cmath>

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

// Builds the essential matrix E = [t]_x R from a relative pose given in the
// Rigid3d parameter layout [qx, qy, qz, qw, tx, ty, tz]. Templated on the
// scalar so it works under Ceres autodiff.
template <typename T>
Eigen::Matrix<T, 3, 3> EssentialMatrixFromPoseParams(
    const T* const cam2_from_cam1) {
  const Eigen::Matrix<T, 3, 3> R =
      EigenQuaternionMap<T>(cam2_from_cam1).toRotationMatrix();
  // Matrix representation of the cross product t x R.
  Eigen::Matrix<T, 3, 3> t_x;
  t_x << T(0), -cam2_from_cam1[6], cam2_from_cam1[5], cam2_from_cam1[6], T(0),
      -cam2_from_cam1[4], -cam2_from_cam1[5], cam2_from_cam1[4], T(0);
  return t_x * R;
}

// Signed Sampson error under an essential/fundamental matrix. point1/point2 are
// points on the image plane, taken as 2D because the error is not invariant to
// the scale of the homogeneous representative, so only (x, y, 1) is correct
// here. For rays use TangentSampsonError. Returns 0 when the denominator
// vanishes.
template <typename T>
T SampsonError(const Eigen::Matrix<T, 3, 3>& E,
               const Eigen::Matrix<T, 2, 1>& point1,
               const Eigen::Matrix<T, 2, 1>& point2) {
  const Eigen::Matrix<T, 3, 1> point2_homogeneous = point2.homogeneous();
  const Eigen::Matrix<T, 3, 1> epipolar_line1 = E * point1.homogeneous();
  const T num = point2_homogeneous.dot(epipolar_line1);
  const Eigen::Matrix<T, 4, 1> denom(point2_homogeneous.dot(E.col(0)),
                                     point2_homogeneous.dot(E.col(1)),
                                     epipolar_line1.x(),
                                     epipolar_line1.y());
  const T denom_norm = denom.norm();
  if (denom_norm == static_cast<T>(0)) {
    return static_cast<T>(0);
  }
  return num / denom_norm;
}

// Signed tangent Sampson error of one correspondence under E, in pixels, using
// the unprojection Jacobians J_ray1 = d(ray1)/d(pixel1) and J_ray2. Returns 0
// when the denominator vanishes. See ComputeSquaredTangentSampsonError.
template <typename T>
T TangentSampsonError(const Eigen::Matrix<T, 3, 3>& E,
                      const Eigen::Matrix<T, 3, 1>& cam_ray1,
                      const Eigen::Matrix<T, 3, 2>& J_ray1,
                      const Eigen::Matrix<T, 3, 1>& cam_ray2,
                      const Eigen::Matrix<T, 3, 2>& J_ray2) {
  const Eigen::Matrix<T, 3, 1> Eray1 = E * cam_ray1;
  const Eigen::Matrix<T, 3, 1> Etray2 = E.transpose() * cam_ray2;
  const T num = cam_ray2.dot(Eray1);
  Eigen::Matrix<T, 4, 1> denom;
  denom << J_ray1.transpose() * Etray2, J_ray2.transpose() * Eray1;
  const T denom_norm = denom.norm();
  if (denom_norm == static_cast<T>(0)) {
    return static_cast<T>(0);
  }
  return num / denom_norm;
}

// Tangent Sampson residual and optional closed-form Jacobian w.r.t. E.
// Returns a zero residual and Jacobian when the denominator vanishes.
inline double TangentSampsonErrorAndJacWrtE(const Eigen::Matrix3d& E,
                                            const Eigen::Vector3d& cam_ray1,
                                            const Eigen::Matrix3x2d& J_ray1,
                                            const Eigen::Vector3d& cam_ray2,
                                            const Eigen::Matrix3x2d& J_ray2,
                                            Eigen::Matrix3d* drdE = nullptr) {
  const Eigen::Vector3d Eray1 = E * cam_ray1;
  const Eigen::Vector3d Etray2 = E.transpose() * cam_ray2;
  const double num = cam_ray2.dot(Eray1);
  const Eigen::Vector2d a = J_ray1.transpose() * Etray2;
  const Eigen::Vector2d b = J_ray2.transpose() * Eray1;
  const double denom = a.squaredNorm() + b.squaredNorm();
  const double sqrt_denom = std::sqrt(denom);
  if (sqrt_denom == 0.0) {
    if (drdE != nullptr) {
      drdE->setZero();
    }
    return 0.0;
  }
  if (drdE != nullptr) {
    // dr/dE = (1/sqrt_denom) ray2 ray1^T
    //         - (num/denom^1.5) (ray2 (J1 a)^T + (J2 b) ray1^T).
    const Eigen::Vector3d J1a = J_ray1 * a;
    const Eigen::Vector3d J2b = J_ray2 * b;
    const double coef = num / (denom * sqrt_denom);
    *drdE = (1.0 / sqrt_denom) * (cam_ray2 * cam_ray1.transpose()) -
            coef * (cam_ray2 * J1a.transpose() + J2b * cam_ray1.transpose());
  }
  return num / sqrt_denom;
}

// E = [t]_x R and optional derivatives for [qx, qy, qz, qw, tx, ty, tz].
inline Eigen::Matrix3d EssentialMatrixAndJacFromPoseParams(
    const double* params, Eigen::Matrix3d* dE /*[7] or nullptr*/) {
  const Eigen::Map<const Eigen::Quaterniond> q(params);
  const Eigen::Matrix3d R = q.toRotationMatrix();
  Eigen::Matrix3d t_x;
  t_x << 0, -params[6], params[5], params[6], 0, -params[4], -params[5],
      params[4], 0;
  if (dE != nullptr) {
    const double x = params[0], y = params[1], z = params[2], w = params[3];
    Eigen::Matrix3d dR[4];
    dR[0] << 0, 2 * y, 2 * z, 2 * y, -4 * x, -2 * w, 2 * z, 2 * w,
        -4 * x;  // dR/dqx
    dR[1] << -4 * y, 2 * x, 2 * w, 2 * x, 0, 2 * z, -2 * w, 2 * z,
        -4 * y;  // dR/dqy
    dR[2] << -4 * z, -2 * w, 2 * x, 2 * w, -4 * z, 2 * y, 2 * x, 2 * y,
        0;                                                          // dR/dqz
    dR[3] << 0, -2 * z, 2 * y, 2 * z, 0, -2 * x, -2 * y, 2 * x, 0;  // dR/dqw
    for (int l = 0; l < 4; ++l) dE[l] = t_x * dR[l];
    Eigen::Matrix3d ex, ey, ez;
    ex << 0, 0, 0, 0, 0, -1, 0, 1, 0;
    ey << 0, 0, 1, 0, 0, 0, -1, 0, 0;
    ez << 0, -1, 0, 1, 0, 0, 0, 0, 0;
    dE[4] = ex * R;  // dE/dtx
    dE[5] = ey * R;  // dE/dty
    dE[6] = ez * R;  // dE/dtz
  }
  return t_x * R;
}

// Pack a pose as [qx, qy, qz, qw, tx, ty, tz], normalizing both components.
inline Eigen::Matrix<double, 7, 1> PoseParamsFromRigid3d(
    const Rigid3d& cam2_from_cam1) {
  Eigen::Matrix<double, 7, 1> params;
  params.head<4>() = cam2_from_cam1.rotation().normalized().coeffs();
  params.tail<3>() = cam2_from_cam1.translation().normalized();
  return params;
}

// Unpack a pose, normalizing its quaternion.
inline Rigid3d Rigid3dFromPoseParams(const double* params) {
  const Eigen::Quaterniond rotation =
      Eigen::Map<const Eigen::Quaterniond>(params).normalized();
  const Eigen::Vector3d translation =
      Eigen::Map<const Eigen::Vector3d>(params + 4);
  return Rigid3d(rotation, translation);
}

// Refines a relative pose by the Sampson error of image-plane point
// correspondences. See SampsonError. The pose is [qx, qy, qz, qw, tx, ty, tz]
// with the translation on the unit sphere, so it needs a SphereManifold on
// tvec. For calibrated rays with unprojection Jacobians use the
// TangentSampsonErrorCostFunctor, which is pixel-accurate for any central
// model.
class SampsonErrorCostFunctor
    : public AutoDiffCostFunctor<SampsonErrorCostFunctor, 1, 7> {
 public:
  SampsonErrorCostFunctor(const Eigen::Vector2d& point1,
                          const Eigen::Vector2d& point2)
      : point1_(point1), point2_(point2) {}

  template <typename T>
  bool operator()(const T* const cam2_from_cam1, T* residuals) const {
    const Eigen::Matrix<T, 3, 3> E =
        EssentialMatrixFromPoseParams(cam2_from_cam1);
    residuals[0] = SampsonError<T>(E, point1_.cast<T>(), point2_.cast<T>());
    return true;
  }

 private:
  const Eigen::Vector2d point1_;
  const Eigen::Vector2d point2_;
};

// Refines a relative pose by the pixel-unit tangent Sampson error of calibrated
// ray correspondences with unprojection Jacobians. See TangentSampsonError.
// Pose layout matches SampsonErrorCostFunctor. Pixel-accurate for any central
// model.
class TangentSampsonErrorCostFunctor
    : public AutoDiffCostFunctor<TangentSampsonErrorCostFunctor, 1, 7> {
 public:
  TangentSampsonErrorCostFunctor(const CamRayWithJac& cam_ray1_with_jac,
                                 const CamRayWithJac& cam_ray2_with_jac)
      : cam_ray1_with_jac_(cam_ray1_with_jac),
        cam_ray2_with_jac_(cam_ray2_with_jac) {}

  template <typename T>
  bool operator()(const T* const cam2_from_cam1, T* residuals) const {
    const Eigen::Matrix<T, 3, 3> E =
        EssentialMatrixFromPoseParams(cam2_from_cam1);
    residuals[0] =
        TangentSampsonError<T>(E,
                               cam_ray1_with_jac_.ray.cast<T>(),
                               cam_ray1_with_jac_.jacobian.cast<T>(),
                               cam_ray2_with_jac_.ray.cast<T>(),
                               cam_ray2_with_jac_.jacobian.cast<T>());
    return true;
  }

 private:
  const CamRayWithJac cam_ray1_with_jac_;
  const CamRayWithJac cam_ray2_with_jac_;
};

}  // namespace colmap

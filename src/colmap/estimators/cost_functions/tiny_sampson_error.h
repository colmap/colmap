// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/estimators/cost_functions/sampson_error.h"
#include "colmap/geometry/pose.h"
#include "colmap/util/types.h"

#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <ceres/tiny_solver_autodiff_function.h>

namespace colmap {

// Tangent Sampson (pixel-space) cost functor for TinySolver refinement of a
// two-view relative pose from calibrated rays with unprojection Jacobians.
// E = [t]_x R is built from the pose [qx, qy, qz, qw, tx, ty, tz] (Rigid3d
// params). The solver applies the manifold. Pixel-accurate for any central
// model, matching EssentialMatrixTangentSampsonEstimator's score.
//
// The residual is r = num / sqrt(denom). The 7-parameter Jacobian dr/d[q,t] is
// computed in closed form (dr/dE contracted with dE/d[q,t]), ~3x faster than
// autodiff here. A unit test checks it against finite differences.
class TinyTangentSampsonErrorCostFunctor {
 public:
  using Scalar = double;
  static constexpr int NUM_RESIDUALS = Eigen::Dynamic;
  static constexpr int NUM_PARAMETERS = 7;

  TinyTangentSampsonErrorCostFunctor(
      const std::vector<CamRayWithJac>& cam_rays1_with_jac,
      const std::vector<CamRayWithJac>& cam_rays2_with_jac)
      : cam_rays1_with_jac_(cam_rays1_with_jac),
        cam_rays2_with_jac_(cam_rays2_with_jac) {}

  int NumResiduals() const {
    return static_cast<int>(cam_rays1_with_jac_.size());
  }

  // jacobian is NUM_RESIDUALS x 7, column-major (or null for residuals only).
  bool operator()(const double* params,
                  double* residuals,
                  double* jacobian) const {
    const Eigen::Map<const Eigen::Quaterniond> q(params);
    const Eigen::Matrix3d R = q.toRotationMatrix();
    Eigen::Matrix3d t_x;
    t_x << 0, -params[6], params[5], params[6], 0, -params[4], -params[5],
        params[4], 0;
    const Eigen::Matrix3d E = t_x * R;

    Eigen::Matrix3d dE[7];
    if (jacobian != nullptr) {
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

    const int n = static_cast<int>(cam_rays1_with_jac_.size());
    for (int i = 0; i < n; ++i) {
      const Eigen::Vector3d& ray1 = cam_rays1_with_jac_[i].ray;
      const Eigen::Vector3d& ray2 = cam_rays2_with_jac_[i].ray;
      const Eigen::Matrix3x2d& J1 = cam_rays1_with_jac_[i].jacobian;
      const Eigen::Matrix3x2d& J2 = cam_rays2_with_jac_[i].jacobian;
      const Eigen::Vector3d Eray1 = E * ray1;
      const Eigen::Vector3d Etray2 = E.transpose() * ray2;
      const double num = ray2.dot(Eray1);
      const Eigen::Vector2d a = J1.transpose() * Etray2;
      const Eigen::Vector2d b = J2.transpose() * Eray1;
      const double denom = a.squaredNorm() + b.squaredNorm();
      const double sqrt_denom = std::sqrt(denom);
      if (sqrt_denom == 0.0) {
        residuals[i] = 0.0;
        if (jacobian != nullptr) {
          for (int l = 0; l < 7; ++l) jacobian[i + l * n] = 0.0;
        }
        continue;
      }
      residuals[i] = num / sqrt_denom;
      if (jacobian != nullptr) {
        // dr/dE = (1/sqrt_denom) ray2 ray1^T
        //         - (num/denom^1.5) (ray2 (J1 a)^T + (J2 b) ray1^T).
        const Eigen::Vector3d J1a = J1 * a;
        const Eigen::Vector3d J2b = J2 * b;
        const double coef = num / (denom * sqrt_denom);
        const Eigen::Matrix3d drdE =
            (1.0 / sqrt_denom) * (ray2 * ray1.transpose()) -
            coef * (ray2 * J1a.transpose() + J2b * ray1.transpose());
        for (int l = 0; l < 7; ++l) {
          jacobian[i + l * n] = drdE.cwiseProduct(dE[l]).sum();
        }
      }
    }
    return true;
  }

 private:
  const std::vector<CamRayWithJac>& cam_rays1_with_jac_;
  const std::vector<CamRayWithJac>& cam_rays2_with_jac_;
};

// Sampson-error cost functor for fixed-size (colmap::TinySolver) refinement of
// a two-view relative pose *and* a shared, unknown focal length.
//
// The pose is parameterized as [qx, qy, qz, qw, tx, ty, tz] and the focal is
// appended as an eighth parameter optimized in log-space (log_f), which keeps
// it strictly positive and gives a scale-invariant step. The essential matrix
// is built from the pose, then converted to the fundamental matrix implied by
// the focal so the Sampson error is measured in *pixel* space:
//
//   F = diag(1/f, 1/f, 1) * E * diag(1/f, 1/f, 1),   f = exp(log_f).
//
// The inputs points1/points2 are therefore principal-point-centered image
// points (u - cx, v - cy), not calibrated rays. The 8 ambient parameters are
// parameterized directly here. The 6-DoF manifold (rotation on SO(3),
// translation on the unit sphere, log-focal) is applied by the solver.
class TinyFocalSampsonErrorCostFunctor {
 public:
  using Scalar = double;
  static constexpr int NUM_RESIDUALS = Eigen::Dynamic;
  static constexpr int NUM_PARAMETERS = 8;

  // ceres::TinySolver-compatible autodiff wrapper for this functor. Note that
  // it stores the functor by reference, so the wrapped functor must outlive it.
  using AutoDiffFunction =
      ceres::TinySolverAutoDiffFunction<TinyFocalSampsonErrorCostFunctor,
                                        NUM_RESIDUALS,
                                        NUM_PARAMETERS>;

  TinyFocalSampsonErrorCostFunctor(const std::vector<Eigen::Vector2d>& points1,
                                   const std::vector<Eigen::Vector2d>& points2)
      : points1_(points1), points2_(points2) {}

  int NumResiduals() const { return static_cast<int>(points1_.size()); }

  template <typename T>
  bool operator()(const T* const params, T* residuals) const {
    // Build E once from the pose, then scale it into the pixel-space
    // fundamental matrix F = diag(inv_f, inv_f, 1) * E * diag(inv_f, inv_f, 1).
    Eigen::Matrix<T, 3, 3> F = EssentialMatrixFromPoseParams(params);
    const T inv_f = ceres::exp(-params[7]);
    F.template topLeftCorner<2, 2>() *= inv_f * inv_f;
    F.template block<2, 1>(0, 2) *= inv_f;
    F.template block<1, 2>(2, 0) *= inv_f;
    for (size_t i = 0; i < points1_.size(); ++i) {
      residuals[i] =
          SampsonError<T>(F, points1_[i].cast<T>(), points2_[i].cast<T>());
    }
    return true;
  }

 private:
  const std::vector<Eigen::Vector2d>& points1_;
  const std::vector<Eigen::Vector2d>& points2_;
};

// Tangent Sampson (pixel-space) cost functor for fixed-size
// (colmap::TinySolver) refinement of a two-view relative pose and a *single*
// unknown focal length, where the second view is already calibrated (the
// semi-calibrated, or "one-sided focal", configuration).
//
// The pose is parameterized as in TinyFocalSampsonErrorCostFunctor
// ([qx, qy, qz, qw, tx, ty, tz]) and the unknown focal of the *first* view is
// appended as an eighth parameter optimized in log-space. The second view needs
// no focal parameter, because it enters as calibrated bearing rays. Following
// ray2^T E b1 = 0 with b1 = K1inv * point1, the mixed epipolar matrix is
//
//   M = E * diag(1/f1, 1/f1, 1),   f1 = exp(log_f1),
//
// i.e. only columns 0-1 are scaled, and ray2^T M point1 = 0. The residual is
// the tangent Sampson error of that constraint, in pixels:
//
//   C       = ray2^T M point1
//   dC/dpx1 = (M^T ray2).head<2>()
//   dC/dpx2 = J2^T (M point1)
//   r       = C / sqrt(||dC/dpx1||^2 + ||dC/dpx2||^2)
//
// Both denominator terms are gradients of the same scalar with respect to
// pixels, so they share units and combine directly. A plain Sampson error would
// instead differentiate the second view with respect to its *ray* and collapse
// onto that term. J2 = d(ray2)/d(pixel2) represents any central model exactly,
// including non-pinhole ones. Matching
// RelativePoseOneSidedFocalEstimator::Residuals matters because LO-RANSAC
// scores the refined model with that residual.
//
// img_points1 are principal-point-centered points of the uncalibrated view.
// Keeping them as raw pixels rather than bearings is deliberate: the
// measurement Jacobian d(x, y, 1)/d(x, y) is then the constant [I2; 0] and f1
// enters only through M. Unprojecting the first view to a bearing instead would
// place f1 inside that view's Jacobian, so differentiating the residual would
// need the mixed second derivative d^2(ray1)/d(pixel1)d(f1), which no camera
// model exposes. For the same reason the Jacobians below hold only while the
// second view's intrinsics are fixed, as they are here.
//
// The 8-parameter Jacobian is computed in closed form, dr/dM contracted with
// dM/d[q, t, log_f1]. A unit test checks it against finite differences and
// against the autodiff wrapper.
class TinyOneSidedFocalTangentSampsonErrorCostFunctor {
 public:
  using Scalar = double;
  static constexpr int NUM_RESIDUALS = Eigen::Dynamic;
  static constexpr int NUM_PARAMETERS = 8;

  // ceres::TinySolver-compatible autodiff wrapper for this functor. Note that
  // it stores the functor by reference, so the wrapped functor must outlive it.
  // The solver uses the analytic Jacobian below, so this only serves to pin
  // that Jacobian to a second implementation in tests.
  using AutoDiffFunction = ceres::TinySolverAutoDiffFunction<
      TinyOneSidedFocalTangentSampsonErrorCostFunctor,
      NUM_RESIDUALS,
      NUM_PARAMETERS>;

  TinyOneSidedFocalTangentSampsonErrorCostFunctor(
      const std::vector<Eigen::Vector2d>& img_points1,
      const std::vector<CamRayWithJac>& cam_rays2_with_jac)
      : img_points1_(img_points1), cam_rays2_with_jac_(cam_rays2_with_jac) {}

  int NumResiduals() const { return static_cast<int>(img_points1_.size()); }

  template <typename T>
  bool operator()(const T* const params, T* residuals) const {
    // Build E once from the pose, then apply the unknown focal to the columns
    // to obtain the matrix relating view-1 image points to view-2 rays.
    Eigen::Matrix<T, 3, 3> M = EssentialMatrixFromPoseParams(params);
    M.template leftCols<2>() *= ceres::exp(-params[7]);
    // Measurement Jacobian d(x, y, 1) / d(x, y) of the uncalibrated view.
    Eigen::Matrix<T, 3, 2> J1 = Eigen::Matrix<T, 3, 2>::Zero();
    J1(0, 0) = T(1);
    J1(1, 1) = T(1);
    for (size_t i = 0; i < img_points1_.size(); ++i) {
      residuals[i] =
          TangentSampsonError<T>(M,
                                 img_points1_[i].cast<T>().homogeneous(),
                                 J1,
                                 cam_rays2_with_jac_[i].ray.cast<T>(),
                                 cam_rays2_with_jac_[i].jacobian.cast<T>());
    }
    return true;
  }

  // jacobian is NUM_RESIDUALS x 8, column-major (or null for residuals only).
  bool operator()(const double* params,
                  double* residuals,
                  double* jacobian) const {
    const Eigen::Map<const Eigen::Quaterniond> q(params);
    const Eigen::Matrix3d R = q.toRotationMatrix();
    Eigen::Matrix3d t_x;
    t_x << 0, -params[6], params[5], params[6], 0, -params[4], -params[5],
        params[4], 0;
    const double inv_f1 = std::exp(-params[7]);
    Eigen::Matrix3d M = t_x * R;
    M.leftCols<2>() *= inv_f1;

    Eigen::Matrix3d dM[8];
    if (jacobian != nullptr) {
      const double x = params[0], y = params[1], z = params[2], w = params[3];
      Eigen::Matrix3d dR[4];
      dR[0] << 0, 2 * y, 2 * z, 2 * y, -4 * x, -2 * w, 2 * z, 2 * w,
          -4 * x;  // dR/dqx
      dR[1] << -4 * y, 2 * x, 2 * w, 2 * x, 0, 2 * z, -2 * w, 2 * z,
          -4 * y;  // dR/dqy
      dR[2] << -4 * z, -2 * w, 2 * x, 2 * w, -4 * z, 2 * y, 2 * x, 2 * y,
          0;                                                          // dR/dqz
      dR[3] << 0, -2 * z, 2 * y, 2 * z, 0, -2 * x, -2 * y, 2 * x, 0;  // dR/dqw
      for (int l = 0; l < 4; ++l) dM[l] = t_x * dR[l];
      Eigen::Matrix3d ex, ey, ez;
      ex << 0, 0, 0, 0, 0, -1, 0, 1, 0;
      ey << 0, 0, 1, 0, 0, 0, -1, 0, 0;
      ez << 0, -1, 0, 1, 0, 0, 0, 0, 0;
      dM[4] = ex * R;  // dE/dtx
      dM[5] = ey * R;  // dE/dty
      dM[6] = ez * R;  // dE/dtz
      // The pose derivatives are of E, so carry them through the same column
      // scaling that turns E into M.
      for (int l = 0; l < 7; ++l) dM[l].leftCols<2>() *= inv_f1;
      // d(E * diag(1/f1, 1/f1, 1)) / d(log_f1) negates the scaled columns and
      // leaves the third one, which carries no focal, unchanged.
      dM[7] = -M;
      dM[7].col(2).setZero();
    }

    const int n = static_cast<int>(img_points1_.size());
    for (int i = 0; i < n; ++i) {
      const Eigen::Vector3d point1 = img_points1_[i].homogeneous();
      const Eigen::Vector3d& ray2 = cam_rays2_with_jac_[i].ray;
      const Eigen::Matrix3x2d& J2 = cam_rays2_with_jac_[i].jacobian;
      const Eigen::Vector3d Mpoint1 = M * point1;
      const double num = ray2.dot(Mpoint1);
      // Constraint gradients in view-1 and view-2 pixels. The former needs no
      // Jacobian, as d(x, y, 1)/d(x, y) merely selects the first two rows.
      const Eigen::Vector2d g1 = (M.transpose() * ray2).head<2>();
      const Eigen::Vector2d g2 = J2.transpose() * Mpoint1;
      const double denom = g1.squaredNorm() + g2.squaredNorm();
      const double sqrt_denom = std::sqrt(denom);
      if (sqrt_denom == 0.0) {
        residuals[i] = 0.0;
        if (jacobian != nullptr) {
          for (int l = 0; l < 8; ++l) jacobian[i + l * n] = 0.0;
        }
        continue;
      }
      residuals[i] = num / sqrt_denom;
      if (jacobian != nullptr) {
        // dr/dM = (1/sqrt_denom) ray2 point1^T
        //         - (num/denom^1.5) (ray2 [g1; 0]^T + (J2 g2) point1^T).
        const Eigen::Vector3d J1g1(g1.x(), g1.y(), 0.0);
        const Eigen::Vector3d J2g2 = J2 * g2;
        const double coef = num / (denom * sqrt_denom);
        const Eigen::Matrix3d drdM =
            (1.0 / sqrt_denom) * (ray2 * point1.transpose()) -
            coef * (ray2 * J1g1.transpose() + J2g2 * point1.transpose());
        for (int l = 0; l < 8; ++l) {
          jacobian[i + l * n] = drdM.cwiseProduct(dM[l]).sum();
        }
      }
    }
    return true;
  }

 private:
  const std::vector<Eigen::Vector2d>& img_points1_;
  const std::vector<CamRayWithJac>& cam_rays2_with_jac_;
};

// Sampson-error cost functor for TinySolver refinement of a fundamental matrix,
// parameterized by the SVD factorization of Bartoli and Sturm, "Non-Linear
// Estimation of the Fundamental Matrix With Minimal Parameters", PAMI 2004:
//
//   F = U * diag(1, sigma, 0) * V^T = U0 V0^T + sigma * U1 V1^T,
//
// where U = R(qU), V = R(qV) and Uk, Vk are their k-th columns. The 9 ambient
// parameters are [qU (xyzw), qV (xyzw), sigma]; the solver applies the 7-DoF
// manifold, so rank 2 holds exactly at every iterate. Fixing the largest
// singular value to 1 is free, as the Sampson error is scale invariant.
//
// The inputs are image points and the residual r = num / sqrt(denom) is the
// signed Sampson error whose square ComputeSquaredSampsonError scores. The
// 9-parameter Jacobian is computed in closed form (dr/dF contracted with
// dF/dparams). A unit test checks it against finite differences.
class TinyFundamentalSampsonErrorCostFunctor {
 public:
  using Scalar = double;
  static constexpr int NUM_RESIDUALS = Eigen::Dynamic;
  static constexpr int NUM_PARAMETERS = 9;

  TinyFundamentalSampsonErrorCostFunctor(
      const std::vector<Eigen::Vector2d>& points1,
      const std::vector<Eigen::Vector2d>& points2)
      : points1_(points1), points2_(points2) {}

  int NumResiduals() const { return static_cast<int>(points1_.size()); }

  // The fundamental matrix implied by the 9 ambient parameters. Optionally
  // outputs the two rotations, which only the Jacobian needs.
  static Eigen::Matrix3d FundamentalFromParams(
      const double* params,
      Eigen::Matrix3d* U_out = nullptr,
      Eigen::Matrix3d* V_out = nullptr) {
    const Eigen::Map<const Eigen::Quaterniond> qU(params);
    const Eigen::Map<const Eigen::Quaterniond> qV(params + 4);
    const Eigen::Matrix3d U = qU.toRotationMatrix();
    const Eigen::Matrix3d V = qV.toRotationMatrix();
    if (U_out != nullptr) *U_out = U;
    if (V_out != nullptr) *V_out = V;
    return U.col(0) * V.col(0).transpose() +
           params[8] * U.col(1) * V.col(1).transpose();
  }

  // jacobian is NUM_RESIDUALS x 9, column-major (or null for residuals only).
  bool operator()(const double* params,
                  double* residuals,
                  double* jacobian) const {
    const double sigma = params[8];
    Eigen::Matrix3d U;
    Eigen::Matrix3d V;
    const Eigen::Matrix3d F = FundamentalFromParams(params, &U, &V);

    Eigen::Matrix3d dF[9];
    if (jacobian != nullptr) {
      Eigen::Vector3d dU0[4], dU1[4], dV0[4], dV1[4];
      RotationCol01Derivatives(params, dU0, dU1);
      RotationCol01Derivatives(params + 4, dV0, dV1);
      for (int l = 0; l < 4; ++l) {
        dF[l] = dU0[l] * V.col(0).transpose() +
                sigma * dU1[l] * V.col(1).transpose();
        dF[4 + l] = U.col(0) * dV0[l].transpose() +
                    sigma * U.col(1) * dV1[l].transpose();
      }
      dF[8] = U.col(1) * V.col(1).transpose();
    }

    const int n = static_cast<int>(points1_.size());
    for (int i = 0; i < n; ++i) {
      const Eigen::Vector3d point1 = points1_[i].homogeneous();
      const Eigen::Vector3d point2 = points2_[i].homogeneous();
      const Eigen::Vector3d Fpoint1 = F * point1;
      const Eigen::Vector3d Ftpoint2 = F.transpose() * point2;
      const double num = point2.dot(Fpoint1);
      // Only the first two components of each constraint gradient enter, as
      // the homogeneous third coordinate is not a free variable.
      const double denom =
          Fpoint1.head<2>().squaredNorm() + Ftpoint2.head<2>().squaredNorm();
      const double sqrt_denom = std::sqrt(denom);
      if (sqrt_denom == 0.0) {
        residuals[i] = 0.0;
        if (jacobian != nullptr) {
          for (int l = 0; l < 9; ++l) jacobian[i + l * n] = 0.0;
        }
        continue;
      }
      residuals[i] = num / sqrt_denom;
      if (jacobian != nullptr) {
        // dr/dF = (1/sqrt_denom) point2 point1^T
        //         - (num/denom^1.5) (g2 point1^T + point2 g1^T), where g1 and
        // g2 are F^T point2 and F point1 with the third component dropped.
        const Eigen::Vector3d g1(Ftpoint2.x(), Ftpoint2.y(), 0.0);
        const Eigen::Vector3d g2(Fpoint1.x(), Fpoint1.y(), 0.0);
        const double coef = num / (denom * sqrt_denom);
        const Eigen::Matrix3d drdF =
            (1.0 / sqrt_denom) * (point2 * point1.transpose()) -
            coef * (g2 * point1.transpose() + point2 * g1.transpose());
        for (int l = 0; l < 9; ++l) {
          jacobian[i + l * n] = drdF.cwiseProduct(dF[l]).sum();
        }
      }
    }
    return true;
  }

 private:
  // Derivatives of the first two columns of R(q) w.r.t. the quaternion
  // coefficients, assuming unit q. The factorization uses no other column.
  static void RotationCol01Derivatives(const double* q,
                                       Eigen::Vector3d* dcol0,
                                       Eigen::Vector3d* dcol1) {
    const double x = q[0], y = q[1], z = q[2], w = q[3];
    dcol0[0] << 0, 2 * y, 2 * z;        // dR.col(0)/dqx
    dcol0[1] << -4 * y, 2 * x, -2 * w;  // dR.col(0)/dqy
    dcol0[2] << -4 * z, 2 * w, 2 * x;   // dR.col(0)/dqz
    dcol0[3] << 0, 2 * z, -2 * y;       // dR.col(0)/dqw
    dcol1[0] << 2 * y, -4 * x, 2 * w;   // dR.col(1)/dqx
    dcol1[1] << 2 * x, 0, 2 * z;        // dR.col(1)/dqy
    dcol1[2] << -2 * w, -4 * z, 2 * y;  // dR.col(1)/dqz
    dcol1[3] << -2 * z, 0, 2 * x;       // dR.col(1)/dqw
  }

  const std::vector<Eigen::Vector2d>& points1_;
  const std::vector<Eigen::Vector2d>& points2_;
};

}  // namespace colmap

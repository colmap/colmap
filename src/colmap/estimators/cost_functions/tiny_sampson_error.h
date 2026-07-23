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
      const Eigen::Matrix<double, 3, 2>& J1 = cam_rays1_with_jac_[i].jacobian;
      const Eigen::Matrix<double, 3, 2>& J2 = cam_rays2_with_jac_[i].jacobian;
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
      const Eigen::Matrix<T, 3, 1> point1 = points1_[i].cast<T>().homogeneous();
      const Eigen::Matrix<T, 3, 1> point2 = points2_[i].cast<T>().homogeneous();
      residuals[i] = SampsonError<T>(F, point1, point2);
    }
    return true;
  }

 private:
  const std::vector<Eigen::Vector2d>& points1_;
  const std::vector<Eigen::Vector2d>& points2_;
};

// Epipolar-error cost functor for fixed-size (colmap::TinySolver) refinement of
// a two-view relative pose and a *single* unknown focal length, where the
// second view is already calibrated (the semi-calibrated, or "one-sided focal",
// configuration).
//
// The pose is parameterized as in TinyFocalSampsonErrorCostFunctor
// ([qx, qy, qz, qw, tx, ty, tz]) and the unknown focal of the *first* view is
// appended as an eighth parameter optimized in log-space. The second view needs
// no focal parameter, because it enters as calibrated bearing rays. Following
// ray2^T E b1 = 0 with b1 = K1inv * point1, the mixed epipolar matrix is
//
//   M = E * diag(1/f1, 1/f1, 1),   f1 = exp(log_f1),
//
// i.e. only columns 0-1 are scaled, and ray2^T M point1 = 0.
//
// Unlike its siblings this minimizes the distance from each point to its
// epipolar line, in first-view pixels, rather than a Sampson error. Sampson
// combines the gradients of both views, and here those are in different units
// (pixels and rays), so it would collapse onto the ray-scale term. Matching
// RelativePoseOneSidedFocalEstimator::Residuals matters because LO-RANSAC
// scores the refined model with that residual.
//
// img_points1 are principal-point-centered points of the uncalibrated view.
// cam_rays2 are calibrated bearing rays of the second view, so any distortion
// (or a non-pinhole projection such as a fisheye or spherical model) is already
// undone by the caller and is represented exactly.
class TinyOneSidedFocalEpipolarErrorCostFunctor {
 public:
  using Scalar = double;
  static constexpr int NUM_RESIDUALS = Eigen::Dynamic;
  static constexpr int NUM_PARAMETERS = 8;

  // ceres::TinySolver-compatible autodiff wrapper for this functor. Note that
  // it stores the functor by reference, so the wrapped functor must outlive it.
  using AutoDiffFunction = ceres::TinySolverAutoDiffFunction<
      TinyOneSidedFocalEpipolarErrorCostFunctor,
      NUM_RESIDUALS,
      NUM_PARAMETERS>;

  TinyOneSidedFocalEpipolarErrorCostFunctor(
      const std::vector<Eigen::Vector2d>& img_points1,
      const std::vector<Eigen::Vector3d>& cam_rays2)
      : img_points1_(img_points1), cam_rays2_(cam_rays2) {}

  int NumResiduals() const { return static_cast<int>(img_points1_.size()); }

  template <typename T>
  bool operator()(const T* const params, T* residuals) const {
    // Build E once from the pose, then apply the unknown focal to the columns
    // to obtain the matrix relating view-1 image points to view-2 rays.
    Eigen::Matrix<T, 3, 3> M = EssentialMatrixFromPoseParams(params);
    const T inv_f1 = ceres::exp(-params[7]);
    M.template leftCols<2>() *= inv_f1;
    const Eigen::Matrix<T, 3, 3> M_transpose = M.transpose();
    for (size_t i = 0; i < img_points1_.size(); ++i) {
      const Eigen::Matrix<T, 3, 1> cam_ray2 = cam_rays2_[i].cast<T>();
      const Eigen::Matrix<T, 3, 1> line1 = M_transpose * cam_ray2;
      const T line_norm = line1.template head<2>().norm();
      if (!(line_norm > static_cast<T>(0))) {
        residuals[i] = static_cast<T>(0);
        continue;
      }
      const Eigen::Matrix<T, 3, 1> point1 =
          img_points1_[i].cast<T>().homogeneous();
      residuals[i] = cam_ray2.dot(M * point1) / line_norm;
    }
    return true;
  }

 private:
  const std::vector<Eigen::Vector2d>& img_points1_;
  const std::vector<Eigen::Vector3d>& cam_rays2_;
};

}  // namespace colmap

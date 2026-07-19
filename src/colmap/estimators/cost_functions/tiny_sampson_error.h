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

#include <vector>

#include <Eigen/Core>
#include <ceres/tiny_solver_autodiff_function.h>

namespace colmap {

// Sampson-error cost functor for fixed-size (colmap::TinySolver) refinement of
// a two-view relative pose.
//
// Like SampsonErrorCostFunctor it minimizes the Sampson error of E = [t]_x R,
// parameterized by the full 7-parameter relative pose
// [qx, qy, qz, qw, tx, ty, tz] (Rigid3d::params layout), and shares the same
// EssentialMatrixFromPoseParams / SampsonError helpers. Unlike the per-point
// SampsonErrorCostFunctor it evaluates all correspondences in a single
// (dynamically sized) residual and builds E only once. The rotation-plus-sphere
// manifold is applied by the solver (see tiny_manifold.h), not baked into this
// functor, so the ambient pose is parameterized directly.
class TinySampsonErrorCostFunctor {
 public:
  using Scalar = double;
  static constexpr int NUM_RESIDUALS = Eigen::Dynamic;
  static constexpr int NUM_PARAMETERS = 7;

  // ceres::TinySolver-compatible autodiff wrapper for this functor. Note that
  // it stores the functor by reference, so the wrapped functor must outlive it.
  using AutoDiffFunction =
      ceres::TinySolverAutoDiffFunction<TinySampsonErrorCostFunctor,
                                        NUM_RESIDUALS,
                                        NUM_PARAMETERS>;

  TinySampsonErrorCostFunctor(const std::vector<Eigen::Vector3d>& cam_rays1,
                              const std::vector<Eigen::Vector3d>& cam_rays2)
      : cam_rays1_(cam_rays1), cam_rays2_(cam_rays2) {}

  int NumResiduals() const { return static_cast<int>(cam_rays1_.size()); }

  template <typename T>
  bool operator()(const T* const cam2_from_cam1, T* residuals) const {
    // Build E once and reuse it across all correspondences.
    const Eigen::Matrix<T, 3, 3> E =
        EssentialMatrixFromPoseParams(cam2_from_cam1);
    for (size_t i = 0; i < cam_rays1_.size(); ++i) {
      residuals[i] =
          SampsonError<T>(E, cam_rays1_[i].cast<T>(), cam_rays2_[i].cast<T>());
    }
    return true;
  }

 private:
  const std::vector<Eigen::Vector3d>& cam_rays1_;
  const std::vector<Eigen::Vector3d>& cam_rays2_;
};

// Sampson-error cost functor for fixed-size (colmap::TinySolver) refinement of
// a two-view relative pose *and* a shared, unknown focal length.
//
// This is the shared-focal analog of TinySampsonErrorCostFunctor. The pose is
// parameterized identically ([qx, qy, qz, qw, tx, ty, tz]) and the focal is
// appended as an eighth parameter optimized in log-space (log_f), which keeps
// it strictly positive and gives a scale-invariant step. The essential matrix
// is built from the pose exactly as in TinySampsonErrorCostFunctor, then
// converted to the fundamental matrix implied by the focal so the Sampson error
// is measured in *pixel* space:
//
//   F = diag(1/f, 1/f, 1) * E * diag(1/f, 1/f, 1),   f = exp(log_f).
//
// The inputs points1/points2 are therefore principal-point-centered image
// points (u - cx, v - cy), not calibrated rays. The 8 ambient parameters are
// parameterized directly here; the 6-DoF manifold (rotation on SO(3),
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

}  // namespace colmap

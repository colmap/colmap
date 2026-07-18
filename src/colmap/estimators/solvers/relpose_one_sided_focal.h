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

#include "colmap/util/eigen_alignment.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

// Relative-pose estimator with one unknown focal length, wrapping PoseLib's
// minimal 6-point one-sided focal solver.
//
// The first view's focal is unknown and recovered, the second view's is known.
// Treating such a pair as fully uncalibrated (a plain fundamental matrix) would
// discard the known intrinsics of the calibrated side. Unlike the shared-focal
// solver it has no singularity at coplanar optical axes, as the calibrated view
// fixes the scale, so it needs no identifiability score.
//
// Inputs are asymmetric: X_t are principal-point-centered points of the
// uncalibrated view (u - cx), in pixels, since its focal is the unknown; Y_t
// are calibrated bearing rays of the second view, so any camera model is
// admissible there, including non-pinhole ones and rays with z <= 0. The
// epipolar constraint is ray2^T M point1 = 0 with M = E * diag(1/f1, 1/f1, 1).
//
// Residuals are squared distances to the epipolar line, in first-view pixels.
// A Sampson error is deliberately not used: it combines the gradients of both
// views, which are in different units here, so it would collapse onto the
// ray-scale term and leave a pixel-valued RANSAC threshold meaningless.
//
// The class serves as both the global (minimal) and local (refinement)
// estimator inside LO-RANSAC: the Refine() hook is detected by loransac.h and
// used for local optimization from the current best model. This is required
// because the minimal solver consumes exactly six points and has no non-minimal
// least-squares counterpart.
class RelativePoseOneSidedFocalEstimator {
 public:
  // Principal-point-centered points of the uncalibrated view.
  using X_t = Eigen::Vector2d;
  // Calibrated bearing rays of the second view.
  using Y_t = Eigen::Vector3d;

  // The estimated model: a calibrated essential matrix plus the recovered focal
  // length of the first (uncalibrated) view.
  struct M_t {
    Eigen::Matrix3d E = Eigen::Matrix3d::Identity();
    double focal = 0.0;
  };

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 6;

  // Estimate relative pose and the first view's focal from >= 6 correspondences
  // by wrapping poselib::relpose_6pt_onesided_focal (uses the first six).
  static void Estimate(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& cam_rays2,
                       std::vector<M_t>* models);

  // Nonlinear local optimization of the joint 6-DoF pose plus the unknown
  // focal, in place, starting from *model. This is the entry point used by
  // LO-RANSAC for local optimization (see SupportsRefineWithInitialModel in
  // loransac.h).
  //
  // Returns true and overwrites *model with the refined model on success. On a
  // degenerate decomposition (or non-positive focal) it returns false and
  // leaves *model unchanged.
  static bool Refine(const std::vector<X_t>& points1,
                     const std::vector<Y_t>& cam_rays2,
                     M_t* model);

  // Squared distance, in first-view pixels, from each point to the epipolar
  // line induced by its ray under M = E * K1inv.
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& cam_rays2,
                        const M_t& model,
                        std::vector<double>* residuals);
};

}  // namespace colmap

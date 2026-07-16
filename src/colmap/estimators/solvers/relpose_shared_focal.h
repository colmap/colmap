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

#include "colmap/geometry/rigid3.h"
#include "colmap/util/eigen_alignment.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

// Relative-pose estimator with one unknown, shared focal length, wrapping
// PoseLib's minimal 6-point solver (poselib::relpose_6pt_shared_focal).
//
// Both images are assumed to share a single unknown focal length, e.g. two
// images captured by the same camera without a reliable focal-length prior.
//
// The shared focal is ill-conditioned when the two optical axes are coplanar,
// i.e. intersecting or parallel (turntable or object-scan capture): the known
// singularity of two-view focal recovery (Bougnoux 1998; Kocur et al., CVPR
// 2024). The essential matrix stays well constrained there, but the focal does
// not.
//
// FocalIdentifiability() scores how far a recovered pose is from this singular
// configuration, so callers can detect and reject an unreliable focal. The
// references characterize the condition but give no such measure; the score and
// its threshold are heuristic.
//
// Inputs (X_t/Y_t) are principal-point-centered image points (u - cx, v - cy),
// not calibrated rays. The estimated model bundles the calibrated essential
// matrix with the recovered focal length. Residuals and refinement are measured
// as pixel-space squared Sampson error via
// F = diag(1/f, 1/f, 1) * E * diag(1/f, 1/f, 1).
//
// The class serves as both the global (minimal) and local (refinement)
// estimator inside LO-RANSAC: the Refine() hook is detected by loransac.h and
// used for local optimization from the current best model. This nonlinear
// refinement is required because the minimal solver consumes exactly six points
// and has no non-minimal least-squares counterpart.
class RelativePoseSharedFocalEstimator {
 public:
  // Principal-point-centered image points (u - cx, v - cy).
  using X_t = Eigen::Vector2d;
  using Y_t = Eigen::Vector2d;

  // The estimated model: a calibrated essential matrix plus the shared focal.
  struct M_t {
    Eigen::Matrix3d E = Eigen::Matrix3d::Identity();
    double focal = 0.0;
  };

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 6;

  // Estimate relative pose and shared focal from >= 6 centered point pairs by
  // wrapping poselib::relpose_6pt_shared_focal (uses the first six points).
  static void Estimate(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& points2,
                       std::vector<M_t>* models);

  // Nonlinear local optimization of the joint 6-DoF pose + shared focal, in
  // place, starting from *model. This is the entry point used by LO-RANSAC for
  // local optimization (see SupportsRefineWithInitialModel in loransac.h).
  //
  // Returns true and overwrites *model with the refined model on success. On a
  // degenerate decomposition (or non-positive focal) it returns false and
  // leaves *model unchanged.
  static bool Refine(const std::vector<X_t>& points1,
                     const std::vector<Y_t>& points2,
                     M_t* model);

  // Squared pixel-space Sampson error of each centered point pair under the
  // fundamental matrix implied by the model (F = Kinv * E * Kinv).
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& model,
                        std::vector<double>* residuals);

  // Score in [0, 1]: |b . (a1 x a2)| for unit optical axes a1 = e_z,
  // a2 = R^T e_z and unit baseline b = (-R^T t).normalized(). Zero iff the axes
  // are coplanar, approaching 1 for skew axes; a zero baseline returns 0.
  static double FocalIdentifiability(const Rigid3d& cam2_from_cam1);

  // Minimum FocalIdentifiability() score for the recovered focal to be treated
  // as reliable. It rejects near-coplanar optical axes (turntable/object-scan
  // capture), where two-view focal recovery is ill-conditioned, while admitting
  // clearly skew configurations that constrain the focal. The value 0.05 was
  // chosen empirically and works reasonably in practice.
  static constexpr double kMinFocalIdentifiability = 0.05;
};

}  // namespace colmap

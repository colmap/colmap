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

#include "colmap/sensor/models.h"

#include <limits>
#include <vector>

#include <Eigen/Core>

namespace colmap {

struct RayFittingOptions {
  // Maximum number of Ceres solver iterations for nonlinear refinement.
  // The solver terminates early on convergence; coupled models (e.g. EUCM)
  // need the headroom.
  int max_num_iterations = 50;

  // Maximum number of 2D-3D correspondences used for fitting. Dense fields
  // are stride-subsampled to this size.
  int max_num_points = 16384;

  // Relative weight of the mean squared focal-length prior error compared to
  // the mean squared pixel reprojection error. A value of zero disables the
  // prior.
  double prior_focal_length_weight = 0.0;

  bool Check() const;
};

struct FittedCamera {
  // Fitted parameters for the target model, in the frame of the input
  // image coordinates.
  std::vector<double> params;
  // Whether fitting succeeded (valid initialization and refinement converged).
  bool success = false;
  // Mean squared objective before/after refinement, including the focal-length
  // prior when enabled.
  double initial_cost = std::numeric_limits<double>::infinity();
  double final_cost = std::numeric_limits<double>::infinity();
};

// Fit intrinsics of `model_id` to dense image-point/camera-ray correspondences,
// following AnyCalib (Tirado-Garin & Civera, ICCV 2025): naive initialization
// (focal length from the image span, principal point at the data center, zero
// distortion), then nonlinear refinement with Ceres.
//
// Unlike upstream AnyCalib, which implements one closed-form solver and one
// analytic Jacobian per camera model over tangent-space residuals, the
// refinement minimizes pixel residuals of the projected rays with Ceres
// autodiff directly on any COLMAP perspective model, whose projections are
// all templated for Jets.
//
// Note: an AnyCalib-style closed-form linear init (incl. an equidistant
// variant for fisheye models) was evaluated and deliberately dropped. Stress
// tests over strong distortion (incl. near-pole division), noise, sparsity,
// partial coverage, and outliers showed the refinement reaches the same
// optimum from the naive start every time, so the linear init only saved
// Ceres iterations at the cost of ~270 lines of solver machinery.
//
// Only perspective models are supported; spherical/panoramic models (e.g.
// EQUIRECTANGULAR) return `success == false`.
// If provided, `prior_focal_lengths` must follow the target model's focal
// parameter order and is weighted according to `prior_focal_length_weight`.
// An invalid prior (wrong size or non-positive/non-finite values) is ignored
// with a warning instead of failing the fit.
FittedCamera FitCameraFromRays(
    CameraModelId model_id,
    const std::vector<Eigen::Vector2d>& img_points,
    const std::vector<Eigen::Vector3d>& cam_rays,
    const RayFittingOptions& options,
    const std::vector<double>& prior_focal_lengths = {});

// Uniform stride indices subsampling `num_points` to at most `max_num_points`.
std::vector<size_t> StrideSubsampleIndices(size_t num_points,
                                           size_t max_num_points);

}  // namespace colmap

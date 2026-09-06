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

  // Rays with incidence angles above this field of view (in degrees) are
  // ignored during linear initialization. Matches AnyCalib's default.
  double max_fov_deg = 170.0;

  bool Check() const;
};

struct FittedCamera {
  // Fitted parameters for the target model, in the frame of the input
  // image coordinates.
  std::vector<double> params;
  // Whether fitting succeeded (linear init valid and refinement converged).
  bool success = false;
  // Mean squared pixel residual before/after refinement.
  double initial_cost = std::numeric_limits<double>::infinity();
  double final_cost = std::numeric_limits<double>::infinity();
};

// Fit intrinsics of `model_id` to dense image-point/camera-ray correspondences,
// following AnyCalib (Tirado-Garin & Civera, ICCV 2025): closed-form linear
// initialization, then nonlinear refinement with Ceres.
//
// Unlike upstream AnyCalib, which implements one closed-form solver and one
// analytic Jacobian per camera model over tangent-space residuals, the
// refinement minimizes pixel residuals of the projected rays with Ceres
// autodiff directly on any COLMAP perspective model, whose projections are
// all templated for Jets. Distortion parameters are initialized in closed
// form for radial models and to zero otherwise.
//
// Only perspective models are supported; spherical/panoramic models (e.g.
// EQUIRECTANGULAR) return `success == false`.
FittedCamera FitCameraFromRays(CameraModelId model_id,
                               const std::vector<Eigen::Vector2d>& img_points,
                               const std::vector<Eigen::Vector3d>& cam_rays,
                               const RayFittingOptions& options);

// Map fitted parameters from a resampled frame back to the original image,
// inverting `f' = s * f`, `c' = s * c + t` for focal lengths and principal
// point. Distortion parameters are scale-invariant and left untouched.
// Matches `BaseCamera.reverse_scale_and_shift` in AnyCalib.
std::vector<double> ReverseScaleAndShiftParams(
    CameraModelId model_id,
    const std::vector<double>& params,
    const Eigen::Vector2d& scale_xy,
    const Eigen::Vector2d& shift_xy);

// Uniform stride indices subsampling `num_points` to at most `max_num_points`.
std::vector<size_t> StrideSubsampleIndices(size_t num_points,
                                           size_t max_num_points);

}  // namespace colmap

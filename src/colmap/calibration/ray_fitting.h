// SPDX-License-Identifier: BSD-3-Clause

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

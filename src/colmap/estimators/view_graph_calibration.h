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

#include "colmap/scene/database.h"

#include <memory>

#include <ceres/ceres.h>

namespace colmap {

struct ViewGraphCalibrationOptions {
  // Random seed for RANSAC-based estimation (-1 for random).
  int random_seed = -1;

  // Whether to cross-validate prior focal lengths by checking the ratio of
  // calibrated vs uncalibrated pairs per camera. When enabled, UNCALIBRATED
  // pairs are converted to CALIBRATED if both cameras have reliable priors.
  bool cross_validate_prior_focal_lengths = true;
  // Minimum ratio of calibrated pairs for a camera to be considered valid
  // during cross-validation.
  double min_calibrated_pair_ratio = 0.5;

  // Whether to re-estimate relative poses after focal length calibration.
  bool reestimate_relative_pose = true;

  // The minimum ratio of the estimated focal length to the prior focal length.
  double min_focal_length_ratio = 0.1;
  // The maximum ratio of the estimated focal length to the prior focal length.
  double max_focal_length_ratio = 10;

  // The maximum calibration error for an image pair.
  double max_calibration_error = 2.;

  // Scaling factor for the loss function.
  double loss_function_scale = 0.01;

  // The options for the solver.
  ceres::Solver::Options solver_options;

  // Options for relative pose re-estimation.
  double relpose_max_error = 1.0;
  int relpose_min_num_inliers = 30;
  double relpose_min_inlier_ratio = 0.25;

  ViewGraphCalibrationOptions() {
    solver_options.num_threads = -1;
    solver_options.max_num_iterations = 100;
    solver_options.function_tolerance = 1e-5;
  }

  // Create loss function for given options.
  std::unique_ptr<ceres::LossFunction> CreateLossFunction() const;
};

// Calibrate the view graph by estimating focal lengths from fundamental
// matrices. This function operates directly on the database, reading both
// UNCALIBRATED and CALIBRATED two-view geometries along with their associated
// cameras. It optimizes focal lengths and updates the camera intrinsics in the
// database. Image pairs with low calibration error have their essential
// matrices computed and relative poses re-estimated, then are upgraded to
// CALIBRATED. Pairs with high calibration error are tagged as DEGENERATE_VGC.
bool CalibrateViewGraph(const ViewGraphCalibrationOptions& options,
                        Database* database);

}  // namespace colmap

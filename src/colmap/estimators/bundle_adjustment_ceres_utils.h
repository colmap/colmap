// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/scene/reconstruction.h"

#include <functional>
#include <set>
#include <utility>

#include <ceres/ceres.h>

namespace colmap {

// Convert a Ceres termination type to COLMAP's backend-agnostic enum.
BundleAdjustmentTerminationType CeresTerminationTypeToTerminationType(
    ceres::TerminationType ceres_type);

// Ceres iteration callback that terminates optimization cleanly when
// `check_if_stopped()` returns true.
class CeresCancellationCallback : public ceres::IterationCallback {
 public:
  explicit CeresCancellationCallback(std::function<bool()> check_if_stopped)
      : check_if_stopped_(std::move(check_if_stopped)) {}

  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary&) override {
    return check_if_stopped_ && check_if_stopped_()
               ? ceres::SOLVER_TERMINATE_SUCCESSFULLY
               : ceres::SOLVER_CONTINUE;
  }

 private:
  std::function<bool()> check_if_stopped_;
};

// Apply subset manifolds and constant-block flags for cameras in a Ceres
// bundle adjustment problem according to `options` and `config`.
void ParameterizeCameras(const BundleAdjustmentOptions& options,
                         const BundleAdjustmentConfig& config,
                         const std::set<camera_t>& camera_ids,
                         Reconstruction& reconstruction,
                         ceres::Problem& problem);

// Normalize pose quaternions and apply SE(3) manifolds / constant-block flags
// for frames and non-reference rig sensors in a Ceres bundle adjustment
// problem according to `options` and `config`.
void ParameterizeRigsAndFrames(const BundleAdjustmentOptions& options,
                               const BundleAdjustmentConfig& config,
                               const std::set<image_t>& image_ids,
                               Reconstruction& reconstruction,
                               ceres::Problem& problem);

}  // namespace colmap

// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/base_controller.h"

namespace colmap {

// Class that controls the global bundle adjustment procedure.
class BundleAdjustmentController : public BaseController {
 public:
  BundleAdjustmentController(
      const BundleAdjustmentOptions& ba_options,
      const BundleAdjustmentConfig& ba_config,
      std::shared_ptr<class Reconstruction> reconstruction);

  BundleAdjustmentController(
      const BundleAdjustmentOptions& ba_options,
      const BundleAdjustmentConfig& ba_config,
      const PosePriorBundleAdjustmentOptions& prior_options,
      std::vector<PosePrior> pose_priors,
      std::shared_ptr<class Reconstruction> reconstruction);

  void Run() override;

  const BundleAdjustmentOptions& Options() const { return ba_options_; }
  const BundleAdjustmentConfig& Config() const { return ba_config_; }
  const std::shared_ptr<class Reconstruction>& Reconstruction() const {
    return reconstruction_;
  }
  const std::shared_ptr<BundleAdjustmentSummary>& Summary() const {
    return summary_;
  }

 private:
  BundleAdjustmentOptions ba_options_;
  BundleAdjustmentConfig ba_config_;
  std::optional<PosePriorBundleAdjustmentOptions> prior_options_;
  std::optional<std::vector<PosePrior>> pose_priors_;
  std::shared_ptr<class Reconstruction> reconstruction_;
  std::shared_ptr<BundleAdjustmentSummary> summary_;
};

}  // namespace colmap

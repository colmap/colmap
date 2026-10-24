// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/option_manager.h"
#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/base_controller.h"

namespace colmap {

// Class that controls the global bundle adjustment procedure.
class BundleAdjustmentController : public BaseController {
 public:
  // Configuration to enable pose prior bundle adjustment.
  struct PosePriorConfig {
    std::vector<PosePrior> pose_priors;
    PosePriorBundleAdjustmentOptions options;
  };

  // If `pose_prior_config` is not null, pose prior bundle adjustment is run
  // instead of plain bundle adjustment.
  BundleAdjustmentController(
      const OptionManager& options,
      std::shared_ptr<Reconstruction> reconstruction,
      std::unique_ptr<PosePriorConfig> pose_prior_config = nullptr);

  void Run();

 private:
  const OptionManager& options_;
  std::shared_ptr<Reconstruction> reconstruction_;
  std::unique_ptr<PosePriorConfig> pose_prior_config_;
};

}  // namespace colmap

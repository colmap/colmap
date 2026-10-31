// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/option_manager.h"
#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/base_controller.h"

#include <optional>

namespace colmap {

// Class that controls the global bundle adjustment procedure.
class BundleAdjustmentController : public BaseController {
 public:
  BundleAdjustmentController(
      const OptionManager& options,
      std::shared_ptr<Reconstruction> reconstruction,
      std::optional<BundleAdjustmentPriors> priors = std::nullopt);

  void Run() override;

 private:
  const OptionManager& options_;
  std::shared_ptr<Reconstruction> reconstruction_;
  std::optional<BundleAdjustmentPriors> priors_;
};

}  // namespace colmap

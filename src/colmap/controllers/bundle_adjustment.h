// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/option_manager.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/base_controller.h"

namespace colmap {

// Class that controls the global bundle adjustment procedure.
class BundleAdjustmentController : public BaseController {
 public:
  BundleAdjustmentController(const OptionManager& options,
                             std::shared_ptr<Reconstruction> reconstruction);

  void Run();

 private:
  const OptionManager& options_;
  std::shared_ptr<Reconstruction> reconstruction_;
};

}  // namespace colmap

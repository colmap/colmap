// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/option_manager.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/base_controller.h"

#include <memory>

namespace colmap {

// Class that controls the global photometric bundle adjustment procedure.
class PhotometricBundleAdjustmentController : public BaseController {
 public:
  PhotometricBundleAdjustmentController(
      const OptionManager& options,
      std::shared_ptr<Reconstruction> reconstruction);

  void Run();

 private:
  const OptionManager& options_;
  std::shared_ptr<Reconstruction> reconstruction_;
};

}  // namespace colmap

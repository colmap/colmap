// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/option_manager.h"
#include "colmap/math/math.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/base_controller.h"

#include <cmath>

namespace colmap {

// Options to constrain the global bundle adjustment with the pose priors stored
// in the database, analogous to the pose prior options of the incremental
// mapper (see IncrementalPipelineOptions).
struct BundleAdjustmentPosePriorOptions {
  // Whether to use priors on the camera positions. The priors are read from
  // the database, so the database path must be set.
  bool use_prior_position = false;

  // Whether to use a robust loss on prior camera positions.
  bool use_robust_loss_on_prior_position = false;

  // Threshold on the residual for the robust position prior loss
  // (sqrt of chi2 for 3DOF at 95% = 2.795).
  double prior_position_loss_scale = std::sqrt(kChiSquare95ThreeDof);

  // PRNG seed for the RANSAC alignment of the reconstruction to the priors.
  // Set to -1 for nondeterministic behavior, or a fixed value to make results
  // reproducible.
  int random_seed = -1;

  bool Check() const;
};

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

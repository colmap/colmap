// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/bundle_adjustment.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/scene/database.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

namespace colmap {
BundleAdjustmentController::BundleAdjustmentController(
    const OptionManager& options,
    std::shared_ptr<Reconstruction> reconstruction)
    : options_(options), reconstruction_(std::move(reconstruction)) {}

void BundleAdjustmentController::Run() {
  THROW_CHECK_NOTNULL(reconstruction_);

  LOG_HEADING1("Global bundle adjustment");
  Timer run_timer;
  run_timer.Start();

  if (reconstruction_->NumRegFrames() == 0) {
    LOG(ERROR) << "Need at least one registered frame.";
    return;
  }
  if (CheckIfStopped()) {
    return;
  }

  BundleAdjustmentOptions ba_options = *options_.bundle_adjustment;
  ba_options.check_if_stopped = [this]() { return CheckIfStopped(); };

  BundleAdjustmentPriors priors;
  if (ba_options.pose_prior.use_position) {
    THROW_CHECK(!options_.database_path->empty())
        << "database_path must be specified when bundle adjustment priors are "
           "enabled.";
    THROW_CHECK_FILE_EXISTS(*options_.database_path);
    priors.pose_priors =
        Database::Open(*options_.database_path)->ReadAllPosePriors();
  }

  // Avoid degeneracies in bundle adjustment.
  ObservationManager(*reconstruction_).FilterObservationsWithNegativeDepth();

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reconstruction_->RegImageIds()) {
    ba_config.AddImage(image_id);
  }

  // Run bundle adjustment.
  std::unique_ptr<BundleAdjuster> bundle_adjuster = CreateDefaultBundleAdjuster(
      ba_options, ba_config, *reconstruction_, priors);
  LOG_IF(WARNING,
         ba_options.pose_prior.use_position &&
             !dynamic_cast<const CeresBundleAdjuster&>(*bundle_adjuster)
                  .NormalizedFromMetric()
                  .has_value())
      << "Position priors were enabled, but insufficient valid priors were "
         "usable; running bundle adjustment without position priors.";
  bundle_adjuster->Solve();
  reconstruction_->UpdatePoint3DErrors();

  run_timer.PrintMinutes();
}

}  // namespace colmap

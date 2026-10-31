// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/bundle_adjustment.h"

#include "colmap/scene/database.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

namespace colmap {

BundleAdjustmentController::BundleAdjustmentController(
    const OptionManager& options,
    std::shared_ptr<Reconstruction> reconstruction,
    std::optional<BundleAdjustmentPriors> priors)
    : options_(options),
      reconstruction_(std::move(reconstruction)),
      priors_(std::move(priors)) {}

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
  if (priors_.has_value()) {
    priors = *priors_;
  } else if (ba_options.pose_prior.use_position) {
    THROW_CHECK(options_.database_path != nullptr &&
                !options_.database_path->empty())
        << "database_path must be specified when bundle adjustment priors are "
           "enabled.";
    THROW_CHECK_FILE_EXISTS(*options_.database_path);
    const auto database = Database::Open(*options_.database_path);
    priors.pose_priors = database->ReadAllPosePriors();
  }

  // Avoid degeneracies in bundle adjustment.
  ObservationManager(*reconstruction_).FilterObservationsWithNegativeDepth();

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reconstruction_->RegImageIds()) {
    ba_config.AddImage(image_id);
  }
  // Fixing the gauge with two cameras leads to a more stable optimization
  // with fewer steps as compared to fixing three points.
  // TODO(jsch): Investigate whether it is safe to not fix the gauge at all,
  // as initial experiments show that it is even faster.
  ba_config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  // Run bundle adjustment.
  std::unique_ptr<BundleAdjuster> bundle_adjuster = CreateDefaultBundleAdjuster(
      ba_options, ba_config, *reconstruction_, priors);
  bundle_adjuster->Solve();
  reconstruction_->UpdatePoint3DErrors();

  run_timer.PrintMinutes();
}

}  // namespace colmap

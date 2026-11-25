// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/photometric_bundle_adjustment.h"

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/photometric_bundle_adjustment.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

namespace colmap {

PhotometricBundleAdjustmentController::PhotometricBundleAdjustmentController(
    const OptionManager& options,
    std::shared_ptr<Reconstruction> reconstruction)
    : options_(options), reconstruction_(std::move(reconstruction)) {}

void PhotometricBundleAdjustmentController::Run() {
  THROW_CHECK_NOTNULL(reconstruction_);

  LOG_HEADING1("Global photometric bundle adjustment");
  Timer run_timer;
  run_timer.Start();

  if (reconstruction_->NumRegFrames() == 0) {
    LOG(ERROR) << "Need at least one registered frame.";
    return;
  }
  if (CheckIfStopped()) {
    return;
  }

  // Avoid degeneracies in bundle adjustment.
  ObservationManager(*reconstruction_).FilterObservationsWithNegativeDepth();

  BundleAdjustmentOptions ba_options = *options_.bundle_adjustment;
  ba_options.check_if_stopped = [this]() { return CheckIfStopped(); };

  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reconstruction_->RegImageIds()) {
    ba_config.AddImage(image_id);
  }

  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreatePhotometricBundleAdjuster(ba_options,
                                      *options_.photometric_bundle_adjustment,
                                      ba_config,
                                      *options_.image_path,
                                      *reconstruction_);
  bundle_adjuster->Solve();
  reconstruction_->UpdatePoint3DErrors();

  run_timer.PrintMinutes();
}

}  // namespace colmap

// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/bundle_adjustment.h"

#include "colmap/sfm/observation_manager.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

namespace colmap {

BundleAdjustmentController::BundleAdjustmentController(
    const BundleAdjustmentOptions& ba_options,
    const BundleAdjustmentConfig& ba_config,
    std::shared_ptr<class Reconstruction> reconstruction)
    : ba_options_(ba_options),
      ba_config_(ba_config),
      reconstruction_(std::move(THROW_CHECK_NOTNULL(reconstruction))) {
  THROW_CHECK(ba_options_.Check());
}

BundleAdjustmentController::BundleAdjustmentController(
    const BundleAdjustmentOptions& ba_options,
    const BundleAdjustmentConfig& ba_config,
    const PosePriorBundleAdjustmentOptions& prior_options,
    std::vector<PosePrior> pose_priors,
    std::shared_ptr<class Reconstruction> reconstruction)
    : ba_options_(ba_options),
      ba_config_(ba_config),
      prior_options_(prior_options),
      pose_priors_(std::move(pose_priors)),
      reconstruction_(std::move(THROW_CHECK_NOTNULL(reconstruction))) {
  THROW_CHECK(ba_options_.Check());
  THROW_CHECK(prior_options_->Check());
}

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
  if (ba_config_.NumImages() < 2 && ba_config_.NumVariablePoints() == 0) {
    LOG(WARNING) << "At least two images must be registered for global "
                    "bundle adjustment";
    return;
  }

  // Avoid degeneracies in bundle adjustment.
  ObservationManager(*reconstruction_).FilterObservationsWithNegativeDepth();

  BundleAdjustmentOptions ba_options = ba_options_;
  ba_options.check_if_stopped = [this]() { return CheckIfStopped(); };
  BundleAdjustmentConfig ba_config = ba_config_;

  std::unique_ptr<BundleAdjuster> bundle_adjuster;
  const bool use_pose_prior_ba =
      prior_options_.has_value() && pose_priors_.has_value();
  if (use_pose_prior_ba) {
    bundle_adjuster = CreatePosePriorBundleAdjuster(ba_options,
                                                    *prior_options_,
                                                    ba_config,
                                                    *pose_priors_,
                                                    *reconstruction_);
  } else {
    if (ba_config.FixedGauge() == BundleAdjustmentGauge::UNSPECIFIED &&
        ba_config.NumImages() >= 2) {
      ba_config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
    }
    bundle_adjuster =
        CreateDefaultBundleAdjuster(ba_options, ba_config, *reconstruction_);
  }

  summary_ = bundle_adjuster->Solve();

  reconstruction_->UpdatePoint3DErrors();

  run_timer.PrintMinutes();
}

}  // namespace colmap

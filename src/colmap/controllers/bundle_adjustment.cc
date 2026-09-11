// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

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

// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/bundle_adjustment.h"

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/math/math.h"
#include "colmap/scene/database.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

namespace colmap {
namespace {

// Loads the pose priors from the database, with GPS priors converted to ENU
// coordinates, as done by the incremental mapper through DatabaseCache.
std::vector<PosePrior> ReadPosePriors(
    const std::filesystem::path& database_path) {
  std::vector<PosePrior> pose_priors =
      Database::Open(database_path)->ReadAllPosePriors();
  ConvertPosePriorsToENU(pose_priors);
  return pose_priors;
}

// Prints the distance between the registered images and their prior positions.
void PrintPosePriorPositionErrors(const Reconstruction& reconstruction,
                                  const std::vector<PosePrior>& pose_priors) {
  std::vector<double> errors2;
  errors2.reserve(pose_priors.size());
  for (const PosePrior& pose_prior : pose_priors) {
    if (!pose_prior.HasPosition() ||
        pose_prior.corr_data_id.sensor_id.type != SensorType::CAMERA ||
        !reconstruction.ExistsImage(pose_prior.corr_data_id.id)) {
      continue;
    }
    const Image& image = reconstruction.Image(pose_prior.corr_data_id.id);
    if (!image.HasPose()) {
      continue;
    }
    errors2.push_back(
        (image.ProjectionCenter() - pose_prior.position).squaredNorm());
  }

  LOG(INFO) << "Error w.r.t. prior positions (" << errors2.size()
            << " images):\n"
            << "  - rmse:   " << std::sqrt(Mean(errors2)) << '\n'
            << "  - median: " << std::sqrt(Median(errors2));
}

}  // namespace

bool BundleAdjustmentPosePriorOptions::Check() const {
  CHECK_OPTION_GT(prior_position_loss_scale, 0.);
  CHECK_OPTION_GE(random_seed, -1);
  return true;
}

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

  // Avoid degeneracies in bundle adjustment.
  ObservationManager(*reconstruction_).FilterObservationsWithNegativeDepth();

  BundleAdjustmentOptions ba_options = *options_.bundle_adjustment;
  ba_options.check_if_stopped = [this]() { return CheckIfStopped(); };

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reconstruction_->RegImageIds()) {
    ba_config.AddImage(image_id);
  }

  const BundleAdjustmentPosePriorOptions& pose_prior_options =
      *options_.bundle_adjustment_pose_prior;

  std::vector<PosePrior> pose_priors;
  bool use_prior_position = false;
  if (pose_prior_options.use_prior_position) {
    THROW_CHECK_FILE_EXISTS(*options_.database_path);
    pose_priors = ReadPosePriors(*options_.database_path);
    const size_t num_registered_pose_priors =
        NumRegisteredPosePriors(pose_priors, ba_config);
    use_prior_position = num_registered_pose_priors >= 3;
    if (!use_prior_position) {
      LOG(WARNING) << "Only " << num_registered_pose_priors
                   << " registered images have a pose prior position (at "
                      "least 3 are required). Running a standard bundle "
                      "adjustment without the pose priors.";
    }
  }

  std::unique_ptr<BundleAdjuster> bundle_adjuster;
  if (use_prior_position) {
    PosePriorBundleAdjustmentOptions prior_options;
    if (pose_prior_options.use_robust_loss_on_prior_position) {
      prior_options.ceres->prior_position_loss_function_type =
          CeresLossFunctionType::CAUCHY;
    }
    prior_options.ceres->prior_position_loss_scale =
        pose_prior_options.prior_position_loss_scale;
    prior_options.alignment_ransac_options.random_seed =
        pose_prior_options.random_seed;
    // If the reconstruction cannot be aligned to the priors, the pose prior
    // bundle adjuster falls back to a plain bundle adjustment with fixed gauge.
    bundle_adjuster = CreatePosePriorBundleAdjuster(
        ba_options, prior_options, ba_config, pose_priors, *reconstruction_);
  } else {
    // Fixing the gauge with two cameras leads to a more stable optimization
    // with fewer steps as compared to fixing three points.
    // TODO(jsch): Investigate whether it is safe to not fix the gauge at all,
    // as initial experiments show that it is even faster.
    ba_config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
    bundle_adjuster =
        CreateDefaultBundleAdjuster(ba_options, ba_config, *reconstruction_);
  }

  // Run bundle adjustment.
  bundle_adjuster->Solve();
  reconstruction_->UpdatePoint3DErrors();

  if (use_prior_position) {
    PrintPosePriorPositionErrors(*reconstruction_, pose_priors);
  }

  run_timer.PrintMinutes();
}

}  // namespace colmap

#include "glomap/estimators/bundle_adjustment.h"

#include "colmap/util/logging.h"

namespace glomap {

bool RunBundleAdjustment(const colmap::BundleAdjustmentOptions& options,
                         colmap::Reconstruction& reconstruction) {
  if (reconstruction.NumImages() == 0) {
    LOG(ERROR) << "Number of images = " << reconstruction.NumImages();
    return false;
  }
  if (reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Number of tracks = " << reconstruction.NumPoints3D();
    return false;
  }

  // Set up bundle adjustment config.
  colmap::BundleAdjustmentConfig ba_config;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (image.HasPose()) {
      ba_config.AddImage(image_id);
    }
  }
  // Use TWO_CAMS_FROM_WORLD for deterministic gauge fixing.
  ba_config.FixGauge(colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  auto ba =
      colmap::CreateDefaultBundleAdjuster(options, ba_config, reconstruction);

  ceres::Solver::Summary summary = ba->Solve();

  if (VLOG_IS_ON(2)) {
    LOG(INFO) << summary.FullReport();
  } else {
    LOG(INFO) << summary.BriefReport();
  }

  return summary.IsSolutionUsable();
}

}  // namespace glomap

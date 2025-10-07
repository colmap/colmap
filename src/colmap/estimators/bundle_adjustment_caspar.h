

#pragma once

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/scene/reconstruction.h"

namespace colmap {

std::unique_ptr<BundleAdjuster> CreateDefaultCasparBundleAdjuster(
    BundleAdjustmentOptions options,
    BundleAdjustmentConfig config,
    Reconstruction& reconstruction);

}  // namespace colmap

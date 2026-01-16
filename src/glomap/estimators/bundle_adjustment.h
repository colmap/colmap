#pragma once

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/scene/reconstruction.h"

namespace glomap {

// Run bundle adjustment using colmap's implementation.
// Sets up the problem with all registered images and uses TWO_CAMS_FROM_WORLD
// for deterministic gauge fixing.
bool RunBundleAdjustment(const colmap::BundleAdjustmentOptions& options,
                         colmap::Reconstruction& reconstruction);

}  // namespace glomap

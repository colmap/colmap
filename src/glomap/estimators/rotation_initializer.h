#pragma once

#include "colmap/scene/reconstruction.h"

#include "glomap/scene/types.h"

namespace glomap {

// Initialize rig rotations from per-image rotations.
// Estimates cam_from_rig for cameras with unknown calibration,
// then computes rig_from_world for each frame.
bool InitializeRigRotationsFromImages(
    const std::unordered_map<image_t, Rigid3d>& cams_from_world,
    colmap::Reconstruction& reconstruction);

}  // namespace glomap
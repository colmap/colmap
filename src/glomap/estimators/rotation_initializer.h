#pragma once

#include "colmap/scene/reconstruction.h"

#include "glomap/scene/types.h"

namespace glomap {

// Initialize the rotations of the rigs from the images
bool ConvertRotationsFromImageToRig(
    const std::unordered_map<image_t, Rigid3d>& cam_from_worlds,
    colmap::Reconstruction& reconstruction);

}  // namespace glomap
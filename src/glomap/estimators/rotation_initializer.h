#pragma once

#include "glomap/scene/types_sfm.h"

namespace glomap {

// Initialize the rotations of the rigs from the images
bool ConvertRotationsFromImageToRig(
    const std::unordered_map<image_t, Rigid3d>& cam_from_worlds,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<frame_t, Frame>& frames);

}  // namespace glomap
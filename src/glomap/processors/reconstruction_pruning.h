
#pragma once

#include "glomap/scene/types_sfm.h"

namespace glomap {

image_t PruneWeaklyConnectedImages(
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& tracks,
    int min_num_images = 2,
    int min_num_observations = 0);

}  // namespace glomap

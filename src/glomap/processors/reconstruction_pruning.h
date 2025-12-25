#pragma once

#include "colmap/scene/frame.h"

#include "glomap/scene/types.h"

namespace glomap {

// Prunes weakly connected images based on track co-visibility.
// Outputs cluster_ids for each frame.
void PruneWeaklyConnectedImages(std::unordered_map<frame_t, Frame>& frames,
                                std::unordered_map<image_t, Image>& images,
                                std::unordered_map<point3D_t, Point3D>& tracks,
                                std::unordered_map<frame_t, int>& cluster_ids,
                                int min_num_images = 2,
                                int min_num_observations = 0);

}  // namespace glomap

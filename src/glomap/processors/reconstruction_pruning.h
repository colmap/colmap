#pragma once

#include "colmap/scene/reconstruction.h"

#include "glomap/scene/types.h"

namespace glomap {

// Prunes weakly connected images based on track co-visibility.
// Outputs cluster_ids for each frame.
void PruneWeaklyConnectedImages(colmap::Reconstruction& reconstruction,
                                std::unordered_map<frame_t, int>& cluster_ids,
                                int min_num_images = 2,
                                int min_num_observations = 0);

}  // namespace glomap

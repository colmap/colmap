#pragma once

#include "glomap/scene/types_sfm.h"

#include "colmap/geometry/pose.h"

namespace glomap {

colmap::Sim3d NormalizeReconstruction(
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    bool fixed_scale = false,
    double extent = 10.,
    double p0 = 0.1,
    double p1 = 0.9);

}  // namespace glomap

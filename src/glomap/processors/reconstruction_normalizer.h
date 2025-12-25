#pragma once

#include "colmap/geometry/pose.h"
#include "colmap/scene/rig.h"
#include "colmap/sensor/models.h"

#include "colmap/scene/frame.h"
#include "glomap/scene/types.h"

namespace glomap {

colmap::Sim3d NormalizeReconstruction(
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<camera_t, colmap::Camera>& cameras,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& tracks,
    bool fixed_scale = false,
    double extent = 10.,
    double p0 = 0.1,
    double p1 = 0.9);

}  // namespace glomap

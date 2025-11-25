#pragma once

#include "glomap/io/colmap_converter.h"
#include "glomap/scene/types_sfm.h"

namespace glomap {

void WriteGlomapReconstruction(
    const std::string& reconstruction_path,
    const std::unordered_map<rig_t, Rig>& rigs,
    const std::unordered_map<camera_t, colmap::Camera>& cameras,
    const std::unordered_map<frame_t, Frame>& frames,
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<point3D_t, colmap::Point3D>& tracks,
    const std::string output_format = "bin",
    const std::string image_path = "");

void WriteColmapReconstruction(const std::string& reconstruction_path,
                               const colmap::Reconstruction& reconstruction,
                               const std::string output_format = "bin");

}  // namespace glomap

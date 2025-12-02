
#pragma once

#include "colmap/scene/database.h"

#include "glomap/scene/types_sfm.h"

namespace glomap {

struct TriangulatorOptions {
  double tri_complete_max_reproj_error = 15.0;
  double tri_merge_max_reproj_error = 15.0;
  double tri_min_angle = 1.0;

  int min_num_matches = 15;
};

bool RetriangulateTracks(const TriangulatorOptions& options,
                         const colmap::Database& database,
                         std::unordered_map<rig_t, Rig>& rigs,
                         std::unordered_map<camera_t, colmap::Camera>& cameras,
                         std::unordered_map<frame_t, Frame>& frames,
                         std::unordered_map<image_t, Image>& images,
                         std::unordered_map<point3D_t, Point3D>& tracks);

}  // namespace glomap

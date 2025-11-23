#pragma once

#include "glomap/scene/types_sfm.h"

namespace glomap {

struct TrackFilter {
  static int FilterTracksByReprojection(
      const ViewGraph& view_graph,
      const std::unordered_map<camera_t, Camera>& cameras,
      const std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks,
      double max_reprojection_error = 1e-2,
      bool in_normalized_image = true);

  static int FilterTracksByAngle(
      const ViewGraph& view_graph,
      const std::unordered_map<camera_t, Camera>& cameras,
      const std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks,
      double max_angle_error = 1.);

  static int FilterTrackTriangulationAngle(
      const ViewGraph& view_graph,
      const std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks,
      double min_angle = 1.);
};

}  // namespace glomap

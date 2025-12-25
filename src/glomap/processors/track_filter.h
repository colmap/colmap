#pragma once

#include "colmap/sensor/models.h"

#include "glomap/scene/view_graph.h"

namespace glomap {

struct TrackFilter {
  static int FilterTracksByReprojection(
      const ViewGraph& view_graph,
      const std::unordered_map<camera_t, colmap::Camera>& cameras,
      const std::unordered_map<image_t, Image>& images,
      std::unordered_map<point3D_t, Point3D>& tracks,
      double max_reprojection_error = 1e-2,
      bool in_normalized_image = true);

  static int FilterTracksByAngle(
      const ViewGraph& view_graph,
      const std::unordered_map<image_t, Image>& images,
      std::unordered_map<point3D_t, Point3D>& tracks,
      double max_angle_error = 1.);

  static int FilterTrackTriangulationAngle(
      const ViewGraph& view_graph,
      const std::unordered_map<image_t, Image>& images,
      std::unordered_map<point3D_t, Point3D>& tracks,
      double min_angle = 1.);
};

}  // namespace glomap

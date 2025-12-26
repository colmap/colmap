#pragma once

#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/models.h"

#include "glomap/scene/view_graph.h"

namespace glomap {

struct TrackFilter {
  static int FilterTracksByReprojection(const ViewGraph& view_graph,
                                        colmap::Reconstruction& reconstruction,
                                        double max_reprojection_error = 1e-2,
                                        bool in_normalized_image = true);

  static int FilterTracksByAngle(const ViewGraph& view_graph,
                                 colmap::Reconstruction& reconstruction,
                                 double max_angle_error = 1.);

  static int FilterTrackTriangulationAngle(
      const ViewGraph& view_graph,
      colmap::Reconstruction& reconstruction,
      double min_angle = 1.);
};

}  // namespace glomap

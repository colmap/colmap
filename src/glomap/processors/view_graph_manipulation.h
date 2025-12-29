#pragma once

#include "colmap/scene/frame.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/models.h"

#include "glomap/scene/view_graph.h"

namespace glomap {

struct ViewGraphManipulator {
  static void UpdateImagePairsConfig(
      ViewGraph& view_graph, const colmap::Reconstruction& reconstruction);

  // Decompose the relative camera postion from the camera config
  static void DecomposeRelPose(ViewGraph& view_graph,
                               colmap::Reconstruction& reconstruction,
                               int num_threads = -1);
};

}  // namespace glomap

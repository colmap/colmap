#pragma once

#include "colmap/scene/frame.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/models.h"

#include "glomap/scene/view_graph.h"

namespace glomap {

struct ViewGraphManipulator {
  // Decompose relative poses from the two-view geometry matrices.
  static void DecomposeRelativePoses(ViewGraph& view_graph,
                                     colmap::Reconstruction& reconstruction,
                                     int num_threads = -1);
};

}  // namespace glomap

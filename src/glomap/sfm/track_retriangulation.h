#pragma once

#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"

namespace glomap {

struct TriangulatorOptions {
  double tri_complete_max_reproj_error = 15.0;
  double tri_merge_max_reproj_error = 15.0;
  double tri_min_angle = 1.0;

  int min_num_matches = 15;
};

bool RetriangulateTracks(const TriangulatorOptions& options,
                         const colmap::Database& database,
                         colmap::Reconstruction& reconstruction);

}  // namespace glomap

#pragma once

#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/models.h"

#include "glomap/scene/view_graph.h"

#include <PoseLib/types.h>

namespace glomap {

struct RelativePoseEstimationOptions {
  // Number of threads.
  int num_threads = -1;

  // Options for poselib solver
  poselib::RansacOptions ransac_options;
  poselib::BundleOptions bundle_options;

  // PRNG seed for RANSAC. If -1 (default), uses non-deterministic seeding.
  // If >= 0, uses deterministic seeding with the given value.
  int random_seed = -1;

  RelativePoseEstimationOptions() { ransac_options.max_iterations = 50000; }
};

void EstimateRelativePoses(ViewGraph& view_graph,
                           colmap::Reconstruction& reconstruction,
                           const RelativePoseEstimationOptions& options);

}  // namespace glomap

#pragma once

#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/models.h"

#include "glomap/scene/view_graph.h"

#include <PoseLib/types.h>

namespace glomap {

struct RelativePoseEstimationOptions {
  // Options for poselib solver
  poselib::RansacOptions ransac_options;
  poselib::BundleOptions bundle_options;

  RelativePoseEstimationOptions() { ransac_options.max_iterations = 50000; }
};

void EstimateRelativePoses(ViewGraph& view_graph,
                           colmap::Reconstruction& reconstruction,
                           const RelativePoseEstimationOptions& options);

}  // namespace glomap

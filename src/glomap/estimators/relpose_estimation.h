#pragma once

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

void EstimateRelativePoses(
    ViewGraph& view_graph,
    std::unordered_map<camera_t, colmap::Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    const RelativePoseEstimationOptions& options);

}  // namespace glomap

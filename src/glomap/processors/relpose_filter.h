#pragma once

#include "colmap/scene/reconstruction.h"

#include "glomap/scene/view_graph.h"

namespace glomap {

struct RelPoseFilter {
  // Filter relative pose based on rotation angle.
  static void FilterRotations(ViewGraph& view_graph,
                              const colmap::Reconstruction& reconstruction,
                              double max_angle_deg = 5.0);

  // Filter relative pose based on number of inliers
  // min_inlier_num: in degree
  static void FilterInlierNum(ViewGraph& view_graph, int min_inlier_num = 30);

  // Filter relative pose based on rate of inliers
  // min_weight: minimal ratio of inliers
  static void FilterInlierRatio(ViewGraph& view_graph,
                                double min_inlier_ratio = 0.25);
};

}  // namespace glomap

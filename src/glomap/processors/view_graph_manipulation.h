#pragma once

#include "colmap/scene/frame.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/models.h"

#include "glomap/scene/view_graph.h"

namespace glomap {

struct ViewGraphManipulater {
  enum StrongClusterCriteria {
    INLIER_NUM,
    WEIGHT,
  };

  // Sparsifies the view graph by probabilistically removing edges.
  // @param random_seed If >= 0, uses deterministic seeding; if < 0, uses
  //                    random_device (non-deterministic).
  static image_pair_t SparsifyGraph(ViewGraph& view_graph,
                                    colmap::Reconstruction& reconstruction,
                                    int expected_degree = 50,
                                    int random_seed = -1);

  static image_t EstablishStrongClusters(
      ViewGraph& view_graph,
      colmap::Reconstruction& reconstruction,
      std::unordered_map<frame_t, int>& cluster_ids,
      StrongClusterCriteria criteria = INLIER_NUM,
      double min_thres = 100,  // require strong edges
      int min_num_images = 2);

  static void UpdateImagePairsConfig(
      ViewGraph& view_graph, const colmap::Reconstruction& reconstruction);

  // Decompose the relative camera postion from the camera config
  static void DecomposeRelPose(ViewGraph& view_graph,
                               colmap::Reconstruction& reconstruction);
};

}  // namespace glomap

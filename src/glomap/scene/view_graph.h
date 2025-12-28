#pragma once

#include "colmap/scene/reconstruction.h"

#include "glomap/scene/image_pair.h"
#include "glomap/scene/types.h"

#include <unordered_map>
#include <unordered_set>

namespace glomap {

struct ViewGraph {
  // Image pairs indexed by normalized pair_id (smaller image ID first).
  // All methods below normalize IDs and invert geometry as needed.
  std::unordered_map<image_pair_t, ImagePair> image_pairs;

  // Image pair operations.
  ImagePair& AddImagePair(image_t image_id1,
                          image_t image_id2,
                          ImagePair image_pair);
  bool HasImagePair(image_t image_id1, image_t image_id2) const;
  std::pair<ImagePair&, bool> Pair(image_t image_id1, image_t image_id2);
  std::pair<const ImagePair&, bool> Pair(image_t image_id1,
                                         image_t image_id2) const;
  ImagePair GetImagePair(image_t image_id1, image_t image_id2) const;
  bool DeleteImagePair(image_t image_id1, image_t image_id2);
  void UpdateImagePair(image_t image_id1,
                       image_t image_id2,
                       ImagePair image_pair);

  // Create the adjacency list for the images in the view graph.
  std::unordered_map<image_t, std::unordered_set<image_t>>
  CreateImageAdjacencyList() const;

  // Create the adjacency list for the frames in the view graph.
  std::unordered_map<frame_t, std::unordered_set<frame_t>>
  CreateFrameAdjacencyList(
      const std::unordered_map<image_t, colmap::Image>& images) const;

  // Mark the images which are not connected to any other images as not
  // registered. Returns the number of images in the largest connected
  // component.
  int KeepLargestConnectedComponents(colmap::Reconstruction& reconstruction);

  // Mark connected clusters of images, where the cluster_id is sorted by the
  // the number of images. Populates `cluster_ids` output parameter.
  int MarkConnectedComponents(const colmap::Reconstruction& reconstruction,
                              std::unordered_map<frame_t, int>& cluster_ids,
                              int min_num_images = -1);

  // Mark image pairs as invalid if their relative rotation differs from the
  // reconstructed rotation by more than max_angle_deg.
  // Keeps existing invalid edges as invalid.
  void FilterByRelativeRotation(const colmap::Reconstruction& reconstruction,
                                double max_angle_deg = 5.0);

  // Mark image pairs as invalid if they have fewer than min_num_inliers.
  // Keeps existing invalid edges as invalid.
  void FilterByNumInliers(int min_num_inliers = 30);

  // Mark image pairs as invalid if their inlier ratio is below
  // min_inlier_ratio.
  // Keeps existing invalid edges as invalid.
  void FilterByInlierRatio(double min_inlier_ratio = 0.25);
};

}  // namespace glomap

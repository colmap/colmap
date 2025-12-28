#pragma once

#include "colmap/scene/image.h"

#include "glomap/scene/view_graph.h"

namespace glomap {

enum class WeightType { INLIER_NUM, INLIER_RATIO };

// Compute maximum spanning tree of the view graph.
//
// Returns the root image_id and populates the parents map where:
// - parents[child_id] = parent_id for each image in the tree
// - parents[root_id] = root_id for the root
//
// Only images with poses are included in the tree.
// The weight type determines how edge weights are computed:
// - INLIER_NUM: number of inlier matches
// - INLIER_RATIO: weight field from ImagePair
image_t MaximumSpanningTree(const ViewGraph& view_graph,
                            const std::unordered_map<image_t, Image>& images,
                            std::unordered_map<image_t, image_t>& parents,
                            WeightType type);

}  // namespace glomap

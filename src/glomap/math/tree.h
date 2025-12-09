#pragma once

#include "glomap/scene/image.h"
#include "glomap/scene/view_graph.h"

namespace glomap {

enum class WeightType { INLIER_NUM, INLIER_RATIO };

image_t MaximumSpanningTree(const ViewGraph& view_graph,
                            const std::unordered_map<image_t, Image>& images,
                            std::unordered_map<image_t, image_t>& parents,
                            WeightType type);
}  // namespace glomap

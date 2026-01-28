#pragma once

#include "colmap/scene/pose_graph.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

using PoseGraphEdgeMap =
    std::unordered_map<colmap::image_pair_t, colmap::PoseGraph::Edge>;
PYBIND11_MAKE_OPAQUE(PoseGraphEdgeMap);

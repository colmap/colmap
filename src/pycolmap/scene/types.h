#pragma once

#include "colmap/scene/camera.h"
#include "colmap/scene/frame.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point2d.h"
#include "colmap/scene/point3d.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/types.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// The scene element stores use colmap::NodeHashMap, whose concrete type depends
// on the selected hash map backend (see colmap/util/hash_containers.h). Alias
// the opaque bound map types to match so the Reconstruction/DatabaseCache
// accessors bind correctly under any backend.
using RigMap = colmap::NodeHashMap<colmap::rig_t, colmap::Rig>;
PYBIND11_MAKE_OPAQUE(RigMap);

using CameraMap = colmap::NodeHashMap<colmap::camera_t, colmap::Camera>;
PYBIND11_MAKE_OPAQUE(CameraMap);

using FrameMap = colmap::NodeHashMap<colmap::frame_t, colmap::Frame>;
PYBIND11_MAKE_OPAQUE(FrameMap);

using ImageMap = colmap::NodeHashMap<colmap::image_t, colmap::Image>;
PYBIND11_MAKE_OPAQUE(ImageMap);

using Point2DVector = std::vector<struct colmap::Point2D>;
PYBIND11_MAKE_OPAQUE(Point2DVector);

using Point3DMap = colmap::NodeHashMap<colmap::point3D_t, colmap::Point3D>;
PYBIND11_MAKE_OPAQUE(Point3DMap);

using PoseGraphEdgeMap =
    colmap::NodeHashMap<colmap::image_pair_t, colmap::PoseGraph::Edge>;
PYBIND11_MAKE_OPAQUE(PoseGraphEdgeMap);

// Generic caster for non-opaque NodeHashMap returns (e.g.
// CorrespondenceGraph::NumMatchesBetweenAllImages). Only needed for the BOOST
// backend; for STD the alias is std::unordered_map, handled by pybind11. Kept
// here next to the PYBIND11_MAKE_OPAQUE element-store aliases above so that the
// opaque full specializations always take precedence over this partial one (the
// flat casters, which never collide with opaque types, live in
// pycolmap/pybind11_extension.h). Requires <pybind11/stl.h> for map_caster.
#if defined(COLMAP_HASH_BOOST)
namespace pybind11 {
namespace detail {

template <typename Key, typename Value, typename Hash, typename Equal>
struct type_caster<colmap::NodeHashMap<Key, Value, Hash, Equal>>
    : map_caster<colmap::NodeHashMap<Key, Value, Hash, Equal>, Key, Value> {};

}  // namespace detail
}  // namespace pybind11
#endif

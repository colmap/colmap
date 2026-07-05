#pragma once

#include "colmap/scene/camera.h"
#include "colmap/scene/frame.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point2d.h"
#include "colmap/scene/point3d.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/util/containers.h"
#include "colmap/util/types.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// The scene element stores use colmap::NodeHashMap, whose concrete type depends
// on the selected hash map backend (see colmap/util/containers.h). Alias the
// opaque bound map types to match so the Reconstruction/DatabaseCache accessors
// bind correctly under any backend.
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

// pybind11's built-in STL casters only recognize std:: containers, so provide
// casters for the colmap flat/node hash aliases used by non-opaque bindings
// (e.g. ObservationManager::ImagePairs, IncrementalMapper::FilteredFrames,
// BundleAdjustmentConfig::VariablePoints, the FilterPoints3D* parameters).
// These are only needed when an alias resolves to a non-std container; for the
// STD backend the aliases are std:: types already covered by pybind11.
#if defined(COLMAP_HASH_BOOST) || defined(COLMAP_HASH_ABSL) || \
    defined(COLMAP_HASH_ANKERL)
namespace pybind11 {
namespace detail {

template <typename Key, typename Value, typename Hash, typename Equal>
struct type_caster<colmap::FlatHashMap<Key, Value, Hash, Equal>>
    : map_caster<colmap::FlatHashMap<Key, Value, Hash, Equal>, Key, Value> {};

template <typename Key, typename Hash, typename Equal>
struct type_caster<colmap::FlatHashSet<Key, Hash, Equal>>
    : set_caster<colmap::FlatHashSet<Key, Hash, Equal>, Key> {};

}  // namespace detail
}  // namespace pybind11
#endif

// boost/abseil provide distinct node-based containers; ankerl has no node
// variant so its Node* aliases resolve to std:: (already handled by pybind11).
// The opaque bound maps (RigMap/CameraMap/... via PYBIND11_MAKE_OPAQUE above)
// are full specializations that take precedence over these generic casters.
#if defined(COLMAP_HASH_BOOST) || defined(COLMAP_HASH_ABSL)
namespace pybind11 {
namespace detail {

template <typename Key, typename Value, typename Hash, typename Equal>
struct type_caster<colmap::NodeHashMap<Key, Value, Hash, Equal>>
    : map_caster<colmap::NodeHashMap<Key, Value, Hash, Equal>, Key, Value> {};

}  // namespace detail
}  // namespace pybind11
#endif

#pragma once

#include "colmap/scene/camera.h"
#include "colmap/scene/frame.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point2d.h"
#include "colmap/scene/point3d.h"
#include "colmap/util/types.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using RigMap = std::unordered_map<colmap::rig_t, colmap::Rig>;
PYBIND11_MAKE_OPAQUE(RigMap);

using CameraMap = std::unordered_map<colmap::camera_t, colmap::Camera>;
PYBIND11_MAKE_OPAQUE(CameraMap);

using FrameMap = std::unordered_map<colmap::frame_t, colmap::Frame>;
PYBIND11_MAKE_OPAQUE(FrameMap);

using ImageMap = std::unordered_map<colmap::image_t, colmap::Image>;
PYBIND11_MAKE_OPAQUE(ImageMap);

using Point2DVector = std::vector<struct colmap::Point2D>;
PYBIND11_MAKE_OPAQUE(Point2DVector);

using Point3DMap = std::unordered_map<colmap::point3D_t, colmap::Point3D>;
PYBIND11_MAKE_OPAQUE(Point3DMap);

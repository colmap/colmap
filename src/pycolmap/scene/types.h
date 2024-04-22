#pragma once

#include "colmap/scene/camera.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point2d.h"
#include "colmap/scene/point3d.h"
#include "colmap/util/types.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace colmap;
namespace py = pybind11;

using Point2DVector = std::vector<struct Point2D>;
PYBIND11_MAKE_OPAQUE(Point2DVector);

using ImageMap = std::unordered_map<image_t, Image>;
PYBIND11_MAKE_OPAQUE(ImageMap);

using CameraMap = std::unordered_map<camera_t, Camera>;
PYBIND11_MAKE_OPAQUE(CameraMap);

using Point3DMap = std::unordered_map<point3D_t, Point3D>;
PYBIND11_MAKE_OPAQUE(Point3DMap);

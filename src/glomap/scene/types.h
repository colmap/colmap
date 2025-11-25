#pragma once

#include "colmap/geometry/rigid3.h"
// #include "colmap/scene/camera_rig.h"
#include "colmap/scene/database.h"
#include "colmap/util/types.h"

#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace glomap {

////////////////////////////////////////////////////////////////////////////////
// Index types, determines the maximum number of objects.
////////////////////////////////////////////////////////////////////////////////

// Unique identifier for cameras.
using colmap::camera_t;

// Unique identifier for images.
using colmap::image_t;

// Unique identifier for frames.
using colmap::frame_t;

// Unique identifier for camera rigs.
using colmap::rig_t;

// Each image pair gets a unique ID, see `Database::ImagePairToPairId`.
using colmap::image_pair_t;
using colmap::kInvalidImagePairId;

// Unique identifier per added 3D point. Since we add many 3D points,
// delete them, and possibly re-add them again, the maximum number of allowed
// unique indices should be large.
using colmap::point3D_t;

using colmap::Rigid3d;

// Unique identifier for sensors, which can be cameras or IMUs.
using colmap::sensor_t;

// Sensor type, used to identify the type of sensor (e.g., camera, IMU).
using colmap::SensorType;

// Unique identifier for sensor data
using colmap::data_t;

// Rig
using colmap::Rig;

}  // namespace glomap

// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/scene/camera.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"

#include <filesystem>
#include <optional>
#include <vector>

namespace colmap {

// Read the rig configuration from a .json file.
// For each rig, the configuration specifies a list of cameras with exactly one
// camera specified as the reference sensor as well as all cameras with a
// specified image prefix. All images with the given name prefix will be
// associated with the camera. In addition, each camera may specify an optional
// known pose in the rig - except for the reference camera whose pose is defined
// as identity. The rotation is expected in the order [w, x, y, z]. Furthermore,
// each camera may specify a custom camera model and parameters.
//
// Example for eth3d/delivery_area:
// [
//   {
//     "cameras": [
//       {
//           "image_prefix": "images_rig_cam4_undistorted/",
//           "ref_sensor": true
//       },
//       {
//           "image_prefix": "images_rig_cam5_undistorted/"
//       },
//       {
//           "image_prefix": "images_rig_cam6_undistorted/"
//       },
//       {
//           "image_prefix": "images_rig_cam7_undistorted/"
//       }
//     ]
//   }
// ]
//
// Example for GoPro cubemaps:
// [
//   {
//     "cameras": [
//         {
//             "image_prefix": "0/",
//             "ref_sensor": true
//         },
//         {
//             "image_prefix": "1/",
//             "cam_from_rig_rotation": [
//                 0.7071067811865475,
//                 0.0,
//                 0.7071067811865476,
//                 0.0
//             ],
//             "cam_from_rig_translation": [
//                 0,
//                 0,
//                 0
//             ]
//         },
//         {
//             "image_prefix": "2/",
//             "cam_from_rig_rotation": [
//                 0.0,
//                 0.0,
//                 1.0,
//                 0.0
//             ],
//             "cam_from_rig_translation": [
//                 0,
//                 0,
//                 0
//             ]
//         },
//         ...
//     ]
//   }
// ]
struct RigConfig {
  struct RigCamera {
    bool ref_sensor = false;
    std::string image_prefix;
    std::optional<Rigid3d> cam_from_rig;
    std::optional<Camera> camera;
  };
  std::vector<RigCamera> cameras;
};
std::vector<RigConfig> ReadRigConfig(const std::filesystem::path& path);

// Applies the given rig configuration to the database and optionally derives
// camera rig extrinsics and intrinsics from the reconstruction, if not defined
// in the config. If the reconstruction is provided, it is also updated with the
// provided config and any previous rigs/frames are cleared and overwritten.
// Existing rigs and frames in the database and reconstruction will be cleared.
// Any unspecified images in the provided configurations will be configured as
// trivial rigs/frames.
void ApplyRigConfig(const std::vector<RigConfig>& configs,
                    Database& database,
                    Reconstruction* reconstruction = nullptr);

}  // namespace colmap

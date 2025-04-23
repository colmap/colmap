// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/scene/camera.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/types.h"

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
std::vector<RigConfig> ReadRigConfig(const std::string& path);

// Applies the given rig configuration to the database and optionally derives
// camera rig extrinsics and intrinsics from the reconstruction, if not defined
// in the config. If the reconstruction is provided, it is also updated with the
// provided config and any previous rigs/frames are cleared and overwritten.
void ApplyRigConfig(const std::vector<RigConfig>& configs,
                    Database& database,
                    Reconstruction* reconstruction = nullptr);

}  // namespace colmap

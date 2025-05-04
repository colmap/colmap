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

#include "colmap/scene/reconstruction_io_utils.h"

#include <fstream>

namespace colmap {

void CreateOneRigPerCamera(Reconstruction& reconstruction) {
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    Rig rig;
    rig.SetRigId(camera_id);
    rig.AddRefSensor(camera.SensorId());
    reconstruction.AddRig(std::move(rig));
  }
}

void CreateFrameForImage(const Image& image,
                         const Rigid3d& cam_from_world,
                         Reconstruction& reconstruction) {
  Frame frame;
  frame.SetFrameId(image.ImageId());
  frame.SetRigId(image.CameraId());
  frame.AddDataId(image.DataId());
  frame.SetRigFromWorld(cam_from_world);
  reconstruction.AddFrame(std::move(frame));
}

std::unordered_map<image_t, Frame*> ExtractImageToFramePtr(
    Reconstruction& reconstruction) {
  std::unordered_map<image_t, Frame*> image_to_frame;
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    for (const data_t& data_id : frame.DataIds()) {
      if (data_id.sensor_id.type == SensorType::CAMERA) {
        THROW_CHECK(
            image_to_frame.emplace(data_id.id, &reconstruction.Frame(frame_id))
                .second);
      }
    }
  }
  return image_to_frame;
}

}  // namespace colmap

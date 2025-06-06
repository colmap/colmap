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

#include "colmap/scene/frame.h"

namespace colmap {

void Frame::SetCamFromWorld(camera_t camera_id, const Rigid3d& cam_from_world) {
  THROW_CHECK_NOTNULL(rig_ptr_);
  const sensor_t sensor_id(SensorType::CAMERA, camera_id);
  if (rig_ptr_->IsRefSensor(sensor_id)) {
    SetRigFromWorld(cam_from_world);
  } else {
    const Rigid3d& cam_from_rig = rig_ptr_->SensorFromRig(sensor_id);
    SetRigFromWorld(Inverse(cam_from_rig) * cam_from_world);
  }
}

std::ostream& operator<<(std::ostream& stream, const Frame& frame) {
  stream << "Frame(frame_id=" << frame.FrameId() << ", rig_id=";
  if (frame.HasRigId()) {
    if (frame.RigId() == kInvalidRigId) {
      stream << "Invalid";
    } else {
      stream << frame.RigId();
    }
  } else {
    stream << "Unknown";
  }
  stream << ", has_pose=" << frame.HasPose() << ", data_ids=[";
  for (const auto& data_id : frame.DataIds()) {
    stream << "(" << data_id.sensor_id.type << ", " << data_id.sensor_id.id
           << ", " << data_id.id << "), ";
  }
  if (!frame.DataIds().empty()) {
    stream.seekp(-2, std::ios_base::end);
  }
  stream << "])";
  return stream;
}

}  // namespace colmap

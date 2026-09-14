// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/reconstruction_io_utils.h"

#include "colmap/util/hash_containers.h"

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

NodeHashMap<image_t, Frame*> ExtractImageToFramePtr(
    Reconstruction& reconstruction) {
  NodeHashMap<image_t, Frame*> image_to_frame;
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    for (const data_t& data_id : frame.ImageIds()) {
      THROW_CHECK(
          image_to_frame.emplace(data_id.id, &reconstruction.Frame(frame_id))
              .second);
    }
  }
  return image_to_frame;
}

}  // namespace colmap

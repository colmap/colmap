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

#include "colmap/estimators/poselib_utils.h"

namespace colmap {

poselib::Camera ConvertCameraToPoseLibCamera(const Camera& camera) {
  return poselib::Camera(
      camera.ModelName(), camera.params, camera.width, camera.height);
}

Camera ConvertPoseLibCameraToCamera(const poselib::Camera& camera) {
  Camera colmap_camera;
  colmap_camera.model_id = CameraModelNameToId(camera.model_name());
  colmap_camera.width = camera.width;
  colmap_camera.height = camera.height;
  colmap_camera.params = camera.params;
  return colmap_camera;
}

poselib::CameraPose ConvertRigid3dToPoseLibPose(const Rigid3d& rigid) {
  poselib::CameraPose pose;
  // PoseLib uses (w, x, y, z) ordering for quaternion
  pose.q << rigid.rotation.w(), rigid.rotation.x(), rigid.rotation.y(),
      rigid.rotation.z();
  pose.t = rigid.translation;
  return pose;
}

Rigid3d ConvertPoseLibPoseToRigid3d(const poselib::CameraPose& pose) {
  // PoseLib quaternion is (w, x, y, z), Eigen::Quaterniond constructor is also
  // (w, x, y, z)
  return Rigid3d(
      Eigen::Quaterniond(pose.q(0), pose.q(1), pose.q(2), pose.q(3)), pose.t);
}

}  // namespace colmap

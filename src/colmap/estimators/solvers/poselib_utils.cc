// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/solvers/poselib_utils.h"

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
  pose.q << rigid.rotation().w(), rigid.rotation().x(), rigid.rotation().y(),
      rigid.rotation().z();
  pose.t = rigid.translation();
  return pose;
}

Rigid3d ConvertPoseLibPoseToRigid3d(const poselib::CameraPose& pose) {
  // PoseLib quaternion is (w, x, y, z), Eigen::Quaterniond constructor is also
  // (w, x, y, z)
  return Rigid3d(Eigen::Quaterniond(pose.q(0), pose.q(1), pose.q(2), pose.q(3)),
                 pose.t);
}

}  // namespace colmap

// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/scene/camera.h"

#include <PoseLib/camera_pose.h>

namespace colmap {

// Convert COLMAP Camera to PoseLib Camera.
poselib::Camera ConvertCameraToPoseLibCamera(const Camera& camera);

// Convert PoseLib Camera to COLMAP Camera.
Camera ConvertPoseLibCameraToCamera(const poselib::Camera& camera);

// Convert COLMAP Rigid3d to PoseLib CameraPose.
poselib::CameraPose ConvertRigid3dToPoseLibPose(const Rigid3d& rigid);

// Convert PoseLib CameraPose to COLMAP Rigid3d.
Rigid3d ConvertPoseLibPoseToRigid3d(const poselib::CameraPose& pose);

}  // namespace colmap

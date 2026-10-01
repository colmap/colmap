// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/solvers/poselib_utils.h"

#include "colmap/geometry/rigid3_matchers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(PoseLibUtils, CameraRoundTrip) {
  Camera camera;
  camera.model_id = CameraModelId::kSimpleRadial;
  camera.width = 1920;
  camera.height = 1080;
  camera.params = {1000.0, 960.0, 540.0, 0.1};

  const poselib::Camera poselib_camera = ConvertCameraToPoseLibCamera(camera);
  const Camera camera_back = ConvertPoseLibCameraToCamera(poselib_camera);

  EXPECT_EQ(camera.model_id, camera_back.model_id);
  EXPECT_EQ(camera.width, camera_back.width);
  EXPECT_EQ(camera.height, camera_back.height);
  EXPECT_EQ(camera.params, camera_back.params);
}

TEST(PoseLibUtils, Rigid3dRoundTrip) {
  const Eigen::Quaterniond rotation =
      Eigen::Quaterniond(0.5, 0.5, 0.5, 0.5).normalized();
  const Eigen::Vector3d translation(1.0, 2.0, 3.0);
  const Rigid3d rigid(rotation, translation);

  const poselib::CameraPose poselib_pose = ConvertRigid3dToPoseLibPose(rigid);
  const Rigid3d rigid_back = ConvertPoseLibPoseToRigid3d(poselib_pose);

  EXPECT_THAT(rigid_back, Rigid3dNear(rigid, 1e-10, 1e-10));
}

}  // namespace
}  // namespace colmap

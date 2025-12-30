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

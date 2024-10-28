// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/scene/camera_rig.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(CameraRig, Empty) {
  CameraRig camera_rig;
  EXPECT_EQ(camera_rig.NumCameras(), 0);
  EXPECT_EQ(camera_rig.NumSnapshots(), 0);
  EXPECT_EQ(camera_rig.GetCameraIds().size(), 0);
  EXPECT_FALSE(camera_rig.HasCamera(0));
}

TEST(CameraRig, AddCamera) {
  CameraRig camera_rig;
  EXPECT_EQ(camera_rig.NumCameras(), 0);
  EXPECT_EQ(camera_rig.NumSnapshots(), 0);
  EXPECT_EQ(camera_rig.GetCameraIds().size(), 0);
  EXPECT_FALSE(camera_rig.HasCamera(0));

  const Rigid3d cam_from_rig1(Eigen::Quaterniond::Identity(),
                              Eigen::Vector3d(0, 1, 2));
  camera_rig.AddCamera(0, cam_from_rig1);
  EXPECT_EQ(camera_rig.NumCameras(), 1);
  EXPECT_EQ(camera_rig.NumSnapshots(), 0);
  EXPECT_EQ(camera_rig.GetCameraIds().size(), 1);
  EXPECT_EQ(camera_rig.GetCameraIds()[0], 0);
  EXPECT_TRUE(camera_rig.HasCamera(0));
  EXPECT_EQ(camera_rig.CamFromRig(0).rotation.coeffs(),
            cam_from_rig1.rotation.coeffs());
  EXPECT_EQ(camera_rig.CamFromRig(0).translation, cam_from_rig1.translation);

  const Rigid3d cam_from_rig2(Eigen::Quaterniond::Identity(),
                              Eigen::Vector3d(0, 1, 2));
  camera_rig.AddCamera(1, cam_from_rig2);
  EXPECT_EQ(camera_rig.NumCameras(), 2);
  EXPECT_EQ(camera_rig.NumSnapshots(), 0);
  EXPECT_EQ(camera_rig.GetCameraIds().size(), 2);
  EXPECT_TRUE(camera_rig.HasCamera(0));
  EXPECT_TRUE(camera_rig.HasCamera(1));
  EXPECT_EQ(camera_rig.CamFromRig(1).rotation.coeffs(),
            cam_from_rig2.rotation.coeffs());
  EXPECT_EQ(camera_rig.CamFromRig(1).translation, cam_from_rig2.translation);
}

TEST(CameraRig, AddSnapshot) {
  CameraRig camera_rig;
  EXPECT_EQ(camera_rig.NumCameras(), 0);
  EXPECT_EQ(camera_rig.NumSnapshots(), 0);
  EXPECT_EQ(camera_rig.GetCameraIds().size(), 0);
  EXPECT_EQ(camera_rig.Snapshots().size(), 0);

  camera_rig.AddCamera(
      0, Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(0, 1, 2)));
  camera_rig.AddCamera(
      1, Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(3, 4, 5)));
  EXPECT_EQ(camera_rig.NumCameras(), 2);
  EXPECT_EQ(camera_rig.NumSnapshots(), 0);
  EXPECT_EQ(camera_rig.Snapshots().size(), 0);

  const std::vector<image_t> image_ids1 = {0, 1};
  camera_rig.AddSnapshot(image_ids1);
  EXPECT_EQ(camera_rig.NumCameras(), 2);
  EXPECT_EQ(camera_rig.NumSnapshots(), 1);
  EXPECT_EQ(camera_rig.Snapshots().size(), 1);
  EXPECT_EQ(camera_rig.Snapshots()[0].size(), 2);
  EXPECT_EQ(camera_rig.Snapshots()[0][0], 0);
  EXPECT_EQ(camera_rig.Snapshots()[0][1], 1);

  const std::vector<image_t> image_ids2 = {2, 3};
  camera_rig.AddSnapshot(image_ids2);
  EXPECT_EQ(camera_rig.NumCameras(), 2);
  EXPECT_EQ(camera_rig.NumSnapshots(), 2);
  EXPECT_EQ(camera_rig.Snapshots().size(), 2);
  EXPECT_EQ(camera_rig.Snapshots()[0].size(), 2);
  EXPECT_EQ(camera_rig.Snapshots()[0][0], 0);
  EXPECT_EQ(camera_rig.Snapshots()[0][1], 1);
  EXPECT_EQ(camera_rig.Snapshots()[1].size(), 2);
  EXPECT_EQ(camera_rig.Snapshots()[1][0], 2);
  EXPECT_EQ(camera_rig.Snapshots()[1][1], 3);
}

TEST(CameraRig, Check) {
  CameraRig camera_rig;
  camera_rig.AddCamera(
      0, Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(0, 1, 2)));
  camera_rig.AddCamera(
      1, Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(3, 4, 5)));
  const std::vector<image_t> image_ids1 = {0, 1};
  camera_rig.AddSnapshot(image_ids1);
  const std::vector<image_t> image_ids2 = {2, 3};
  camera_rig.AddSnapshot(image_ids2);

  Reconstruction reconstruction;

  Camera camera1 = Camera::CreateFromModelName(0, "PINHOLE", 1, 1, 1);
  reconstruction.AddCamera(camera1);

  Camera camera2 = Camera::CreateFromModelName(1, "PINHOLE", 1, 1, 1);
  reconstruction.AddCamera(camera2);

  Image image1;
  image1.SetImageId(0);
  image1.SetCameraId(camera1.camera_id);
  reconstruction.AddImage(image1);

  Image image2;
  image2.SetImageId(1);
  image2.SetCameraId(camera2.camera_id);
  reconstruction.AddImage(image2);

  Image image3;
  image3.SetImageId(2);
  image3.SetCameraId(camera1.camera_id);
  reconstruction.AddImage(image3);

  Image image4;
  image4.SetImageId(3);
  image4.SetCameraId(camera2.camera_id);
  reconstruction.AddImage(image4);

  camera_rig.SetRefCameraId(0);
  camera_rig.Check(reconstruction);
}

TEST(CameraRig, ComputeRigFromWorldScale) {
  CameraRig camera_rig;
  camera_rig.AddCamera(
      0, Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(0, 0, 0)));
  camera_rig.AddCamera(
      1, Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(2, 4, 6)));
  const std::vector<image_t> image_ids1 = {0, 1};
  camera_rig.AddSnapshot(image_ids1);

  Reconstruction reconstruction;

  Camera camera1 = Camera::CreateFromModelName(0, "PINHOLE", 1, 1, 1);
  reconstruction.AddCamera(camera1);

  Camera camera2 = Camera::CreateFromModelName(1, "PINHOLE", 1, 1, 1);
  reconstruction.AddCamera(camera2);

  Image image1;
  image1.SetImageId(0);
  image1.SetCameraId(camera1.camera_id);
  image1.SetCamFromWorld(Rigid3d());
  reconstruction.AddImage(image1);

  Image image2;
  image2.SetImageId(1);
  image2.SetCameraId(camera2.camera_id);
  image2.SetCamFromWorld(Rigid3d());
  image2.CamFromWorld().translation = Eigen::Vector3d(1, 2, 3);
  reconstruction.AddImage(image2);

  camera_rig.SetRefCameraId(0);
  camera_rig.Check(reconstruction);

  EXPECT_EQ(camera_rig.ComputeRigFromWorldScale(reconstruction), 2.0);

  reconstruction.Image(1).CamFromWorld().translation = Eigen::Vector3d(0, 0, 0);
  EXPECT_TRUE(std::isnan(camera_rig.ComputeRigFromWorldScale(reconstruction)));
}

TEST(CameraRig, ComputeCamsFromRigs) {
  CameraRig camera_rig;
  camera_rig.AddCamera(
      0, Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(0, 0, 0)));
  camera_rig.AddCamera(
      1, Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(0, 0, 0)));
  const std::vector<image_t> image_ids1 = {0, 1};
  camera_rig.AddSnapshot(image_ids1);

  Reconstruction reconstruction;

  Camera camera1 = Camera::CreateFromModelName(0, "PINHOLE", 1, 1, 1);
  reconstruction.AddCamera(camera1);

  Camera camera2 = Camera::CreateFromModelName(1, "PINHOLE", 1, 1, 1);
  reconstruction.AddCamera(camera2);

  Image image1;
  image1.SetImageId(0);
  image1.SetCameraId(camera1.camera_id);
  image1.SetCamFromWorld(Rigid3d());
  reconstruction.AddImage(image1);

  Image image2;
  image2.SetImageId(1);
  image2.SetCameraId(camera2.camera_id);
  image2.SetCamFromWorld(Rigid3d());
  image2.CamFromWorld().translation = Eigen::Vector3d(1, 2, 3);
  reconstruction.AddImage(image2);

  camera_rig.SetRefCameraId(0);
  camera_rig.Check(reconstruction);
  camera_rig.ComputeCamsFromRigs(reconstruction);
  EXPECT_EQ(camera_rig.CamFromRig(0).rotation.coeffs(),
            Eigen::Quaterniond::Identity().coeffs());
  EXPECT_EQ(camera_rig.CamFromRig(0).translation, Eigen::Vector3d(0, 0, 0));
  EXPECT_EQ(camera_rig.CamFromRig(1).rotation.coeffs(),
            Eigen::Quaterniond::Identity().coeffs());
  EXPECT_EQ(camera_rig.CamFromRig(1).translation, Eigen::Vector3d(1, 2, 3));

  const std::vector<image_t> image_ids2 = {2, 3};
  camera_rig.AddSnapshot(image_ids2);

  Image image3;
  image3.SetImageId(2);
  image3.SetCameraId(camera1.camera_id);
  image3.SetCamFromWorld(Rigid3d());
  reconstruction.AddImage(image3);

  Image image4;
  image4.SetImageId(3);
  image4.SetCameraId(camera2.camera_id);
  image4.SetCamFromWorld(Rigid3d());
  image4.CamFromWorld().translation = Eigen::Vector3d(2, 4, 6);
  reconstruction.AddImage(image4);

  camera_rig.Check(reconstruction);
  camera_rig.ComputeCamsFromRigs(reconstruction);
  EXPECT_EQ(camera_rig.CamFromRig(0).rotation.coeffs(),
            Eigen::Quaterniond::Identity().coeffs());
  EXPECT_EQ(camera_rig.CamFromRig(0).translation, Eigen::Vector3d(0, 0, 0));
  EXPECT_EQ(camera_rig.CamFromRig(1).rotation.coeffs(),
            Eigen::Quaterniond::Identity().coeffs());
  EXPECT_EQ(camera_rig.CamFromRig(1).translation, Eigen::Vector3d(1.5, 3, 4.5));

  const std::vector<image_t> image_ids3 = {4};
  camera_rig.AddSnapshot(image_ids3);

  Image image5;
  image5.SetImageId(4);
  image5.SetCameraId(camera1.camera_id);
  image5.SetCamFromWorld(Rigid3d());
  reconstruction.AddImage(image5);

  camera_rig.Check(reconstruction);
  camera_rig.ComputeCamsFromRigs(reconstruction);
  EXPECT_EQ(camera_rig.CamFromRig(0).rotation.coeffs(),
            Eigen::Quaterniond::Identity().coeffs());
  EXPECT_EQ(camera_rig.CamFromRig(0).translation, Eigen::Vector3d(0, 0, 0));
  EXPECT_EQ(camera_rig.CamFromRig(1).rotation.coeffs(),
            Eigen::Quaterniond::Identity().coeffs());
  EXPECT_EQ(camera_rig.CamFromRig(1).translation, Eigen::Vector3d(1.5, 3, 4.5));
}

TEST(CameraRig, ComputeRigFromWorld) {
  CameraRig camera_rig;
  camera_rig.AddCamera(
      0, Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(0, 1, 2)));
  camera_rig.AddCamera(
      1, Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(3, 4, 5)));
  const std::vector<image_t> image_ids1 = {0, 1};
  camera_rig.AddSnapshot(image_ids1);

  Reconstruction reconstruction;

  Camera camera1 = Camera::CreateFromModelName(0, "PINHOLE", 1, 1, 1);
  reconstruction.AddCamera(camera1);

  Camera camera2 = Camera::CreateFromModelName(1, "PINHOLE", 1, 1, 1);
  reconstruction.AddCamera(camera2);

  Image image1;
  image1.SetImageId(0);
  image1.SetCameraId(camera1.camera_id);
  image1.SetCamFromWorld(Rigid3d());
  reconstruction.AddImage(image1);

  Image image2;
  image2.SetImageId(1);
  image2.SetCameraId(camera2.camera_id);
  image2.SetCamFromWorld(Rigid3d());
  image2.CamFromWorld().translation = Eigen::Vector3d(3, 3, 3);
  reconstruction.AddImage(image2);

  camera_rig.SetRefCameraId(0);
  camera_rig.Check(reconstruction);

  const Rigid3d rig_from_world =
      camera_rig.ComputeRigFromWorld(0, reconstruction);
  EXPECT_EQ(rig_from_world.rotation.coeffs(),
            Eigen::Quaterniond::Identity().coeffs());
  EXPECT_EQ(rig_from_world.translation, Eigen::Vector3d(0, -1, -2));
}

}  // namespace
}  // namespace colmap

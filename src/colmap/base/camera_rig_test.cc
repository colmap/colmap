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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#define TEST_NAME "base/camera_rig"
#include "colmap/base/camera_rig.h"

#include "colmap/util/testing.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestEmpty) {
  CameraRig camera_rig;
  BOOST_CHECK_EQUAL(camera_rig.NumCameras(), 0);
  BOOST_CHECK_EQUAL(camera_rig.NumSnapshots(), 0);
  BOOST_CHECK_EQUAL(camera_rig.GetCameraIds().size(), 0);
  BOOST_CHECK_EQUAL(camera_rig.HasCamera(0), false);
}

BOOST_AUTO_TEST_CASE(TestAddCamera) {
  CameraRig camera_rig;
  BOOST_CHECK_EQUAL(camera_rig.NumCameras(), 0);
  BOOST_CHECK_EQUAL(camera_rig.NumSnapshots(), 0);
  BOOST_CHECK_EQUAL(camera_rig.GetCameraIds().size(), 0);
  BOOST_CHECK_EQUAL(camera_rig.HasCamera(0), false);

  camera_rig.AddCamera(
      0, ComposeIdentityQuaternion(), Eigen::Vector3d(0, 1, 2));
  BOOST_CHECK_EQUAL(camera_rig.NumCameras(), 1);
  BOOST_CHECK_EQUAL(camera_rig.NumSnapshots(), 0);
  BOOST_CHECK_EQUAL(camera_rig.GetCameraIds().size(), 1);
  BOOST_CHECK_EQUAL(camera_rig.GetCameraIds()[0], 0);
  BOOST_CHECK_EQUAL(camera_rig.HasCamera(0), true);
  BOOST_CHECK_EQUAL(camera_rig.RelativeQvec(0), ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(camera_rig.RelativeTvec(0), Eigen::Vector3d(0, 1, 2));

  camera_rig.AddCamera(
      1, ComposeIdentityQuaternion(), Eigen::Vector3d(3, 4, 5));
  BOOST_CHECK_EQUAL(camera_rig.NumCameras(), 2);
  BOOST_CHECK_EQUAL(camera_rig.NumSnapshots(), 0);
  BOOST_CHECK_EQUAL(camera_rig.GetCameraIds().size(), 2);
  BOOST_CHECK_EQUAL(camera_rig.HasCamera(0), true);
  BOOST_CHECK_EQUAL(camera_rig.HasCamera(1), true);
  BOOST_CHECK_EQUAL(camera_rig.RelativeQvec(1), ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(camera_rig.RelativeTvec(1), Eigen::Vector3d(3, 4, 5));
}

BOOST_AUTO_TEST_CASE(TestAddSnapshot) {
  CameraRig camera_rig;
  BOOST_CHECK_EQUAL(camera_rig.NumCameras(), 0);
  BOOST_CHECK_EQUAL(camera_rig.NumSnapshots(), 0);
  BOOST_CHECK_EQUAL(camera_rig.GetCameraIds().size(), 0);
  BOOST_CHECK_EQUAL(camera_rig.Snapshots().size(), 0);

  camera_rig.AddCamera(
      0, ComposeIdentityQuaternion(), Eigen::Vector3d(0, 1, 2));
  camera_rig.AddCamera(
      1, ComposeIdentityQuaternion(), Eigen::Vector3d(3, 4, 5));
  BOOST_CHECK_EQUAL(camera_rig.NumCameras(), 2);
  BOOST_CHECK_EQUAL(camera_rig.NumSnapshots(), 0);
  BOOST_CHECK_EQUAL(camera_rig.Snapshots().size(), 0);

  const std::vector<image_t> image_ids1 = {0, 1};
  camera_rig.AddSnapshot(image_ids1);
  BOOST_CHECK_EQUAL(camera_rig.NumCameras(), 2);
  BOOST_CHECK_EQUAL(camera_rig.NumSnapshots(), 1);
  BOOST_CHECK_EQUAL(camera_rig.Snapshots().size(), 1);
  BOOST_CHECK_EQUAL(camera_rig.Snapshots()[0].size(), 2);
  BOOST_CHECK_EQUAL(camera_rig.Snapshots()[0][0], 0);
  BOOST_CHECK_EQUAL(camera_rig.Snapshots()[0][1], 1);

  const std::vector<image_t> image_ids2 = {2, 3};
  camera_rig.AddSnapshot(image_ids2);
  BOOST_CHECK_EQUAL(camera_rig.NumCameras(), 2);
  BOOST_CHECK_EQUAL(camera_rig.NumSnapshots(), 2);
  BOOST_CHECK_EQUAL(camera_rig.Snapshots().size(), 2);
  BOOST_CHECK_EQUAL(camera_rig.Snapshots()[0].size(), 2);
  BOOST_CHECK_EQUAL(camera_rig.Snapshots()[0][0], 0);
  BOOST_CHECK_EQUAL(camera_rig.Snapshots()[0][1], 1);
  BOOST_CHECK_EQUAL(camera_rig.Snapshots()[1].size(), 2);
  BOOST_CHECK_EQUAL(camera_rig.Snapshots()[1][0], 2);
  BOOST_CHECK_EQUAL(camera_rig.Snapshots()[1][1], 3);
}

BOOST_AUTO_TEST_CASE(TestCheck) {
  CameraRig camera_rig;
  camera_rig.AddCamera(
      0, ComposeIdentityQuaternion(), Eigen::Vector3d(0, 1, 2));
  camera_rig.AddCamera(
      1, ComposeIdentityQuaternion(), Eigen::Vector3d(3, 4, 5));
  const std::vector<image_t> image_ids1 = {0, 1};
  camera_rig.AddSnapshot(image_ids1);
  const std::vector<image_t> image_ids2 = {2, 3};
  camera_rig.AddSnapshot(image_ids2);

  Reconstruction reconstruction;

  Camera camera1;
  camera1.SetCameraId(0);
  camera1.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction.AddCamera(camera1);

  Camera camera2;
  camera2.SetCameraId(1);
  camera2.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction.AddCamera(camera2);

  Image image1;
  image1.SetImageId(0);
  image1.SetCameraId(camera1.CameraId());
  reconstruction.AddImage(image1);

  Image image2;
  image2.SetImageId(1);
  image2.SetCameraId(camera2.CameraId());
  reconstruction.AddImage(image2);

  Image image3;
  image3.SetImageId(2);
  image3.SetCameraId(camera1.CameraId());
  reconstruction.AddImage(image3);

  Image image4;
  image4.SetImageId(3);
  image4.SetCameraId(camera2.CameraId());
  reconstruction.AddImage(image4);

  camera_rig.SetRefCameraId(0);
  camera_rig.Check(reconstruction);
}

BOOST_AUTO_TEST_CASE(TestComputeScale) {
  CameraRig camera_rig;
  camera_rig.AddCamera(
      0, ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0));
  camera_rig.AddCamera(
      1, ComposeIdentityQuaternion(), Eigen::Vector3d(2, 4, 6));
  const std::vector<image_t> image_ids1 = {0, 1};
  camera_rig.AddSnapshot(image_ids1);

  Reconstruction reconstruction;

  Camera camera1;
  camera1.SetCameraId(0);
  camera1.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction.AddCamera(camera1);

  Camera camera2;
  camera2.SetCameraId(1);
  camera2.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction.AddCamera(camera2);

  Image image1;
  image1.SetImageId(0);
  image1.SetCameraId(camera1.CameraId());
  image1.SetQvec(ComposeIdentityQuaternion());
  image1.SetTvec(Eigen::Vector3d(0, 0, 0));
  reconstruction.AddImage(image1);

  Image image2;
  image2.SetImageId(1);
  image2.SetCameraId(camera2.CameraId());
  image2.SetQvec(ComposeIdentityQuaternion());
  image2.SetTvec(Eigen::Vector3d(1, 2, 3));
  reconstruction.AddImage(image2);

  camera_rig.SetRefCameraId(0);
  camera_rig.Check(reconstruction);

  BOOST_CHECK_EQUAL(camera_rig.ComputeScale(reconstruction), 2.0);

  reconstruction.Image(1).SetTvec(Eigen::Vector3d(0, 0, 0));
  BOOST_CHECK(IsNaN(camera_rig.ComputeScale(reconstruction)));
}

BOOST_AUTO_TEST_CASE(TestComputeRelativePoses) {
  CameraRig camera_rig;
  camera_rig.AddCamera(
      0, ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0));
  camera_rig.AddCamera(
      1, ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0));
  const std::vector<image_t> image_ids1 = {0, 1};
  camera_rig.AddSnapshot(image_ids1);

  Reconstruction reconstruction;

  Camera camera1;
  camera1.SetCameraId(0);
  camera1.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction.AddCamera(camera1);

  Camera camera2;
  camera2.SetCameraId(1);
  camera2.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction.AddCamera(camera2);

  Image image1;
  image1.SetImageId(0);
  image1.SetCameraId(camera1.CameraId());
  image1.SetQvec(ComposeIdentityQuaternion());
  image1.SetTvec(Eigen::Vector3d(0, 0, 0));
  reconstruction.AddImage(image1);

  Image image2;
  image2.SetImageId(1);
  image2.SetCameraId(camera2.CameraId());
  image2.SetQvec(ComposeIdentityQuaternion());
  image2.SetTvec(Eigen::Vector3d(1, 2, 3));
  reconstruction.AddImage(image2);

  camera_rig.SetRefCameraId(0);
  camera_rig.Check(reconstruction);
  camera_rig.ComputeRelativePoses(reconstruction);
  BOOST_CHECK_EQUAL(camera_rig.RelativeQvec(0), ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(camera_rig.RelativeTvec(0), Eigen::Vector3d(0, 0, 0));
  BOOST_CHECK_EQUAL(camera_rig.RelativeQvec(1), ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(camera_rig.RelativeTvec(1), Eigen::Vector3d(1, 2, 3));

  const std::vector<image_t> image_ids2 = {2, 3};
  camera_rig.AddSnapshot(image_ids2);

  Image image3;
  image3.SetImageId(2);
  image3.SetCameraId(camera1.CameraId());
  image3.SetQvec(ComposeIdentityQuaternion());
  image3.SetTvec(Eigen::Vector3d(0, 0, 0));
  reconstruction.AddImage(image3);

  Image image4;
  image4.SetImageId(3);
  image4.SetCameraId(camera2.CameraId());
  image4.SetQvec(ComposeIdentityQuaternion());
  image4.SetTvec(Eigen::Vector3d(2, 4, 6));
  reconstruction.AddImage(image4);

  camera_rig.Check(reconstruction);
  camera_rig.ComputeRelativePoses(reconstruction);
  BOOST_CHECK_EQUAL(camera_rig.RelativeQvec(0), ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(camera_rig.RelativeTvec(0), Eigen::Vector3d(0, 0, 0));
  BOOST_CHECK_EQUAL(camera_rig.RelativeQvec(1), ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(camera_rig.RelativeTvec(1), Eigen::Vector3d(1.5, 3, 4.5));

  const std::vector<image_t> image_ids3 = {4};
  camera_rig.AddSnapshot(image_ids3);

  Image image5;
  image5.SetImageId(4);
  image5.SetCameraId(camera1.CameraId());
  image5.SetQvec(ComposeIdentityQuaternion());
  image5.SetTvec(Eigen::Vector3d(0, 0, 0));
  reconstruction.AddImage(image5);

  camera_rig.Check(reconstruction);
  camera_rig.ComputeRelativePoses(reconstruction);
  BOOST_CHECK_EQUAL(camera_rig.RelativeQvec(0), ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(camera_rig.RelativeTvec(0), Eigen::Vector3d(0, 0, 0));
  BOOST_CHECK_EQUAL(camera_rig.RelativeQvec(1), ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(camera_rig.RelativeTvec(1), Eigen::Vector3d(1.5, 3, 4.5));
}

BOOST_AUTO_TEST_CASE(TestComputeAbsolutePose) {
  CameraRig camera_rig;
  camera_rig.AddCamera(
      0, ComposeIdentityQuaternion(), Eigen::Vector3d(0, 1, 2));
  camera_rig.AddCamera(
      1, ComposeIdentityQuaternion(), Eigen::Vector3d(3, 4, 5));
  const std::vector<image_t> image_ids1 = {0, 1};
  camera_rig.AddSnapshot(image_ids1);

  Reconstruction reconstruction;

  Camera camera1;
  camera1.SetCameraId(0);
  camera1.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction.AddCamera(camera1);

  Camera camera2;
  camera2.SetCameraId(1);
  camera2.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction.AddCamera(camera2);

  Image image1;
  image1.SetImageId(0);
  image1.SetCameraId(camera1.CameraId());
  image1.SetQvec(ComposeIdentityQuaternion());
  image1.SetTvec(Eigen::Vector3d(0, 0, 0));
  reconstruction.AddImage(image1);

  Image image2;
  image2.SetImageId(1);
  image2.SetCameraId(camera2.CameraId());
  image2.SetQvec(ComposeIdentityQuaternion());
  image2.SetTvec(Eigen::Vector3d(3, 3, 3));
  reconstruction.AddImage(image2);

  camera_rig.SetRefCameraId(0);
  camera_rig.Check(reconstruction);

  Eigen::Vector4d abs_qvec;
  Eigen::Vector3d abs_tvec;
  camera_rig.ComputeAbsolutePose(0, reconstruction, &abs_qvec, &abs_tvec);
  BOOST_CHECK_EQUAL(abs_qvec, ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(abs_tvec, Eigen::Vector3d(0, -1, -2));
}

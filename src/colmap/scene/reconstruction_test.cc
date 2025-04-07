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

#include "colmap/scene/reconstruction.h"

#include "colmap/geometry/pose.h"
#include "colmap/geometry/sim3.h"
#include "colmap/scene/reconstruction_io.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/models.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include <gtest/gtest.h>

namespace colmap {
namespace {

void ExpectValidCameraPtrs(const Reconstruction& reconstruction) {
  for (const auto& image : reconstruction.Images()) {
    EXPECT_TRUE(image.second.HasCameraPtr());
    auto& camera = reconstruction.Camera(image.second.CameraId());
    EXPECT_EQ(image.second.CameraPtr(), &camera);
  }
}

void ExpectEqualReconstructions(const Reconstruction& reconstruction1,
                                const Reconstruction& reconstruction2) {
  // compare cameras
  std::stringstream stream1_cameras, stream2_cameras;
  WriteCamerasText(reconstruction1, stream1_cameras);
  WriteCamerasText(reconstruction2, stream2_cameras);
  EXPECT_EQ(stream1_cameras.str(), stream2_cameras.str());

  // compare images
  std::stringstream stream1_images, stream2_images;
  WriteImagesText(reconstruction1, stream1_images);
  WriteImagesText(reconstruction2, stream2_images);
  EXPECT_EQ(stream1_images.str(), stream2_images.str());

  // compare point3ds
  std::stringstream stream1_points3D, stream2_points3D;
  WritePoints3DText(reconstruction1, stream1_points3D);
  WritePoints3DText(reconstruction2, stream2_points3D);
  EXPECT_EQ(stream1_points3D.str(), stream2_points3D.str());
}

void GenerateReconstruction(const image_t num_images,
                            Reconstruction* reconstruction) {
  const size_t kNumPoints2D = 10;

  Camera camera = Camera::CreateFromModelName(1, "PINHOLE", 1, 1, 1);
  reconstruction->AddCamera(camera);

  for (image_t image_id = 1; image_id <= num_images; ++image_id) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(camera.camera_id);
    image.SetName("image" + std::to_string(image_id));
    image.SetPoints2D(
        std::vector<Eigen::Vector2d>(kNumPoints2D, Eigen::Vector2d::Zero()));
    image.SetCamFromWorld(Rigid3d());
    reconstruction->AddImage(image);
  }
}

TEST(Reconstruction, Empty) {
  Reconstruction reconstruction;
  EXPECT_EQ(reconstruction.NumCameras(), 0);
  EXPECT_EQ(reconstruction.NumImages(), 0);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
}

TEST(Reconstruction, ConstructCopy) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_cameras = 2;
  synthetic_dataset_options.num_images = 5;
  synthetic_dataset_options.num_points3D = 21;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  const Reconstruction reconstruction_copy(reconstruction);
  ExpectEqualReconstructions(reconstruction, reconstruction_copy);
}

TEST(Reconstruction, AssignCopy) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_cameras = 2;
  synthetic_dataset_options.num_images = 5;
  synthetic_dataset_options.num_points3D = 21;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  Reconstruction reconstruction_copy;
  reconstruction_copy = reconstruction;
  ExpectEqualReconstructions(reconstruction, reconstruction_copy);
}

TEST(Reconstruction, Print) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_cameras = 1;
  synthetic_dataset_options.num_images = 2;
  synthetic_dataset_options.num_points3D = 3;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  std::ostringstream stream;
  stream << reconstruction;
  EXPECT_EQ(stream.str(),
            "Reconstruction(num_cameras=1, num_images=2, num_reg_images=2, "
            "num_points3D=3)");
}

TEST(Reconstruction, AddCamera) {
  Reconstruction reconstruction;
  Camera camera =
      Camera::CreateFromModelId(1, SimplePinholeCameraModel::model_id, 1, 1, 1);
  reconstruction.AddCamera(camera);
  EXPECT_TRUE(reconstruction.ExistsCamera(camera.camera_id));
  EXPECT_EQ(reconstruction.Camera(camera.camera_id).camera_id,
            camera.camera_id);
  EXPECT_EQ(reconstruction.Cameras().count(camera.camera_id), 1);
  EXPECT_EQ(reconstruction.Cameras().size(), 1);
  EXPECT_EQ(reconstruction.NumCameras(), 1);
  EXPECT_EQ(reconstruction.NumImages(), 0);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
}

TEST(Reconstruction, AddImage) {
  Reconstruction reconstruction;
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimplePinhole, 1, 1, 1);
  Image image;
  image.SetCameraId(camera.camera_id);
  image.SetImageId(1);
  EXPECT_ANY_THROW(reconstruction.AddImage(image));
  reconstruction.AddCamera(camera);
  reconstruction.AddImage(image);
  EXPECT_TRUE(reconstruction.ExistsImage(1));
  EXPECT_EQ(reconstruction.Image(1).ImageId(), 1);
  EXPECT_FALSE(reconstruction.Image(1).HasPose());
  EXPECT_EQ(reconstruction.Images().count(1), 1);
  EXPECT_EQ(reconstruction.Images().size(), 1);
  EXPECT_EQ(reconstruction.NumCameras(), 1);
  EXPECT_EQ(reconstruction.NumImages(), 1);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  ExpectValidCameraPtrs(reconstruction);
}

TEST(Reconstruction, AddPoint3D) {
  Reconstruction reconstruction;
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  EXPECT_TRUE(reconstruction.ExistsPoint3D(point3D_id));
  EXPECT_EQ(reconstruction.Point3D(point3D_id).track.Length(), 0);
  EXPECT_EQ(reconstruction.Points3D().count(point3D_id), 1);
  EXPECT_EQ(reconstruction.Points3D().size(), 1);
  EXPECT_EQ(reconstruction.NumCameras(), 0);
  EXPECT_EQ(reconstruction.NumImages(), 0);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  EXPECT_EQ(reconstruction.Point3DIds().count(point3D_id), 1);
}

TEST(Reconstruction, AddObservation) {
  Reconstruction reconstruction;
  GenerateReconstruction(3, &reconstruction);
  Track track;
  track.AddElement(1, 0);
  track.AddElement(2, 1);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), track);
  EXPECT_EQ(reconstruction.Image(1).NumPoints3D(), 1);
  EXPECT_TRUE(reconstruction.Image(1).Point2D(0).HasPoint3D());
  EXPECT_FALSE(reconstruction.Image(1).Point2D(1).HasPoint3D());
  EXPECT_EQ(reconstruction.Image(2).NumPoints3D(), 1);
  EXPECT_FALSE(reconstruction.Image(2).Point2D(0).HasPoint3D());
  EXPECT_TRUE(reconstruction.Image(2).Point2D(1).HasPoint3D());
  EXPECT_EQ(reconstruction.Point3D(point3D_id).track.Length(), 2);
  reconstruction.AddObservation(point3D_id, TrackElement(3, 2));
  EXPECT_EQ(reconstruction.Image(3).NumPoints3D(), 1);
  EXPECT_TRUE(reconstruction.Image(3).Point2D(2).HasPoint3D());
  EXPECT_EQ(reconstruction.Point3D(point3D_id).track.Length(), 3);
}

TEST(Reconstruction, MergePoints3D) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d(0, 0, 0), Track());
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  reconstruction.Point3D(point3D_id1).color =
      Eigen::Matrix<uint8_t, 3, 1>(0, 0, 0);
  const point3D_t point3D_id2 =
      reconstruction.AddPoint3D(Eigen::Vector3d(1, 1, 1), Track());
  reconstruction.AddObservation(point3D_id2, TrackElement(1, 1));
  reconstruction.AddObservation(point3D_id2, TrackElement(2, 1));
  reconstruction.Point3D(point3D_id2).color =
      Eigen::Matrix<uint8_t, 3, 1>(20, 20, 20);
  const point3D_t merged_point3D_id =
      reconstruction.MergePoints3D(point3D_id1, point3D_id2);
  EXPECT_FALSE(reconstruction.ExistsPoint3D(point3D_id1));
  EXPECT_FALSE(reconstruction.ExistsPoint3D(point3D_id2));
  EXPECT_TRUE(reconstruction.ExistsPoint3D(merged_point3D_id));
  EXPECT_EQ(reconstruction.Image(1).Point2D(0).point3D_id, merged_point3D_id);
  EXPECT_EQ(reconstruction.Image(1).Point2D(1).point3D_id, merged_point3D_id);
  EXPECT_EQ(reconstruction.Image(2).Point2D(0).point3D_id, merged_point3D_id);
  EXPECT_EQ(reconstruction.Image(2).Point2D(1).point3D_id, merged_point3D_id);
  EXPECT_TRUE(reconstruction.Point3D(merged_point3D_id)
                  .xyz.isApprox(Eigen::Vector3d(0.5, 0.5, 0.5)));
  EXPECT_EQ(reconstruction.Point3D(merged_point3D_id).color,
            Eigen::Vector3ub(10, 10, 10));
}

TEST(Reconstruction, DeletePoint3D) {
  Reconstruction reconstruction;
  GenerateReconstruction(1, &reconstruction);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  reconstruction.AddObservation(point3D_id, TrackElement(1, 0));
  reconstruction.DeletePoint3D(point3D_id);
  EXPECT_FALSE(reconstruction.ExistsPoint3D(point3D_id));
  EXPECT_EQ(reconstruction.Image(1).NumPoints3D(), 0);
}

TEST(Reconstruction, DeleteObservation) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(0, 0, 0), Track());
  reconstruction.AddObservation(point3D_id, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id, TrackElement(1, 1));
  reconstruction.AddObservation(point3D_id, TrackElement(1, 2));
  reconstruction.DeleteObservation(1, 0);
  EXPECT_EQ(reconstruction.Point3D(point3D_id).track.Length(), 2);
  EXPECT_FALSE(reconstruction.Image(point3D_id).Point2D(0).HasPoint3D());
  reconstruction.DeleteObservation(1, 1);
  EXPECT_FALSE(reconstruction.ExistsPoint3D(point3D_id));
  EXPECT_FALSE(reconstruction.Image(point3D_id).Point2D(1).HasPoint3D());
  EXPECT_FALSE(reconstruction.Image(point3D_id).Point2D(2).HasPoint3D());
}

TEST(Reconstruction, RegisterImage) {
  Reconstruction reconstruction;
  GenerateReconstruction(1, &reconstruction);
  EXPECT_EQ(reconstruction.NumRegImages(), 1);
  EXPECT_TRUE(reconstruction.Image(1).HasPose());
  reconstruction.RegisterImage(1);
  EXPECT_EQ(reconstruction.NumRegImages(), 1);
  EXPECT_TRUE(reconstruction.Image(1).HasPose());
  reconstruction.DeRegisterImage(1);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
  EXPECT_FALSE(reconstruction.Image(1).HasPose());
}

TEST(Reconstruction, Normalize) {
  Reconstruction reconstruction;
  GenerateReconstruction(3, &reconstruction);
  reconstruction.DeRegisterImage(1);
  reconstruction.DeRegisterImage(2);
  reconstruction.DeRegisterImage(3);
  Sim3d tform = reconstruction.Normalize(/*fixed_scale=*/false);
  EXPECT_EQ(tform.scale, 1);
  EXPECT_EQ(tform.rotation.coeffs(), Eigen::Quaterniond::Identity().coeffs());
  EXPECT_EQ(tform.translation, Eigen::Vector3d::Zero());
  reconstruction.Image(1).SetCamFromWorld(Rigid3d());
  reconstruction.Image(2).SetCamFromWorld(Rigid3d());
  reconstruction.Image(3).SetCamFromWorld(Rigid3d());
  reconstruction.Image(1).CamFromWorld().translation.z() = -20.0;
  reconstruction.Image(2).CamFromWorld().translation.z() = -10.0;
  reconstruction.Image(3).CamFromWorld().translation.z() = 0.0;
  reconstruction.RegisterImage(1);
  reconstruction.RegisterImage(2);
  reconstruction.RegisterImage(3);
  reconstruction.Normalize(/*fixed_scale=*/true);
  EXPECT_NEAR(
      reconstruction.Image(1).CamFromWorld().translation.z(), -10, 1e-6);
  EXPECT_NEAR(reconstruction.Image(2).CamFromWorld().translation.z(), 0, 1e-6);
  EXPECT_NEAR(reconstruction.Image(3).CamFromWorld().translation.z(), 10, 1e-6);
  reconstruction.Normalize(/*fixed_scale=*/false);
  EXPECT_NEAR(reconstruction.Image(1).CamFromWorld().translation.z(), -5, 1e-6);
  EXPECT_NEAR(reconstruction.Image(2).CamFromWorld().translation.z(), 0, 1e-6);
  EXPECT_NEAR(reconstruction.Image(3).CamFromWorld().translation.z(), 5, 1e-6);
  reconstruction.Normalize(/*fixed_scale=*/false, 5);
  EXPECT_NEAR(
      reconstruction.Image(1).CamFromWorld().translation.z(), -2.5, 1e-6);
  EXPECT_NEAR(reconstruction.Image(2).CamFromWorld().translation.z(), 0, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(3).CamFromWorld().translation.z(), 2.5, 1e-6);
  reconstruction.Normalize(/*fixed_scale=*/false, 10, 0.0, 1.0);
  EXPECT_NEAR(reconstruction.Image(1).CamFromWorld().translation.z(), -5, 1e-6);
  EXPECT_NEAR(reconstruction.Image(2).CamFromWorld().translation.z(), 0, 1e-6);
  EXPECT_NEAR(reconstruction.Image(3).CamFromWorld().translation.z(), 5, 1e-6);
  tform = reconstruction.Normalize(/*fixed_scale=*/false, 20);
  EXPECT_NEAR(
      reconstruction.Image(1).CamFromWorld().translation.z(), -10, 1e-6);
  EXPECT_NEAR(reconstruction.Image(2).CamFromWorld().translation.z(), 0, 1e-6);
  EXPECT_NEAR(reconstruction.Image(3).CamFromWorld().translation.z(), 10, 1e-6);
  reconstruction.Transform(Inverse(tform));
  EXPECT_NEAR(reconstruction.Image(1).CamFromWorld().translation.z(), -5, 1e-6);
  EXPECT_NEAR(reconstruction.Image(2).CamFromWorld().translation.z(), 0, 1e-6);
  EXPECT_NEAR(reconstruction.Image(3).CamFromWorld().translation.z(), 5, 1e-6);
  reconstruction.Transform(tform);
  EXPECT_NEAR(
      reconstruction.Image(1).CamFromWorld().translation.z(), -10, 1e-6);
  EXPECT_NEAR(reconstruction.Image(2).CamFromWorld().translation.z(), 0, 1e-6);
  EXPECT_NEAR(reconstruction.Image(3).CamFromWorld().translation.z(), 10, 1e-6);
  Image image;
  image.SetCameraId(reconstruction.Image(1).CameraId());
  image.SetImageId(4);
  image.SetCamFromWorld(Rigid3d());
  reconstruction.AddImage(image);
  reconstruction.RegisterImage(4);
  image.SetImageId(5);
  reconstruction.AddImage(image);
  reconstruction.RegisterImage(5);
  image.SetImageId(6);
  reconstruction.AddImage(image);
  reconstruction.RegisterImage(6);
  image.SetImageId(7);
  reconstruction.AddImage(image);
  reconstruction.RegisterImage(7);
  reconstruction.Image(4).CamFromWorld().translation.z() = -7.5;
  reconstruction.Image(5).CamFromWorld().translation.z() = -5.0;
  reconstruction.Image(6).CamFromWorld().translation.z() = 5.0;
  reconstruction.Image(7).CamFromWorld().translation.z() = 7.5;
  reconstruction.RegisterImage(7);
  reconstruction.Normalize(/*fixed_scale=*/false, 10, 0.0, 1.0);
  EXPECT_NEAR(reconstruction.Image(1).CamFromWorld().translation.z(), -5, 1e-6);
  EXPECT_NEAR(reconstruction.Image(2).CamFromWorld().translation.z(), 0, 1e-6);
  EXPECT_NEAR(reconstruction.Image(3).CamFromWorld().translation.z(), 5, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(4).CamFromWorld().translation.z(), -3.75, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(5).CamFromWorld().translation.z(), -2.5, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(6).CamFromWorld().translation.z(), 2.5, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(7).CamFromWorld().translation.z(), 3.75, 1e-6);
}

TEST(Reconstruction, ComputeBoundsAndCentroidEmpty) {
  Reconstruction reconstruction;
  const Eigen::Vector3d centroid = reconstruction.ComputeCentroid(0.0, 1.0);
  const Eigen::AlignedBox3d bbox = reconstruction.ComputeBoundingBox(0.0, 1.0);
  EXPECT_NEAR(centroid(0), 0, 1e-6);
  EXPECT_NEAR(centroid(1), 0, 1e-6);
  EXPECT_NEAR(centroid(2), 0, 1e-6);
  EXPECT_NEAR(bbox.min().x(), 0, 1e-6);
  EXPECT_NEAR(bbox.min().y(), 0, 1e-6);
  EXPECT_NEAR(bbox.min().z(), 0, 1e-6);
  EXPECT_NEAR(bbox.max().x(), 0, 1e-6);
  EXPECT_NEAR(bbox.max().y(), 0, 1e-6);
  EXPECT_NEAR(bbox.max().z(), 0, 1e-6);
}

TEST(Reconstruction, ComputeBoundsAndCentroid) {
  Reconstruction reconstruction;
  reconstruction.AddPoint3D(Eigen::Vector3d(3.0, 0.0, 0.0), Track());
  reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 3.0, 0.0), Track());
  reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 3.0), Track());
  const Eigen::Vector3d centroid = reconstruction.ComputeCentroid(0.0, 1.0);
  const Eigen::AlignedBox3d bbox = reconstruction.ComputeBoundingBox(0.0, 1.0);
  EXPECT_NEAR(centroid(0), 1.0, 1e-6);
  EXPECT_NEAR(centroid(1), 1.0, 1e-6);
  EXPECT_NEAR(centroid(2), 1.0, 1e-6);
  EXPECT_NEAR(bbox.min().x(), 0, 1e-6);
  EXPECT_NEAR(bbox.min().y(), 0, 1e-6);
  EXPECT_NEAR(bbox.min().z(), 0, 1e-6);
  EXPECT_NEAR(bbox.max().x(), 3.0, 1e-6);
  EXPECT_NEAR(bbox.max().y(), 3.0, 1e-6);
  EXPECT_NEAR(bbox.max().z(), 3.0, 1e-6);
}

TEST(Reconstruction, Crop) {
  Reconstruction reconstruction;
  GenerateReconstruction(3, &reconstruction);
  point3D_t point_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 0.0), Track());
  reconstruction.AddObservation(point_id, TrackElement(1, 1));
  point_id = reconstruction.AddPoint3D(Eigen::Vector3d(0.5, 0.5, 0.0), Track());
  reconstruction.AddObservation(point_id, TrackElement(1, 2));
  point_id = reconstruction.AddPoint3D(Eigen::Vector3d(1.0, 1.0, 0.0), Track());
  reconstruction.AddObservation(point_id, TrackElement(2, 3));
  point_id = reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 0.5), Track());
  reconstruction.AddObservation(point_id, TrackElement(2, 4));
  point_id = reconstruction.AddPoint3D(Eigen::Vector3d(0.5, 0.5, 1.0), Track());
  reconstruction.AddObservation(point_id, TrackElement(3, 5));

  // Check correct reconstruction setup
  EXPECT_EQ(reconstruction.NumCameras(), 1);
  EXPECT_EQ(reconstruction.NumImages(), 3);
  EXPECT_EQ(reconstruction.NumRegImages(), 3);
  EXPECT_EQ(reconstruction.NumPoints3D(), 5);

  std::pair<Eigen::Vector3d, Eigen::Vector3d> bbox;

  // Test emtpy reconstruction after cropping.
  const Reconstruction cropped1 = reconstruction.Crop(Eigen::AlignedBox3d(
      Eigen::Vector3d(-1, -1, -1), Eigen::Vector3d(-0.5, -0.5, -0.5)));
  EXPECT_EQ(cropped1.NumCameras(), 1);
  EXPECT_EQ(cropped1.NumImages(), 3);
  EXPECT_EQ(cropped1.NumRegImages(), 0);
  EXPECT_EQ(cropped1.NumPoints3D(), 0);

  // Test reconstruction with contents after cropping
  const Reconstruction cropped2 = reconstruction.Crop(Eigen::AlignedBox3d(
      Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(0.75, 0.75, 0.75)));
  EXPECT_EQ(cropped2.NumCameras(), 1);
  EXPECT_EQ(cropped2.NumImages(), 3);
  EXPECT_EQ(cropped2.NumRegImages(), 2);
  EXPECT_EQ(cropped2.NumPoints3D(), 3);
  EXPECT_TRUE(cropped2.Image(1).HasPose());
  EXPECT_TRUE(cropped2.Image(2).HasPose());
  EXPECT_FALSE(cropped2.Image(3).HasPose());
}

TEST(Reconstruction, Transform) {
  Reconstruction reconstruction;
  GenerateReconstruction(3, &reconstruction);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(1, 1, 1), Track());
  reconstruction.AddObservation(point3D_id, TrackElement(1, 1));
  reconstruction.AddObservation(point3D_id, TrackElement(2, 1));
  reconstruction.Transform(
      Sim3d(2, Eigen::Quaterniond::Identity(), Eigen::Vector3d(0, 1, 2)));
  EXPECT_EQ(reconstruction.Image(1).ProjectionCenter(),
            Eigen::Vector3d(0, 1, 2));
  EXPECT_EQ(reconstruction.Point3D(point3D_id).xyz, Eigen::Vector3d(2, 3, 4));
}

TEST(Reconstruction, FindImageWithName) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  EXPECT_EQ(reconstruction.FindImageWithName("image1"),
            &reconstruction.Image(1));
  EXPECT_EQ(reconstruction.FindImageWithName("image2"),
            &reconstruction.Image(2));
  EXPECT_TRUE(reconstruction.FindImageWithName("image3") == nullptr);
}

TEST(Reconstruction, FindCommonRegImageIds) {
  Reconstruction reconstruction1;
  GenerateReconstruction(5, &reconstruction1);
  Reconstruction reconstruction2;
  GenerateReconstruction(5, &reconstruction2);
  reconstruction1.DeRegisterImage(1);
  reconstruction1.Image(2).SetName("foo");
  reconstruction2.DeRegisterImage(3);
  reconstruction2.Image(4).SetName("bar");
  const auto common_image_ids =
      reconstruction1.FindCommonRegImageIds(reconstruction2);
  ASSERT_EQ(common_image_ids.size(), 1);
  EXPECT_EQ(common_image_ids[0].first, 5);
  EXPECT_EQ(common_image_ids[0].second, 5);
  EXPECT_EQ(common_image_ids,
            reconstruction2.FindCommonRegImageIds(reconstruction1));
}

TEST(Reconstruction, ComputeNumObservations) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  EXPECT_EQ(reconstruction.ComputeNumObservations(), 0);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  EXPECT_EQ(reconstruction.ComputeNumObservations(), 1);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 1));
  EXPECT_EQ(reconstruction.ComputeNumObservations(), 2);
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  EXPECT_EQ(reconstruction.ComputeNumObservations(), 3);
}

TEST(Reconstruction, ComputeMeanTrackLength) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  EXPECT_EQ(reconstruction.ComputeMeanTrackLength(), 0);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  EXPECT_EQ(reconstruction.ComputeMeanTrackLength(), 0);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  EXPECT_EQ(reconstruction.ComputeMeanTrackLength(), 1);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 1));
  EXPECT_EQ(reconstruction.ComputeMeanTrackLength(), 2);
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  EXPECT_EQ(reconstruction.ComputeMeanTrackLength(), 3);
}

TEST(Reconstruction, ComputeMeanObservationsPerRegImage) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  EXPECT_EQ(reconstruction.ComputeMeanObservationsPerRegImage(), 0);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  EXPECT_EQ(reconstruction.ComputeMeanObservationsPerRegImage(), 0);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  EXPECT_EQ(reconstruction.ComputeMeanObservationsPerRegImage(), 0.5);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 1));
  EXPECT_EQ(reconstruction.ComputeMeanObservationsPerRegImage(), 1.0);
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  EXPECT_EQ(reconstruction.ComputeMeanObservationsPerRegImage(), 1.5);
}

TEST(Reconstruction, ComputeMeanReprojectionError) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  EXPECT_EQ(reconstruction.ComputeMeanReprojectionError(), 0);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  EXPECT_EQ(reconstruction.ComputeMeanReprojectionError(), 0);
  reconstruction.Point3D(point3D_id1).error = 0.0;
  EXPECT_EQ(reconstruction.ComputeMeanReprojectionError(), 0);
  reconstruction.Point3D(point3D_id1).error = 1.0;
  EXPECT_EQ(reconstruction.ComputeMeanReprojectionError(), 1);
  reconstruction.Point3D(point3D_id1).error = 2.0;
  EXPECT_EQ(reconstruction.ComputeMeanReprojectionError(), 2.0);
}

TEST(Reconstruction, UpdatePoint3DErrors) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  EXPECT_EQ(reconstruction.ComputeMeanReprojectionError(), 0);
  Track track;
  track.AddElement(1, 0);
  reconstruction.Image(1).Point2D(0).xy = Eigen::Vector2d(0.5, 0.5);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(0, 0, 1), track);
  EXPECT_EQ(reconstruction.Point3D(point3D_id).error, -1);
  reconstruction.UpdatePoint3DErrors();
  EXPECT_EQ(reconstruction.Point3D(point3D_id).error, 0);
  reconstruction.Point3D(point3D_id).xyz = Eigen::Vector3d(0, 1, 1);
  reconstruction.UpdatePoint3DErrors();
  EXPECT_EQ(reconstruction.Point3D(point3D_id).error, 1);
}

TEST(Reconstruction, DeleteAllPoints2DAndPoints3D) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_cameras = 2;
  synthetic_dataset_options.num_images = 20;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.point2D_stddev = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  reconstruction.DeleteAllPoints2DAndPoints3D();
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  ExpectValidCameraPtrs(reconstruction);
}

}  // namespace
}  // namespace colmap

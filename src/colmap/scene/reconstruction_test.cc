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

#include "colmap/scene/reconstruction.h"

#include "colmap/geometry/pose.h"
#include "colmap/geometry/sim3.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/sensor/models.h"

#include <gtest/gtest.h>

namespace colmap {

void GenerateReconstruction(const image_t num_images,
                            Reconstruction* reconstruction) {
  const size_t kNumPoints2D = 10;

  Camera camera;
  camera.SetCameraId(1);
  camera.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction->AddCamera(camera);

  for (image_t image_id = 1; image_id <= num_images; ++image_id) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(camera.CameraId());
    image.SetName("image" + std::to_string(image_id));
    image.SetPoints2D(
        std::vector<Eigen::Vector2d>(kNumPoints2D, Eigen::Vector2d::Zero()));
    image.SetRegistered(true);
    reconstruction->AddImage(std::move(image));
  }
}

TEST(Reconstruction, Empty) {
  Reconstruction reconstruction;
  EXPECT_EQ(reconstruction.NumCameras(), 0);
  EXPECT_EQ(reconstruction.NumImages(), 0);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  EXPECT_EQ(reconstruction.NumImagePairs(), 0);
}

TEST(Reconstruction, AddCamera) {
  Reconstruction reconstruction;
  Camera camera;
  camera.SetCameraId(1);
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1, 1, 1);
  reconstruction.AddCamera(camera);
  EXPECT_TRUE(reconstruction.ExistsCamera(camera.CameraId()));
  EXPECT_EQ(reconstruction.Camera(camera.CameraId()).CameraId(),
            camera.CameraId());
  EXPECT_EQ(reconstruction.Cameras().count(camera.CameraId()), 1);
  EXPECT_EQ(reconstruction.Cameras().size(), 1);
  EXPECT_EQ(reconstruction.NumCameras(), 1);
  EXPECT_EQ(reconstruction.NumImages(), 0);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  EXPECT_EQ(reconstruction.NumImagePairs(), 0);
}

TEST(Reconstruction, AddImage) {
  Reconstruction reconstruction;
  Image image;
  image.SetImageId(1);
  reconstruction.AddImage(image);
  EXPECT_TRUE(reconstruction.ExistsImage(1));
  EXPECT_EQ(reconstruction.Image(1).ImageId(), 1);
  EXPECT_FALSE(reconstruction.Image(1).IsRegistered());
  EXPECT_EQ(reconstruction.Images().count(1), 1);
  EXPECT_EQ(reconstruction.Images().size(), 1);
  EXPECT_EQ(reconstruction.NumCameras(), 0);
  EXPECT_EQ(reconstruction.NumImages(), 1);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  EXPECT_EQ(reconstruction.NumImagePairs(), 0);
}

TEST(Reconstruction, AddPoint3D) {
  Reconstruction reconstruction;
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  EXPECT_TRUE(reconstruction.ExistsPoint3D(point3D_id));
  EXPECT_EQ(reconstruction.Point3D(point3D_id).Track().Length(), 0);
  EXPECT_EQ(reconstruction.Points3D().count(point3D_id), 1);
  EXPECT_EQ(reconstruction.Points3D().size(), 1);
  EXPECT_EQ(reconstruction.NumCameras(), 0);
  EXPECT_EQ(reconstruction.NumImages(), 0);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  EXPECT_EQ(reconstruction.NumImagePairs(), 0);
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
  EXPECT_EQ(reconstruction.Point3D(point3D_id).Track().Length(), 2);
  reconstruction.AddObservation(point3D_id, TrackElement(3, 2));
  EXPECT_EQ(reconstruction.Image(3).NumPoints3D(), 1);
  EXPECT_TRUE(reconstruction.Image(3).Point2D(2).HasPoint3D());
  EXPECT_EQ(reconstruction.Point3D(point3D_id).Track().Length(), 3);
}

TEST(Reconstruction, MergePoints3D) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d(0, 0, 0), Track());
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  reconstruction.Point3D(point3D_id1).Color() =
      Eigen::Matrix<uint8_t, 3, 1>(0, 0, 0);
  const point3D_t point3D_id2 =
      reconstruction.AddPoint3D(Eigen::Vector3d(1, 1, 1), Track());
  reconstruction.AddObservation(point3D_id2, TrackElement(1, 1));
  reconstruction.AddObservation(point3D_id2, TrackElement(2, 1));
  reconstruction.Point3D(point3D_id2).Color() =
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
                  .XYZ()
                  .isApprox(Eigen::Vector3d(0.5, 0.5, 0.5)));
  EXPECT_EQ(reconstruction.Point3D(merged_point3D_id).Color(),
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
  EXPECT_EQ(reconstruction.Point3D(point3D_id).Track().Length(), 2);
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
  EXPECT_TRUE(reconstruction.Image(1).IsRegistered());
  EXPECT_TRUE(reconstruction.IsImageRegistered(1));
  reconstruction.RegisterImage(1);
  EXPECT_EQ(reconstruction.NumRegImages(), 1);
  EXPECT_TRUE(reconstruction.Image(1).IsRegistered());
  EXPECT_TRUE(reconstruction.IsImageRegistered(1));
  reconstruction.DeRegisterImage(1);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
  EXPECT_FALSE(reconstruction.Image(1).IsRegistered());
  EXPECT_FALSE(reconstruction.IsImageRegistered(1));
}

TEST(Reconstruction, Normalize) {
  Reconstruction reconstruction;
  GenerateReconstruction(3, &reconstruction);
  reconstruction.Image(1).CamFromWorld().translation.z() = -10.0;
  reconstruction.Image(2).CamFromWorld().translation.z() = 0.0;
  reconstruction.Image(3).CamFromWorld().translation.z() = 10.0;
  reconstruction.DeRegisterImage(1);
  reconstruction.DeRegisterImage(2);
  reconstruction.DeRegisterImage(3);
  reconstruction.Normalize();
  EXPECT_NEAR(
      reconstruction.Image(1).CamFromWorld().translation.z(), -10, 1e-6);
  EXPECT_NEAR(reconstruction.Image(2).CamFromWorld().translation.z(), 0, 1e-6);
  EXPECT_NEAR(reconstruction.Image(3).CamFromWorld().translation.z(), 10, 1e-6);
  reconstruction.RegisterImage(1);
  reconstruction.RegisterImage(2);
  reconstruction.RegisterImage(3);
  reconstruction.Normalize();
  EXPECT_NEAR(reconstruction.Image(1).CamFromWorld().translation.z(), -5, 1e-6);
  EXPECT_NEAR(reconstruction.Image(2).CamFromWorld().translation.z(), 0, 1e-6);
  EXPECT_NEAR(reconstruction.Image(3).CamFromWorld().translation.z(), 5, 1e-6);
  reconstruction.Normalize(5);
  EXPECT_NEAR(
      reconstruction.Image(1).CamFromWorld().translation.z(), -2.5, 1e-6);
  EXPECT_NEAR(reconstruction.Image(2).CamFromWorld().translation.z(), 0, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(3).CamFromWorld().translation.z(), 2.5, 1e-6);
  reconstruction.Normalize(10, 0.0, 1.0);
  EXPECT_NEAR(reconstruction.Image(1).CamFromWorld().translation.z(), -5, 1e-6);
  EXPECT_NEAR(reconstruction.Image(2).CamFromWorld().translation.z(), 0, 1e-6);
  EXPECT_NEAR(reconstruction.Image(3).CamFromWorld().translation.z(), 5, 1e-6);
  reconstruction.Normalize(20);
  Image image;
  image.SetImageId(4);
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
  reconstruction.Normalize(10, 0.0, 1.0);
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

TEST(Reconstruction, ComputeBoundsAndCentroid) {
  Reconstruction reconstruction;

  // Test emtpy reconstruction first
  auto centroid = reconstruction.ComputeCentroid(0.0, 1.0);
  auto bbox = reconstruction.ComputeBoundingBox(0.0, 1.0);
  EXPECT_LT(std::abs(centroid(0)), 1e-6);
  EXPECT_LT(std::abs(centroid(1)), 1e-6);
  EXPECT_LT(std::abs(centroid(2)), 1e-6);
  EXPECT_LT(std::abs(bbox.first(0)), 1e-6);
  EXPECT_LT(std::abs(bbox.first(1)), 1e-6);
  EXPECT_LT(std::abs(bbox.first(2)), 1e-6);
  EXPECT_LT(std::abs(bbox.second(0)), 1e-6);
  EXPECT_LT(std::abs(bbox.second(1)), 1e-6);
  EXPECT_LT(std::abs(bbox.second(2)), 1e-6);

  // Test reconstruction with 3D points
  reconstruction.AddPoint3D(Eigen::Vector3d(3.0, 0.0, 0.0), Track());
  reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 3.0, 0.0), Track());
  reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 3.0), Track());
  centroid = reconstruction.ComputeCentroid(0.0, 1.0);
  bbox = reconstruction.ComputeBoundingBox(0.0, 1.0);
  EXPECT_LT(std::abs(centroid(0) - 1.0), 1e-6);
  EXPECT_LT(std::abs(centroid(1) - 1.0), 1e-6);
  EXPECT_LT(std::abs(centroid(2) - 1.0), 1e-6);
  EXPECT_LT(std::abs(bbox.first(0)), 1e-6);
  EXPECT_LT(std::abs(bbox.first(1)), 1e-6);
  EXPECT_LT(std::abs(bbox.first(2)), 1e-6);
  EXPECT_LT(std::abs(bbox.second(0) - 3.0), 1e-6);
  EXPECT_LT(std::abs(bbox.second(1) - 3.0), 1e-6);
  EXPECT_LT(std::abs(bbox.second(2) - 3.0), 1e-6);
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
  bbox.first = Eigen::Vector3d(-1, -1, -1);
  bbox.second = Eigen::Vector3d(-0.5, -0.5, -0.5);
  Reconstruction cropped1 = reconstruction.Crop(bbox);
  EXPECT_EQ(cropped1.NumCameras(), 1);
  EXPECT_EQ(cropped1.NumImages(), 3);
  EXPECT_EQ(cropped1.NumRegImages(), 0);
  EXPECT_EQ(cropped1.NumPoints3D(), 0);

  // Test reconstruction with contents after cropping
  bbox.first = Eigen::Vector3d(0.0, 0.0, 0.0);
  bbox.second = Eigen::Vector3d(0.75, 0.75, 0.75);
  Reconstruction recon2 = reconstruction.Crop(bbox);
  EXPECT_EQ(recon2.NumCameras(), 1);
  EXPECT_EQ(recon2.NumImages(), 3);
  EXPECT_EQ(recon2.NumRegImages(), 2);
  EXPECT_EQ(recon2.NumPoints3D(), 3);
  EXPECT_TRUE(recon2.IsImageRegistered(1));
  EXPECT_TRUE(recon2.IsImageRegistered(2));
  EXPECT_FALSE(recon2.IsImageRegistered(3));
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
  EXPECT_EQ(reconstruction.Point3D(point3D_id).XYZ(), Eigen::Vector3d(2, 3, 4));
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

TEST(Reconstruction, FilterPoints3D) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3D(0.0, 0.0, std::unordered_set<point3D_t>{});
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3D(
      0.0, 0.0, std::unordered_set<point3D_t>{point3D_id1 + 1});
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3D(
      0.0, 0.0, std::unordered_set<point3D_t>{point3D_id1});
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id2 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  reconstruction.AddObservation(point3D_id2, TrackElement(1, 0));
  reconstruction.FilterPoints3D(
      0.0, 0.0, std::unordered_set<point3D_t>{point3D_id2});
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id3 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.5, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id3, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id3, TrackElement(2, 0));
  reconstruction.FilterPoints3D(
      0.0, 0.0, std::unordered_set<point3D_t>{point3D_id3});
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3D(
      0.0, 1e-3, std::unordered_set<point3D_t>{point3D_id3});
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id4 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.6, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id4, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id4, TrackElement(2, 0));
  reconstruction.FilterPoints3D(
      0.1, 0.0, std::unordered_set<point3D_t>{point3D_id4});
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3D(
      0.09, 0.0, std::unordered_set<point3D_t>{point3D_id4});
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
}

TEST(Reconstruction, FilterPoints3DInImages) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3DInImages(
      0.0, 0.0, std::unordered_set<image_t>{});
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3DInImages(
      0.0, 0.0, std::unordered_set<image_t>{1});
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  reconstruction.FilterPoints3DInImages(
      0.0, 0.0, std::unordered_set<image_t>{2});
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3DInImages(
      0.0, 0.0, std::unordered_set<image_t>{1});
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id3 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.5, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id3, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id3, TrackElement(2, 0));
  reconstruction.FilterPoints3DInImages(
      0.0, 0.0, std::unordered_set<image_t>{1});
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3DInImages(
      0.0, 1e-3, std::unordered_set<image_t>{1});
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id4 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.6, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id4, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id4, TrackElement(2, 0));
  reconstruction.FilterPoints3DInImages(
      0.1, 0.0, std::unordered_set<image_t>{1});
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3DInImages(
      0.09, 0.0, std::unordered_set<image_t>{1});
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
}

TEST(Reconstruction, FilterAllPoints) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterAllPoints3D(0.0, 0.0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id2 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  reconstruction.AddObservation(point3D_id2, TrackElement(1, 0));
  reconstruction.FilterAllPoints3D(0.0, 0.0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id3 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.5, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id3, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id3, TrackElement(2, 0));
  reconstruction.FilterAllPoints3D(0.0, 0.0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterAllPoints3D(0.0, 1e-3);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id4 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.6, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id4, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id4, TrackElement(2, 0));
  reconstruction.FilterAllPoints3D(0.1, 0.0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterAllPoints3D(0.09, 0.0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
}

TEST(Reconstruction, FilterObservationsWithNegativeDepth) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d(0, 0, 1), Track());
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterObservationsWithNegativeDepth();
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.Point3D(point3D_id1).XYZ(2) = 0.001;
  reconstruction.FilterObservationsWithNegativeDepth();
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.Point3D(point3D_id1).XYZ(2) = 0.0;
  reconstruction.FilterObservationsWithNegativeDepth();
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  reconstruction.Point3D(point3D_id1).XYZ(2) = 0.001;
  reconstruction.FilterObservationsWithNegativeDepth();
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.Point3D(point3D_id1).XYZ(2) = 0.0;
  reconstruction.FilterObservationsWithNegativeDepth();
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
}

TEST(Reconstruction, FilterImages) {
  Reconstruction reconstruction;
  GenerateReconstruction(4, &reconstruction);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  reconstruction.AddObservation(point3D_id1, TrackElement(3, 0));
  reconstruction.FilterImages(0.0, 10.0, 1.0);
  EXPECT_EQ(reconstruction.NumRegImages(), 3);
  reconstruction.DeleteObservation(3, 0);
  reconstruction.FilterImages(0.0, 10.0, 1.0);
  EXPECT_EQ(reconstruction.NumRegImages(), 2);
  reconstruction.FilterImages(0.0, 0.9, 1.0);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
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
  reconstruction.Point3D(point3D_id1).SetError(0.0);
  EXPECT_EQ(reconstruction.ComputeMeanReprojectionError(), 0);
  reconstruction.Point3D(point3D_id1).SetError(1.0);
  EXPECT_EQ(reconstruction.ComputeMeanReprojectionError(), 1);
  reconstruction.Point3D(point3D_id1).SetError(2.0);
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
  EXPECT_EQ(reconstruction.Point3D(point3D_id).Error(), -1);
  reconstruction.UpdatePoint3DErrors();
  EXPECT_EQ(reconstruction.Point3D(point3D_id).Error(), 0);
  reconstruction.Point3D(point3D_id).SetXYZ(Eigen::Vector3d(0, 1, 1));
  reconstruction.UpdatePoint3DErrors();
  EXPECT_EQ(reconstruction.Point3D(point3D_id).Error(), 1);
}

}  // namespace colmap

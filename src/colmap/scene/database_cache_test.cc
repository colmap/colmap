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

#include "colmap/scene/database_cache.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(DatabaseCache, Empty) {
  DatabaseCache cache;
  EXPECT_EQ(cache.NumRigs(), 0);
  EXPECT_EQ(cache.NumCameras(), 0);
  EXPECT_EQ(cache.NumFrames(), 0);
  EXPECT_EQ(cache.NumImages(), 0);
  EXPECT_EQ(cache.NumPosePriors(), 0);
}

TEST(DatabaseCache, ConstructFromDatabase) {
  Database database(Database::kInMemoryDatabasePath);
  Camera camera1 = Camera::CreateFromModelId(
      kInvalidCameraId, SimplePinholeCameraModel::model_id, 1, 1, 1);
  camera1.camera_id = database.WriteCamera(camera1);
  Camera camera2 = Camera::CreateFromModelId(
      kInvalidCameraId, SimplePinholeCameraModel::model_id, 2, 2, 2);
  camera2.camera_id = database.WriteCamera(camera2);

  Rig rig;
  rig.AddRefSensor(sensor_t(SensorType::CAMERA, camera1.camera_id));
  rig.AddSensor(sensor_t(SensorType::CAMERA, camera2.camera_id));
  const rig_t rig_id = database.WriteRig(rig);

  Image image1;
  image1.SetName("image1");
  image1.SetCameraId(camera1.camera_id);
  image1.SetImageId(database.WriteImage(image1));
  Image image2;
  image2.SetName("image2");
  image2.SetCameraId(camera2.camera_id);
  image2.SetImageId(database.WriteImage(image2));
  Image image3;
  image3.SetName("image3");
  image3.SetCameraId(camera1.camera_id);
  image3.SetImageId(database.WriteImage(image3));
  Image image4;
  image4.SetName("image4");
  image4.SetCameraId(camera2.camera_id);
  image4.SetImageId(database.WriteImage(image4));

  Frame frame1;
  frame1.SetRigId(rig_id);
  frame1.AddDataId(data_t(sensor_t(SensorType::CAMERA, camera1.camera_id),
                          image1.ImageId()));
  frame1.AddDataId(data_t(sensor_t(SensorType::CAMERA, camera2.camera_id),
                          image2.ImageId()));
  frame1.SetFrameId(database.WriteFrame(frame1));
  Frame frame2;
  frame2.SetRigId(rig_id);
  frame2.AddDataId(data_t(sensor_t(SensorType::CAMERA, camera1.camera_id),
                          image3.ImageId()));
  frame2.AddDataId(data_t(sensor_t(SensorType::CAMERA, camera2.camera_id),
                          image4.ImageId()));
  frame2.SetFrameId(database.WriteFrame(frame2));

  database.WritePosePrior(image1.ImageId(),
                          PosePrior(Eigen::Vector3d::Random()));
  database.WritePosePrior(image2.ImageId(),
                          PosePrior(Eigen::Vector3d::Random()));

  database.WriteKeypoints(image1.ImageId(), FeatureKeypoints(10));
  database.WriteKeypoints(image2.ImageId(), FeatureKeypoints(5));
  database.WriteKeypoints(image3.ImageId(), FeatureKeypoints(6));
  database.WriteKeypoints(image4.ImageId(), FeatureKeypoints(3));

  TwoViewGeometry two_view_geometry;
  two_view_geometry.inlier_matches = {{0, 1}};
  two_view_geometry.config =
      TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC;
  two_view_geometry.F = Eigen::Matrix3d::Random();
  two_view_geometry.E = Eigen::Matrix3d::Random();
  two_view_geometry.H = Eigen::Matrix3d::Random();
  two_view_geometry.cam2_from_cam1 =
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
  database.WriteTwoViewGeometry(
      image1.ImageId(), image2.ImageId(), two_view_geometry);
  database.WriteTwoViewGeometry(
      image2.ImageId(), image3.ImageId(), two_view_geometry);
  database.WriteTwoViewGeometry(
      image3.ImageId(), image4.ImageId(), two_view_geometry);

  auto cache = DatabaseCache::Create(database,
                                     /*min_num_matches=*/0,
                                     /*ignore_watermarks=*/false,
                                     /*image_names=*/{});

  EXPECT_EQ(cache->NumRigs(), 1);
  EXPECT_EQ(cache->NumCameras(), 2);
  EXPECT_EQ(cache->NumFrames(), 2);
  EXPECT_EQ(cache->NumImages(), 4);
  EXPECT_EQ(cache->NumPosePriors(), 2);

  EXPECT_TRUE(cache->ExistsCamera(camera1.camera_id));
  EXPECT_EQ(cache->Camera(camera1.camera_id), camera1);
  EXPECT_TRUE(cache->ExistsCamera(camera2.camera_id));
  EXPECT_EQ(cache->Camera(camera2.camera_id), camera2);

  EXPECT_TRUE(cache->ExistsFrame(frame1.FrameId()));
  EXPECT_EQ(cache->Frame(frame1.FrameId()), frame1);
  EXPECT_TRUE(cache->ExistsFrame(frame2.FrameId()));
  EXPECT_EQ(cache->Frame(frame2.FrameId()), frame2);

  EXPECT_TRUE(cache->ExistsImage(image1.ImageId()));
  EXPECT_EQ(cache->Image(image1.ImageId()).NumPoints2D(), 10);
  EXPECT_TRUE(cache->ExistsImage(image2.ImageId()));
  EXPECT_EQ(cache->Image(image2.ImageId()).NumPoints2D(), 5);
  EXPECT_TRUE(cache->ExistsImage(image3.ImageId()));
  EXPECT_EQ(cache->Image(image3.ImageId()).NumPoints2D(), 6);
  EXPECT_TRUE(cache->ExistsImage(image4.ImageId()));
  EXPECT_EQ(cache->Image(image4.ImageId()).NumPoints2D(), 3);

  EXPECT_TRUE(cache->ExistsPosePrior(image1.ImageId()));
  EXPECT_TRUE(cache->PosePrior(image1.ImageId()).IsValid());
  EXPECT_TRUE(cache->ExistsPosePrior(image2.ImageId()));
  EXPECT_TRUE(cache->PosePrior(image2.ImageId()).IsValid());
  EXPECT_FALSE(cache->ExistsPosePrior(image3.ImageId()));
  EXPECT_FALSE(cache->ExistsPosePrior(image4.ImageId()));

  const auto correspondence_graph = cache->CorrespondenceGraph();
  EXPECT_TRUE(correspondence_graph->ExistsImage(image1.ImageId()));
  EXPECT_EQ(correspondence_graph->NumCorrespondencesForImage(image1.ImageId()),
            1);
  EXPECT_EQ(correspondence_graph->NumObservationsForImage(image1.ImageId()), 1);
  EXPECT_TRUE(correspondence_graph->ExistsImage(image2.ImageId()));
  EXPECT_EQ(correspondence_graph->NumCorrespondencesForImage(image2.ImageId()),
            2);
  EXPECT_EQ(correspondence_graph->NumObservationsForImage(image2.ImageId()), 2);
  EXPECT_EQ(correspondence_graph->NumCorrespondencesForImage(image3.ImageId()),
            2);
  EXPECT_EQ(correspondence_graph->NumObservationsForImage(image3.ImageId()), 2);
  EXPECT_EQ(correspondence_graph->NumCorrespondencesForImage(image4.ImageId()),
            1);
  EXPECT_EQ(correspondence_graph->NumObservationsForImage(image4.ImageId()), 1);
}

TEST(DatabaseCache, ConstructFromLegacyDatabaseWithoutRigsAndFrames) {
  Database database(Database::kInMemoryDatabasePath);
  const Camera camera = Camera::CreateFromModelId(
      kInvalidCameraId, SimplePinholeCameraModel::model_id, 1, 1, 1);
  const camera_t camera_id = database.WriteCamera(camera);
  Image image1;
  image1.SetName("image1");
  image1.SetCameraId(camera_id);
  Image image2;
  image2.SetName("image2");
  image2.SetCameraId(camera_id);
  const image_t image_id1 = database.WriteImage(image1);
  const image_t image_id2 = database.WriteImage(image2);
  database.WritePosePrior(image_id1, PosePrior(Eigen::Vector3d::Random()));
  database.WritePosePrior(image_id2, PosePrior(Eigen::Vector3d::Random()));
  database.WriteKeypoints(image_id1, FeatureKeypoints(10));
  database.WriteKeypoints(image_id2, FeatureKeypoints(5));
  TwoViewGeometry two_view_geometry;
  two_view_geometry.inlier_matches = {{0, 1}};
  two_view_geometry.config =
      TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC;
  two_view_geometry.F = Eigen::Matrix3d::Random();
  two_view_geometry.E = Eigen::Matrix3d::Random();
  two_view_geometry.H = Eigen::Matrix3d::Random();
  two_view_geometry.cam2_from_cam1 =
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
  database.WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
  auto cache = DatabaseCache::Create(database,
                                     /*min_num_matches=*/0,
                                     /*ignore_watermarks=*/false,
                                     /*image_names=*/{});
  EXPECT_EQ(cache->NumCameras(), 1);
  EXPECT_EQ(cache->NumImages(), 2);
  EXPECT_EQ(cache->NumPosePriors(), 2);
  EXPECT_TRUE(cache->ExistsCamera(camera_id));
  EXPECT_EQ(cache->Camera(camera_id).model_id, camera.model_id);
  EXPECT_TRUE(cache->ExistsImage(image_id1));
  EXPECT_TRUE(cache->ExistsImage(image_id2));
  EXPECT_EQ(cache->Image(image_id1).NumPoints2D(), 10);
  EXPECT_EQ(cache->Image(image_id2).NumPoints2D(), 5);
  EXPECT_TRUE(cache->PosePrior(image_id1).IsValid());
  EXPECT_TRUE(cache->PosePrior(image_id2).IsValid());
  const auto correspondence_graph = cache->CorrespondenceGraph();
  EXPECT_TRUE(cache->CorrespondenceGraph()->ExistsImage(image_id1));
  EXPECT_EQ(cache->CorrespondenceGraph()->NumCorrespondencesForImage(image_id1),
            1);
  EXPECT_EQ(cache->CorrespondenceGraph()->NumObservationsForImage(image_id1),
            1);
  EXPECT_TRUE(cache->CorrespondenceGraph()->ExistsImage(image_id2));
  EXPECT_EQ(cache->CorrespondenceGraph()->NumCorrespondencesForImage(image_id2),
            1);
  EXPECT_EQ(cache->CorrespondenceGraph()->NumObservationsForImage(image_id2),
            1);
}

TEST(DatabaseCache, ConstructFromCustom) {
  DatabaseCache cache;
  EXPECT_EQ(cache.NumRigs(), 0);
  EXPECT_EQ(cache.NumCameras(), 0);
  EXPECT_EQ(cache.NumFrames(), 0);
  EXPECT_EQ(cache.NumImages(), 0);
  EXPECT_EQ(cache.NumPosePriors(), 0);

  constexpr rig_t kRigId = 41;
  Rig rig;
  rig.SetRigId(kRigId);
  cache.AddRig(rig);

  constexpr camera_t kCameraId = 42;
  cache.AddCamera(Camera::CreateFromModelId(
      /*camera_id*/ kCameraId, SimplePinholeCameraModel::model_id, 1, 1, 1));

  constexpr frame_t kFrameId = 43;
  Frame frame;
  frame.SetFrameId(kFrameId);
  cache.AddFrame(frame);

  constexpr image_t kImageId = 44;
  Image image;
  image.SetImageId(kImageId);
  image.SetName("image");
  image.SetCameraId(kCameraId);
  cache.AddImage(image);
  cache.AddPosePrior(kImageId, PosePrior(Eigen::Vector3d::Random()));

  EXPECT_EQ(cache.NumCameras(), 1);
  EXPECT_EQ(cache.NumImages(), 1);
  EXPECT_EQ(cache.NumPosePriors(), 1);
  EXPECT_TRUE(cache.ExistsRig(kRigId));
  EXPECT_TRUE(cache.ExistsCamera(kCameraId));
  EXPECT_TRUE(cache.ExistsFrame(kFrameId));
  EXPECT_TRUE(cache.ExistsImage(kImageId));
  EXPECT_TRUE(cache.ExistsPosePrior(kImageId));
}

}  // namespace
}  // namespace colmap

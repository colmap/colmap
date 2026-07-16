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

#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/util/eigen_matchers.h"

#include <cmath>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

std::shared_ptr<Database> CreateTestDatabase() {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Camera camera1 = Camera::CreateFromModelId(
      kInvalidCameraId, SimplePinholeCameraModel::model_id, 1, 1, 1);
  camera1.camera_id = database->WriteCamera(camera1);
  Camera camera2 = Camera::CreateFromModelId(
      kInvalidCameraId, SimplePinholeCameraModel::model_id, 2, 2, 2);
  camera2.camera_id = database->WriteCamera(camera2);

  Rig rig;
  rig.AddRefSensor(camera1.SensorId());
  rig.AddSensor(camera2.SensorId());
  const rig_t rig_id = database->WriteRig(rig);

  Image image1;
  image1.SetName("image1");
  image1.SetCameraId(camera1.camera_id);
  image1.SetImageId(database->WriteImage(image1));
  Image image2;
  image2.SetName("image2");
  image2.SetCameraId(camera2.camera_id);
  image2.SetImageId(database->WriteImage(image2));
  Image image3;
  image3.SetName("image3");
  image3.SetCameraId(camera1.camera_id);
  image3.SetImageId(database->WriteImage(image3));
  Image image4;
  image4.SetName("image4");
  image4.SetCameraId(camera2.camera_id);
  image4.SetImageId(database->WriteImage(image4));

  PosePrior pose_prior1;
  pose_prior1.corr_data_id = image1.DataId();
  pose_prior1.position = Eigen::Vector3d::Random();
  pose_prior1.pose_prior_id = database->WritePosePrior(pose_prior1);
  PosePrior pose_prior2;
  pose_prior1.corr_data_id = image2.DataId();
  pose_prior2.position = Eigen::Vector3d::Random();
  pose_prior2.pose_prior_id = database->WritePosePrior(pose_prior2);

  Frame frame1;
  frame1.SetRigId(rig_id);
  frame1.AddDataId(image1.DataId());
  frame1.AddDataId(image2.DataId());
  frame1.SetFrameId(database->WriteFrame(frame1));
  Frame frame2;
  frame2.SetRigId(rig_id);
  frame2.AddDataId(image3.DataId());
  frame2.AddDataId(image4.DataId());
  frame2.SetFrameId(database->WriteFrame(frame2));

  database->WriteKeypoints(image1.ImageId(), FeatureKeypoints(10));
  database->WriteKeypoints(image2.ImageId(), FeatureKeypoints(5));
  database->WriteKeypoints(image3.ImageId(), FeatureKeypoints(6));
  database->WriteKeypoints(image4.ImageId(), FeatureKeypoints(3));

  TwoViewGeometry two_view_geometry;
  two_view_geometry.inlier_matches = {{0, 1}};
  two_view_geometry.config =
      TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC;
  two_view_geometry.F = Eigen::Matrix3d::Random();
  two_view_geometry.E = Eigen::Matrix3d::Random();
  two_view_geometry.H = Eigen::Matrix3d::Random();
  two_view_geometry.cam2_from_cam1 =
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
  database->WriteTwoViewGeometry(
      image1.ImageId(), image2.ImageId(), two_view_geometry);
  database->WriteTwoViewGeometry(
      image2.ImageId(), image3.ImageId(), two_view_geometry);
  database->WriteTwoViewGeometry(
      image3.ImageId(), image4.ImageId(), two_view_geometry);

  return database;
}

TEST(DatabaseCache, Empty) {
  DatabaseCache cache;
  EXPECT_EQ(cache.NumRigs(), 0);
  EXPECT_EQ(cache.NumCameras(), 0);
  EXPECT_EQ(cache.NumFrames(), 0);
  EXPECT_EQ(cache.NumImages(), 0);
  EXPECT_EQ(cache.NumPosePriors(), 0);
}

TEST(DatabaseCache, ConstructFromDatabase) {
  auto database = CreateTestDatabase();
  auto cache = DatabaseCache::Create(*database, {});

  EXPECT_EQ(cache->NumRigs(), 1);
  EXPECT_EQ(cache->NumCameras(), 2);
  EXPECT_EQ(cache->NumFrames(), 2);
  EXPECT_EQ(cache->NumImages(), 4);
  EXPECT_EQ(cache->NumPosePriors(), 2);

  for (const Rig& rig : database->ReadAllRigs()) {
    EXPECT_TRUE(cache->ExistsRig(rig.RigId()));
    EXPECT_EQ(cache->Rig(rig.RigId()), rig);
  }

  for (const Camera& camera : database->ReadAllCameras()) {
    EXPECT_TRUE(cache->ExistsCamera(camera.camera_id));
    EXPECT_EQ(cache->Camera(camera.camera_id), camera);
  }

  for (const Frame& frame : database->ReadAllFrames()) {
    EXPECT_TRUE(cache->ExistsFrame(frame.FrameId()));
    EXPECT_EQ(cache->Frame(frame.FrameId()), frame);
  }

  const std::vector<Image> images = database->ReadAllImages();
  EXPECT_TRUE(cache->ExistsImage(images[0].ImageId()));
  EXPECT_EQ(cache->Image(images[0].ImageId()).NumPoints2D(), 10);
  EXPECT_TRUE(cache->ExistsImage(images[1].ImageId()));
  EXPECT_EQ(cache->Image(images[1].ImageId()).NumPoints2D(), 5);
  EXPECT_TRUE(cache->ExistsImage(images[2].ImageId()));
  EXPECT_EQ(cache->Image(images[2].ImageId()).NumPoints2D(), 6);
  EXPECT_TRUE(cache->ExistsImage(images[3].ImageId()));
  EXPECT_EQ(cache->Image(images[3].ImageId()).NumPoints2D(), 3);

  EXPECT_EQ(cache->PosePriors(), database->ReadAllPosePriors());

  const auto correspondence_graph = cache->CorrespondenceGraph();
  EXPECT_TRUE(correspondence_graph->ExistsImage(images[0].ImageId()));
  EXPECT_EQ(
      correspondence_graph->NumCorrespondencesForImage(images[0].ImageId()), 1);
  EXPECT_EQ(correspondence_graph->NumObservationsForImage(images[0].ImageId()),
            1);
  EXPECT_TRUE(correspondence_graph->ExistsImage(images[1].ImageId()));
  EXPECT_EQ(
      correspondence_graph->NumCorrespondencesForImage(images[1].ImageId()), 2);
  EXPECT_EQ(correspondence_graph->NumObservationsForImage(images[1].ImageId()),
            2);
  EXPECT_EQ(
      correspondence_graph->NumCorrespondencesForImage(images[2].ImageId()), 2);
  EXPECT_EQ(correspondence_graph->NumObservationsForImage(images[2].ImageId()),
            2);
  EXPECT_EQ(
      correspondence_graph->NumCorrespondencesForImage(images[3].ImageId()), 1);
  EXPECT_EQ(correspondence_graph->NumObservationsForImage(images[3].ImageId()),
            1);

  // Extends this existing test case to also cover
  // DatabaseCache::ConvertPosePriorsToENU's shared-ENU reference selection
  // and covariance-basis rotation. At lat=0, lon=90, GPSTransform::ENUFromECEF
  // reduces to the hand-derived signed permutation matrix
  // R90=[[-1,0,0],[0,0,1],[0,1,0]] (its own transpose and inverse), and at
  // lat=0, lon=0 to R0=[[0,1,0],[0,0,1],[1,0,0]]; the row-local-to-shared-ENU
  // rotation for a row at (0,0) with reference (0,90) is therefore
  // S = R90 * R0^T = [[0,0,-1],[0,1,0],[1,0,0]] (checked by hand, not by
  // calling GPSTransform), which conjugates the diagonal covariance
  // diag(1,4,9) into diag(9,4,1) (axes 0 and 2 swap).
  auto wgs84_database = Database::Open(kInMemorySqliteDatabasePath);

  PosePrior ref_prior;
  ref_prior.corr_data_id = data_t(sensor_t(SensorType::CAMERA, 1), 101);
  ref_prior.coordinate_system = PosePrior::CoordinateSystem::WGS84;
  ref_prior.position = Eigen::Vector3d(0.0, 90.0, 500.0);
  wgs84_database->WritePosePrior(ref_prior);

  PosePrior other_prior;
  other_prior.corr_data_id = data_t(sensor_t(SensorType::CAMERA, 1), 102);
  other_prior.coordinate_system = PosePrior::CoordinateSystem::WGS84;
  other_prior.position = Eigen::Vector3d(0.0, 0.0, PosePrior::kNaN);
  other_prior.position_covariance = Eigen::Vector3d(1.0, 4.0, 9.0).asDiagonal();
  wgs84_database->WritePosePrior(other_prior);

  DatabaseCache::Options enu_options;
  enu_options.convert_pose_priors_to_enu = true;
  auto enu_cache = DatabaseCache::Create(*wgs84_database, enu_options);
  ASSERT_EQ(enu_cache->NumPosePriors(), 2);

  const PosePrior* converted_ref = nullptr;
  const PosePrior* converted_other = nullptr;
  for (const auto& prior : enu_cache->PosePriors()) {
    if (prior.corr_data_id.id == 101) {
      converted_ref = &prior;
    } else if (prior.corr_data_id.id == 102) {
      converted_other = &prior;
    }
  }
  ASSERT_NE(converted_ref, nullptr);
  ASSERT_NE(converted_other, nullptr);

  // The reference row is, by construction, the sole full-position row, so
  // the geometric median is exactly its own ECEF point and it converts to
  // the shared-ENU origin.
  EXPECT_EQ(converted_ref->coordinate_system,
            PosePrior::CoordinateSystem::CARTESIAN);
  const Eigen::Vector3d zero3 = Eigen::Vector3d::Zero();
  EXPECT_THAT(converted_ref->position, EigenMatrixNear(zero3, 1e-6));

  // The horizontal-only row at 90 degrees of longitude away: East/North are
  // finite (computed using the shared reference altitude only), Up stays
  // NaN (no altitude was fabricated), and its covariance is rotated by the
  // non-identity S derived above.
  EXPECT_EQ(converted_other->coordinate_system,
            PosePrior::CoordinateSystem::CARTESIAN);
  EXPECT_TRUE(std::isfinite(converted_other->position.x()));
  EXPECT_TRUE(std::isfinite(converted_other->position.y()));
  EXPECT_FALSE(std::isfinite(converted_other->position.z()));
  const Eigen::Matrix3d expected_cov =
      Eigen::Vector3d(9.0, 4.0, 1.0).asDiagonal();
  EXPECT_THAT(converted_other->position_covariance,
              EigenMatrixNear(expected_cov, 1e-6));

  ASSERT_TRUE(enu_cache->PosePriorEnuOrigin().has_value());
  EXPECT_NEAR(enu_cache->PosePriorEnuOrigin()->x(), 0.0, 1e-9);
  EXPECT_NEAR(enu_cache->PosePriorEnuOrigin()->y(), 90.0, 1e-9);
  EXPECT_NEAR(enu_cache->PosePriorEnuOrigin()->z(), 500.0, 1e-9);
}

TEST(DatabaseCache, ConstructFromDatabaseWithCustomImages) {
  auto database = CreateTestDatabase();

  // Note that the first two images are part of the same frame.
  const std::vector<Image> images = database->ReadAllImages();
  DatabaseCache::Options options;
  options.image_names = {images[0].Name(), images[1].Name()};
  auto cache = DatabaseCache::Create(*database, options);

  EXPECT_EQ(cache->NumRigs(), 1);
  EXPECT_EQ(cache->NumCameras(), 2);
  EXPECT_EQ(cache->NumFrames(), 1);
  EXPECT_EQ(cache->NumImages(), 2);
  EXPECT_EQ(cache->NumPosePriors(), 2);

  const auto correspondence_graph = cache->CorrespondenceGraph();
  EXPECT_TRUE(correspondence_graph->ExistsImage(images[0].ImageId()));
  EXPECT_EQ(
      correspondence_graph->NumCorrespondencesForImage(images[0].ImageId()), 1);
  EXPECT_EQ(correspondence_graph->NumObservationsForImage(images[0].ImageId()),
            1);
  EXPECT_TRUE(correspondence_graph->ExistsImage(images[1].ImageId()));
  EXPECT_EQ(
      correspondence_graph->NumCorrespondencesForImage(images[1].ImageId()), 1);
  EXPECT_EQ(correspondence_graph->NumObservationsForImage(images[1].ImageId()),
            1);
}

std::shared_ptr<Database> CreateLegacyTestDatabase() {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  const Camera camera = Camera::CreateFromModelId(
      kInvalidCameraId, SimplePinholeCameraModel::model_id, 1, 1, 1);
  const camera_t camera_id = database->WriteCamera(camera);
  Image image1;
  image1.SetName("image1");
  image1.SetCameraId(camera_id);
  Image image2;
  image2.SetName("image2");
  image2.SetCameraId(camera_id);
  Image image3;
  image3.SetName("image3");
  image3.SetCameraId(camera_id);
  const image_t image_id1 = database->WriteImage(image1);
  const image_t image_id2 = database->WriteImage(image2);
  const image_t image_id3 = database->WriteImage(image3);
  database->WriteKeypoints(image_id1, FeatureKeypoints(10));
  database->WriteKeypoints(image_id2, FeatureKeypoints(5));
  database->WriteKeypoints(image_id3, FeatureKeypoints(7));
  TwoViewGeometry two_view_geometry;
  two_view_geometry.inlier_matches = {{0, 1}};
  two_view_geometry.config =
      TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC;
  two_view_geometry.F = Eigen::Matrix3d::Random();
  two_view_geometry.E = Eigen::Matrix3d::Random();
  two_view_geometry.H = Eigen::Matrix3d::Random();
  two_view_geometry.cam2_from_cam1 =
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
  database->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
  database->WriteTwoViewGeometry(image_id1, image_id3, two_view_geometry);

  return database;
}

TEST(DatabaseCache, ConstructFromLegacyDatabaseWithoutRigsAndFrames) {
  auto database = CreateLegacyTestDatabase();
  auto cache = DatabaseCache::Create(*database, {});
  EXPECT_EQ(cache->NumCameras(), 1);
  EXPECT_EQ(cache->NumImages(), 3);
  EXPECT_EQ(cache->NumPosePriors(), 0);
  EXPECT_TRUE(cache->ExistsCamera(1));
  EXPECT_EQ(cache->Camera(1).model_id, CameraModelId::kSimplePinhole);
  EXPECT_TRUE(cache->ExistsImage(1));
  EXPECT_TRUE(cache->ExistsImage(2));
  EXPECT_TRUE(cache->ExistsImage(3));
  EXPECT_EQ(cache->Image(1).NumPoints2D(), 10);
  EXPECT_EQ(cache->Image(2).NumPoints2D(), 5);
  EXPECT_EQ(cache->Image(3).NumPoints2D(), 7);
  const auto correspondence_graph = cache->CorrespondenceGraph();
  EXPECT_TRUE(cache->CorrespondenceGraph()->ExistsImage(1));
  EXPECT_EQ(cache->CorrespondenceGraph()->NumCorrespondencesForImage(1), 2);
  EXPECT_EQ(cache->CorrespondenceGraph()->NumObservationsForImage(1), 1);
  EXPECT_TRUE(cache->CorrespondenceGraph()->ExistsImage(2));
  EXPECT_EQ(cache->CorrespondenceGraph()->NumCorrespondencesForImage(2), 1);
  EXPECT_EQ(cache->CorrespondenceGraph()->NumObservationsForImage(2), 1);
  EXPECT_TRUE(cache->CorrespondenceGraph()->ExistsImage(3));
  EXPECT_EQ(cache->CorrespondenceGraph()->NumCorrespondencesForImage(3), 1);
  EXPECT_EQ(cache->CorrespondenceGraph()->NumObservationsForImage(3), 1);
}

TEST(DatabaseCache, ConstructFromLegacyDatabaseWithCustomImages) {
  auto database = CreateLegacyTestDatabase();
  const std::vector<Image> images = database->ReadAllImages();
  DatabaseCache::Options options;
  options.image_names = {images[0].Name(), images[2].Name()};
  auto cache = DatabaseCache::Create(*database, options);
  EXPECT_EQ(cache->NumCameras(), 1);
  EXPECT_EQ(cache->NumImages(), 2);
  EXPECT_EQ(cache->NumPosePriors(), 0);
  EXPECT_TRUE(cache->ExistsCamera(1));
  EXPECT_EQ(cache->Camera(1).model_id, CameraModelId::kSimplePinhole);
  EXPECT_TRUE(cache->ExistsImage(1));
  EXPECT_TRUE(cache->ExistsImage(3));
  EXPECT_EQ(cache->Image(1).NumPoints2D(), 10);
  EXPECT_EQ(cache->Image(3).NumPoints2D(), 7);
  const auto correspondence_graph = cache->CorrespondenceGraph();
  EXPECT_TRUE(cache->CorrespondenceGraph()->ExistsImage(1));
  EXPECT_EQ(cache->CorrespondenceGraph()->NumCorrespondencesForImage(1), 1);
  EXPECT_EQ(cache->CorrespondenceGraph()->NumObservationsForImage(1), 1);
  EXPECT_TRUE(cache->CorrespondenceGraph()->ExistsImage(3));
  EXPECT_EQ(cache->CorrespondenceGraph()->NumCorrespondencesForImage(3), 1);
  EXPECT_EQ(cache->CorrespondenceGraph()->NumObservationsForImage(3), 1);
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
      /*camera_id=*/kCameraId, SimplePinholeCameraModel::model_id, 1, 1, 1));

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

  constexpr pose_prior_t kPosePriorId = 45;
  PosePrior pose_prior;
  pose_prior.position = Eigen::Vector3d::Random();
  pose_prior.pose_prior_id = kPosePriorId;
  cache.AddPosePrior(pose_prior);

  EXPECT_EQ(cache.NumCameras(), 1);
  EXPECT_EQ(cache.NumImages(), 1);
  EXPECT_EQ(cache.NumPosePriors(), 1);
  EXPECT_TRUE(cache.ExistsRig(kRigId));
  EXPECT_TRUE(cache.ExistsCamera(kCameraId));
  EXPECT_TRUE(cache.ExistsFrame(kFrameId));
  EXPECT_TRUE(cache.ExistsImage(kImageId));
  EXPECT_THAT(cache.PosePriors(), testing::ElementsAre(pose_prior));
}

TEST(DatabaseCache, NonConstCorrespondenceGraph) {
  auto database = CreateTestDatabase();
  auto cache = DatabaseCache::Create(*database, {});

  // Non-const overload returns a mutable shared_ptr.
  std::shared_ptr<CorrespondenceGraph> mutable_graph =
      cache->CorrespondenceGraph();
  EXPECT_NE(mutable_graph, nullptr);

  // Verify it can be used to update a two-view geometry.
  const auto image_pairs = mutable_graph->ImagePairs();
  ASSERT_FALSE(image_pairs.empty());
  const auto [image_id1, image_id2] = PairIdToImagePair(image_pairs[0]);

  TwoViewGeometry geom = mutable_graph->ExtractTwoViewGeometry(
      image_id1, image_id2, /*extract_inlier_matches=*/false);
  const Rigid3d new_pose(Eigen::Quaterniond::UnitRandom(),
                         Eigen::Vector3d::Random());
  geom.cam2_from_cam1 = new_pose;
  mutable_graph->UpdateTwoViewGeometry(image_id1, image_id2, geom);

  // Read back through the const accessor and verify the update is visible.
  const DatabaseCache& const_cache = *cache;
  TwoViewGeometry updated =
      const_cache.CorrespondenceGraph()->ExtractTwoViewGeometry(
          image_id1, image_id2, false);
  ASSERT_TRUE(updated.cam2_from_cam1.has_value());
  EXPECT_THAT(updated.cam2_from_cam1.value(),
              Rigid3dNear(new_pose, 1e-6, 1e-6));
}

}  // namespace
}  // namespace colmap

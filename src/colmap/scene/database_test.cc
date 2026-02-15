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

#include "colmap/scene/database.h"

#include "colmap/scene/database_sqlite.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <filesystem>
#include <thread>

#include <Eigen/Geometry>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#ifdef _WIN32
#include <windows.h>
#endif

namespace colmap {
namespace {

class ParameterizedDatabaseTests
    : public ::testing::TestWithParam<std::function<std::shared_ptr<Database>(
          const std::filesystem::path&)>> {};

TEST_P(ParameterizedDatabaseTests, OpenInMemory) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
}

TEST_P(ParameterizedDatabaseTests, OpenCloseInMemory) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  database->Close();
  // Any database operation after closing the database should fail.
  EXPECT_ANY_THROW(database->ExistsCamera(42));
  database->Close();
}

TEST_P(ParameterizedDatabaseTests, OpenFile) {
  std::shared_ptr<Database> database =
      GetParam()(CreateTestDir() / "database.db");
}

TEST_P(ParameterizedDatabaseTests, OpenCloseFile) {
  std::shared_ptr<Database> database =
      GetParam()(CreateTestDir() / "database.db");
  database->Close();
  // Any database operation after closing the database should fail.
  EXPECT_ANY_THROW(database->ExistsCamera(42));
  database->Close();
}

TEST_P(ParameterizedDatabaseTests, OpenFileWithNonASCIIPath) {
  const auto database_path = CreateTestDir() / u8"äöü時临.db";
  std::shared_ptr<Database> database = GetParam()(database_path);
  EXPECT_TRUE(ExistsPath(database_path));
}

TEST_P(ParameterizedDatabaseTests, Transaction) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  DatabaseTransaction database_transaction(database.get());
}

TEST_P(ParameterizedDatabaseTests, TransactionMultiThreaded) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);

  constexpr int kNumThreads = 3;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&database]() {
      DatabaseTransaction database_transaction(database.get());
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_P(ParameterizedDatabaseTests, Empty) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  EXPECT_EQ(database->NumCameras(), 0);
  EXPECT_EQ(database->NumFrames(), 0);
  EXPECT_EQ(database->NumImages(), 0);
  EXPECT_EQ(database->NumKeypoints(), 0);
  EXPECT_EQ(database->MaxNumKeypoints(), 0);
  EXPECT_EQ(database->NumDescriptors(), 0);
  EXPECT_EQ(database->MaxNumDescriptors(), 0);
  EXPECT_EQ(database->NumMatches(), 0);
  EXPECT_EQ(database->NumMatchedImagePairs(), 0);
  EXPECT_EQ(database->NumVerifiedImagePairs(), 0);
}

TEST_P(ParameterizedDatabaseTests, Rig) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  EXPECT_EQ(database->NumRigs(), 0);
  Rig rig;
  rig.AddRefSensor(sensor_t(SensorType::CAMERA, 1));
  rig.SetRigId(database->WriteRig(rig));
  EXPECT_EQ(database->NumRigs(), 1);
  EXPECT_TRUE(database->ExistsRig(rig.RigId()));
  EXPECT_EQ(database->ReadRig(rig.RigId()), rig);

  database->ClearRigs();
  EXPECT_EQ(database->NumRigs(), 0);

  rig.AddSensor(
      sensor_t(SensorType::CAMERA, 2),
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random()));
  rig.AddSensor(sensor_t(SensorType::IMU, 3));
  rig.AddSensor(
      sensor_t(SensorType::IMU, 4),
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random()));
  rig.SetRigId(database->WriteRig(rig));
  EXPECT_EQ(database->NumRigs(), 1);
  EXPECT_TRUE(database->ExistsRig(rig.RigId()));
  EXPECT_EQ(database->ReadRig(rig.RigId()), rig);
  EXPECT_EQ(database->ReadRigWithSensor(sensor_t(SensorType::CAMERA, 1)), rig);
  EXPECT_EQ(database->ReadRigWithSensor(sensor_t(SensorType::IMU, 4)), rig);
  EXPECT_EQ(database->ReadRigWithSensor(sensor_t(SensorType::IMU, 42)),
            std::nullopt);
  rig.SensorFromRig(sensor_t(SensorType::CAMERA, 2)) =
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
  database->UpdateRig(rig);
  EXPECT_EQ(database->ReadRig(rig.RigId()), rig);
  Rig rig2;
  rig2.AddRefSensor(sensor_t(SensorType::IMU, 10));
  rig2.SetRigId(rig.RigId() + 1);
  database->WriteRig(rig2, /*use_rig_id=*/true);
  EXPECT_EQ(database->NumRigs(), 2);
  EXPECT_TRUE(database->ExistsRig(rig.RigId()));
  EXPECT_TRUE(database->ExistsRig(rig2.RigId()));
  EXPECT_EQ(database->ReadAllRigs().size(), 2);
  EXPECT_EQ(database->ReadAllRigs()[0].RigId(), rig.RigId());
  EXPECT_EQ(database->ReadAllRigs()[1].RigId(), rig2.RigId());
  database->ClearRigs();
  EXPECT_EQ(database->NumRigs(), 0);
}

TEST_P(ParameterizedDatabaseTests, Camera) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  EXPECT_EQ(database->NumCameras(), 0);
  Camera camera = Camera::CreateFromModelName(
      kInvalidCameraId, "SIMPLE_PINHOLE", 1.0, 1, 1);
  camera.camera_id = database->WriteCamera(camera);
  EXPECT_EQ(database->NumCameras(), 1);
  EXPECT_TRUE(database->ExistsCamera(camera.camera_id));
  EXPECT_EQ(database->ReadCamera(camera.camera_id).camera_id, camera.camera_id);
  EXPECT_EQ(database->ReadCamera(camera.camera_id).model_id, camera.model_id);
  EXPECT_EQ(database->ReadCamera(camera.camera_id), camera);
  camera.SetFocalLength(2 * camera.FocalLength());
  database->UpdateCamera(camera);
  EXPECT_EQ(database->ReadCamera(camera.camera_id), camera);
  Camera camera2 = camera;
  camera2.camera_id = camera.camera_id + 1;
  database->WriteCamera(camera2, true);
  EXPECT_EQ(database->NumCameras(), 2);
  EXPECT_TRUE(database->ExistsCamera(camera.camera_id));
  EXPECT_TRUE(database->ExistsCamera(camera2.camera_id));
  EXPECT_EQ(database->ReadAllCameras().size(), 2);
  EXPECT_THAT(database->ReadAllCameras(),
              testing::ElementsAre(camera, camera2));
  database->ClearCameras();
  EXPECT_EQ(database->NumCameras(), 0);
}

TEST_P(ParameterizedDatabaseTests, Frame) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  Rig rig;
  rig.AddRefSensor(sensor_t(SensorType::CAMERA, 1));
  rig.SetRigId(database->WriteRig(rig));
  EXPECT_EQ(database->NumFrames(), 0);

  Frame frame;
  frame.SetRigId(rig.RigId());
  frame.SetFrameId(database->WriteFrame(frame));
  EXPECT_EQ(database->NumFrames(), 1);
  EXPECT_TRUE(database->ExistsFrame(frame.FrameId()));
  EXPECT_EQ(database->ReadFrame(frame.FrameId()), frame);

  database->ClearFrames();
  EXPECT_EQ(database->NumFrames(), 0);

  frame.AddDataId(data_t(sensor_t(SensorType::IMU, 1), 2));
  frame.AddDataId(data_t(sensor_t(SensorType::CAMERA, 1), 3));
  frame.SetFrameId(database->WriteFrame(frame));
  EXPECT_EQ(database->NumFrames(), 1);
  EXPECT_TRUE(database->ExistsFrame(frame.FrameId()));
  EXPECT_EQ(database->ReadFrame(frame.FrameId()), frame);

  frame.AddDataId(data_t(sensor_t(SensorType::CAMERA, 2), 4));
  database->UpdateFrame(frame);
  EXPECT_EQ(database->ReadFrame(frame.FrameId()), frame);
  Frame frame2;
  frame2.SetRigId(rig.RigId());
  frame2.AddDataId(data_t(sensor_t(SensorType::CAMERA, 2), 5));
  frame2.SetFrameId(frame.FrameId() + 1);
  database->WriteFrame(frame2, /*use_frame_id=*/true);
  EXPECT_EQ(database->NumFrames(), 2);
  EXPECT_TRUE(database->ExistsFrame(frame.FrameId()));
  EXPECT_TRUE(database->ExistsFrame(frame2.FrameId()));
  EXPECT_EQ(database->ReadAllFrames().size(), 2);
  EXPECT_EQ(database->ReadAllFrames()[0].FrameId(), frame.FrameId());
  EXPECT_EQ(database->ReadAllFrames()[1].FrameId(), frame2.FrameId());

  database->ClearFrames();
  EXPECT_EQ(database->NumFrames(), 0);
}

TEST_P(ParameterizedDatabaseTests, Image) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  Camera camera = Camera::CreateFromModelName(
      kInvalidCameraId, "SIMPLE_PINHOLE", 1.0, 1, 1);
  camera.camera_id = database->WriteCamera(camera);
  Rig rig;
  rig.AddRefSensor(sensor_t(SensorType::CAMERA, camera.camera_id));
  rig.SetRigId(database->WriteRig(rig));
  EXPECT_EQ(database->NumImages(), 0);
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.camera_id);
  image.SetImageId(database->WriteImage(image));
  Frame frame;
  frame.SetRigId(rig.RigId());
  frame.AddDataId(image.DataId());
  frame.SetFrameId(database->WriteFrame(frame));
  image.SetFrameId(frame.FrameId());
  EXPECT_EQ(database->NumImages(), 1);
  EXPECT_TRUE(database->ExistsImage(image.ImageId()));
  EXPECT_EQ(database->ReadImage(image.ImageId()), image);
  EXPECT_EQ(database->ReadImageWithName(image.Name()), image);
  EXPECT_EQ(database->ReadImageWithName("foobar"), std::nullopt);
  image.SetName("test_changed");
  database->UpdateImage(image);
  EXPECT_EQ(database->ReadImage(image.ImageId()), image);
  Image image2 = image;
  image2.SetName("test2");
  image2.SetImageId(image.ImageId() + 1);
  frame.AddDataId(image2.DataId());
  database->UpdateFrame(frame);
  EXPECT_EQ(database->WriteImage(image2, /*use_image_id=*/true),
            image2.ImageId());
  EXPECT_EQ(database->NumImages(), 2);
  EXPECT_TRUE(database->ExistsImage(image.ImageId()));
  EXPECT_TRUE(database->ExistsImage(image2.ImageId()));
  EXPECT_THAT(database->ReadAllImages(), testing::ElementsAre(image, image2));
  database->ClearImages();
  EXPECT_EQ(database->NumImages(), 0);
}

TEST_P(ParameterizedDatabaseTests, PosePrior) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  Camera camera;
  camera.camera_id = database->WriteCamera(camera);
  Image image;
  image.SetCameraId(camera.camera_id);
  EXPECT_EQ(database->NumPosePriors(), 0);
  PosePrior pose_prior;
  pose_prior.corr_data_id = image.DataId();
  pose_prior.position = Eigen::Vector3d(0.1, 0.2, 0.3);
  pose_prior.position_covariance = Eigen::Matrix3d::Random();
  pose_prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
  pose_prior.gravity = Eigen::Vector3d::Random();
  pose_prior.pose_prior_id = database->WritePosePrior(pose_prior);
  EXPECT_ANY_THROW(database->WritePosePrior(pose_prior));
  EXPECT_EQ(database->NumPosePriors(), 1);
  EXPECT_EQ(database->ReadPosePrior(pose_prior.pose_prior_id,
                                    /*is_deprecated_image_prior=*/false),
            pose_prior);
  pose_prior.position_covariance = Eigen::Matrix3d::Identity();
  database->UpdatePosePrior(pose_prior);
  EXPECT_EQ(database->ReadPosePrior(pose_prior.pose_prior_id,
                                    /*is_deprecated_image_prior=*/false),
            pose_prior);
  EXPECT_THAT(database->ReadAllPosePriors(), testing::ElementsAre(pose_prior));
  database->ClearPosePriors();
  EXPECT_EQ(database->NumPosePriors(), 0);
}

TEST_P(ParameterizedDatabaseTests, Keypoints) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  Camera camera;
  camera.camera_id = database->WriteCamera(camera);
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.camera_id);
  image.SetImageId(database->WriteImage(image));
  EXPECT_EQ(database->NumKeypoints(), 0);
  EXPECT_EQ(database->NumKeypointsForImage(image.ImageId()), 0);
  const FeatureKeypoints keypoints = FeatureKeypoints(10);
  database->WriteKeypoints(image.ImageId(), keypoints);
  EXPECT_EQ(keypoints, database->ReadKeypoints(image.ImageId()));
  EXPECT_EQ(database->NumKeypoints(), 10);
  EXPECT_EQ(database->MaxNumKeypoints(), 10);
  EXPECT_EQ(database->NumKeypointsForImage(image.ImageId()), 10);
  FeatureKeypoints keypoints2 = FeatureKeypoints(20);
  image.SetName("test2");
  image.SetImageId(database->WriteImage(image));
  database->WriteKeypoints(image.ImageId(), keypoints2);
  EXPECT_EQ(keypoints2, database->ReadKeypoints(image.ImageId()));
  EXPECT_EQ(database->NumKeypoints(), 30);
  EXPECT_EQ(database->MaxNumKeypoints(), 20);
  EXPECT_EQ(database->NumKeypointsForImage(image.ImageId()), 20);
  keypoints2[0].x += 1;
  database->UpdateKeypoints(image.ImageId(), keypoints2);
  EXPECT_EQ(keypoints2, database->ReadKeypoints(image.ImageId()));
  database->ClearKeypoints();
  EXPECT_EQ(database->NumKeypoints(), 0);
  EXPECT_EQ(database->MaxNumKeypoints(), 0);
  EXPECT_EQ(database->NumKeypointsForImage(image.ImageId()), 0);
}

TEST_P(ParameterizedDatabaseTests, ReadKeypointsEmpty) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  Camera camera;
  camera.camera_id = database->WriteCamera(camera);
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.camera_id);
  image.SetImageId(database->WriteImage(image));
  // Reading keypoints for an image with no keypoints should return empty.
  const FeatureKeypoints keypoints = database->ReadKeypoints(image.ImageId());
  EXPECT_TRUE(keypoints.empty());
}

TEST_P(ParameterizedDatabaseTests, Descriptors) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  Camera camera;
  camera.camera_id = database->WriteCamera(camera);
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.camera_id);
  image.SetImageId(database->WriteImage(image));
  EXPECT_EQ(database->NumDescriptors(), 0);
  EXPECT_EQ(database->NumDescriptorsForImage(image.ImageId()), 0);
  const FeatureDescriptors descriptors(FeatureExtractorType::SIFT,
                                       FeatureDescriptorsData::Random(10, 128));
  database->WriteDescriptors(image.ImageId(), descriptors);
  const FeatureDescriptors descriptors_read =
      database->ReadDescriptors(image.ImageId());
  EXPECT_EQ(descriptors.data.rows(), descriptors_read.data.rows());
  EXPECT_EQ(descriptors.data.cols(), descriptors_read.data.cols());
  EXPECT_EQ(descriptors.type, descriptors_read.type);
  for (Eigen::Index r = 0; r < descriptors.data.rows(); ++r) {
    for (Eigen::Index c = 0; c < descriptors.data.cols(); ++c) {
      EXPECT_EQ(descriptors.data(r, c), descriptors_read.data(r, c));
    }
  }
  EXPECT_EQ(database->NumDescriptors(), 10);
  EXPECT_EQ(database->MaxNumDescriptors(), 10);
  EXPECT_EQ(database->NumDescriptorsForImage(image.ImageId()), 10);
  const FeatureDescriptors descriptors2(FeatureExtractorType::UNDEFINED,
                                        FeatureDescriptorsData(20, 128));
  image.SetName("test2");
  image.SetImageId(database->WriteImage(image));
  database->WriteDescriptors(image.ImageId(), descriptors2);
  const FeatureDescriptors descriptors2_read =
      database->ReadDescriptors(image.ImageId());
  EXPECT_EQ(descriptors2.type, descriptors2_read.type);
  EXPECT_EQ(database->NumDescriptors(), 30);
  EXPECT_EQ(database->MaxNumDescriptors(), 20);
  EXPECT_EQ(database->NumDescriptorsForImage(image.ImageId()), 20);
  database->ClearDescriptors();
  EXPECT_EQ(database->NumDescriptors(), 0);
  EXPECT_EQ(database->MaxNumDescriptors(), 0);
  EXPECT_EQ(database->NumDescriptorsForImage(image.ImageId()), 0);
}

TEST_P(ParameterizedDatabaseTests, DescriptorFeatureTypeDefault) {
  // Test that descriptors written with UNDEFINED type are read back correctly,
  // and that the database default (for migration) is SIFT (0).
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  Camera camera;
  camera.camera_id = database->WriteCamera(camera);
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.camera_id);
  image.SetImageId(database->WriteImage(image));

  // Write descriptors with UNDEFINED type (default).
  FeatureDescriptors descriptors;
  descriptors.data = FeatureDescriptorsData(5, 128);
  EXPECT_EQ(descriptors.type, FeatureExtractorType::UNDEFINED);
  database->WriteDescriptors(image.ImageId(), descriptors);

  // Read back and verify the type is preserved.
  const FeatureDescriptors descriptors_read =
      database->ReadDescriptors(image.ImageId());
  EXPECT_EQ(descriptors_read.type, FeatureExtractorType::UNDEFINED);

  // Write another image with SIFT type.
  image.SetName("test2");
  image.SetImageId(database->WriteImage(image));
  const FeatureDescriptors descriptors_sift(FeatureExtractorType::SIFT,
                                            FeatureDescriptorsData(5, 128));
  database->WriteDescriptors(image.ImageId(), descriptors_sift);

  const FeatureDescriptors descriptors_sift_read =
      database->ReadDescriptors(image.ImageId());
  EXPECT_EQ(descriptors_sift_read.type, FeatureExtractorType::SIFT);
}

TEST_P(ParameterizedDatabaseTests, Matches) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  const image_t image_id1 = 1;
  const image_t image_id2 = 2;
  constexpr int kNumMatches = 1000;
  FeatureMatches matches12(kNumMatches);
  FeatureMatches matches21(kNumMatches);
  for (size_t i = 0; i < matches12.size(); ++i) {
    matches12[i].point2D_idx1 = i;
    matches12[i].point2D_idx2 = 10000 + i;
    matches21[i].point2D_idx1 = 10000 + i;
    matches21[i].point2D_idx2 = i;
  }

  auto expectValidMatches = [&]() {
    EXPECT_EQ(database->NumMatchedImagePairs(), 1);
    const FeatureMatches matches_read12 =
        database->ReadMatches(image_id1, image_id2);
    EXPECT_EQ(matches12.size(), matches_read12.size());
    for (size_t i = 0; i < matches12.size(); ++i) {
      EXPECT_EQ(matches12[i].point2D_idx1, matches_read12[i].point2D_idx1);
      EXPECT_EQ(matches12[i].point2D_idx2, matches_read12[i].point2D_idx2);
    }
    const FeatureMatches matches_read21 =
        database->ReadMatches(image_id2, image_id1);
    EXPECT_EQ(matches12.size(), matches_read21.size());
    for (size_t i = 0; i < matches12.size(); ++i) {
      EXPECT_EQ(matches12[i].point2D_idx1, matches_read21[i].point2D_idx2);
      EXPECT_EQ(matches12[i].point2D_idx2, matches_read21[i].point2D_idx1);
    }
  };

  EXPECT_EQ(database->NumMatchedImagePairs(), 0);
  database->WriteMatches(image_id1, image_id2, matches12);
  expectValidMatches();
  database->DeleteMatches(image_id1, image_id2);
  EXPECT_EQ(database->NumMatchedImagePairs(), 0);
  database->WriteMatches(image_id2, image_id1, matches21);
  expectValidMatches();

  EXPECT_EQ(database->ReadAllMatchesBlob().size(), 1);
  EXPECT_EQ(database->ReadAllMatchesBlob()[0].first,
            ImagePairToPairId(image_id1, image_id2));
  const std::vector<std::pair<image_pair_t, FeatureMatches>> matches =
      database->ReadAllMatches();
  EXPECT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0].first, ImagePairToPairId(image_id1, image_id2));
  const std::vector<std::pair<image_pair_t, int>> pair_ids_and_num_matches =
      database->ReadNumMatches();
  EXPECT_EQ(pair_ids_and_num_matches.size(), 1);
  EXPECT_EQ(pair_ids_and_num_matches[0].first,
            ImagePairToPairId(image_id1, image_id2));
  EXPECT_EQ(pair_ids_and_num_matches[0].second, matches[0].second.size());
  EXPECT_EQ(database->NumMatches(), kNumMatches);
  database->DeleteMatches(image_id1, image_id2);
  EXPECT_EQ(database->NumMatches(), 0);
  database->WriteMatches(image_id1, image_id2, matches12);
  EXPECT_EQ(database->NumMatches(), kNumMatches);
  database->ClearMatches();
  EXPECT_EQ(database->NumMatches(), 0);
}

TEST_P(ParameterizedDatabaseTests, TwoViewGeometry) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  const image_t image_id1 = 1;
  const image_t image_id2 = 2;
  TwoViewGeometry two_view_geometry;
  two_view_geometry.inlier_matches = FeatureMatches(1000);
  two_view_geometry.config =
      TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC;
  two_view_geometry.F = Eigen::Matrix3d::Random();
  two_view_geometry.E = Eigen::Matrix3d::Random();
  two_view_geometry.H = Eigen::Matrix3d::Random();
  two_view_geometry.cam2_from_cam1 =
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
  database->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
  const TwoViewGeometry two_view_geometry_read =
      database->ReadTwoViewGeometry(image_id1, image_id2);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(),
            two_view_geometry_read.inlier_matches.size());
  for (size_t i = 0; i < two_view_geometry_read.inlier_matches.size(); ++i) {
    EXPECT_EQ(two_view_geometry.inlier_matches[i].point2D_idx1,
              two_view_geometry_read.inlier_matches[i].point2D_idx1);
    EXPECT_EQ(two_view_geometry.inlier_matches[i].point2D_idx2,
              two_view_geometry_read.inlier_matches[i].point2D_idx2);
  }

  EXPECT_EQ(two_view_geometry.config, two_view_geometry_read.config);
  EXPECT_EQ(two_view_geometry.F, two_view_geometry_read.F);
  EXPECT_EQ(two_view_geometry.E, two_view_geometry_read.E);
  EXPECT_EQ(two_view_geometry.H, two_view_geometry_read.H);
  EXPECT_TRUE(two_view_geometry.cam2_from_cam1.has_value());
  EXPECT_TRUE(two_view_geometry_read.cam2_from_cam1.has_value());
  EXPECT_EQ(two_view_geometry.cam2_from_cam1->rotation().coeffs(),
            two_view_geometry_read.cam2_from_cam1->rotation().coeffs());
  EXPECT_EQ(two_view_geometry.cam2_from_cam1->translation(),
            two_view_geometry_read.cam2_from_cam1->translation());

  const TwoViewGeometry two_view_geometry_read_inv =
      database->ReadTwoViewGeometry(image_id2, image_id1);
  EXPECT_EQ(two_view_geometry_read_inv.inlier_matches.size(),
            two_view_geometry_read.inlier_matches.size());
  for (size_t i = 0; i < two_view_geometry_read.inlier_matches.size(); ++i) {
    EXPECT_EQ(two_view_geometry_read_inv.inlier_matches[i].point2D_idx2,
              two_view_geometry_read.inlier_matches[i].point2D_idx1);
    EXPECT_EQ(two_view_geometry_read_inv.inlier_matches[i].point2D_idx1,
              two_view_geometry_read.inlier_matches[i].point2D_idx2);
  }

  EXPECT_EQ(two_view_geometry_read_inv.config, two_view_geometry_read.config);
  EXPECT_EQ(two_view_geometry_read_inv.F.value().transpose(),
            two_view_geometry_read.F.value());
  EXPECT_EQ(two_view_geometry_read_inv.E.value().transpose(),
            two_view_geometry_read.E.value());
  EXPECT_TRUE(two_view_geometry_read_inv.H.value().inverse().eval().isApprox(
      two_view_geometry_read.H.value()));
  EXPECT_TRUE(two_view_geometry_read_inv.cam2_from_cam1.has_value());
  EXPECT_TRUE(two_view_geometry_read_inv.cam2_from_cam1->rotation().isApprox(
      Inverse(*two_view_geometry_read.cam2_from_cam1).rotation()));
  EXPECT_TRUE(two_view_geometry_read_inv.cam2_from_cam1->translation().isApprox(
      Inverse(*two_view_geometry_read.cam2_from_cam1).translation()));

  const std::vector<std::pair<image_pair_t, TwoViewGeometry>>
      two_view_geometries = database->ReadTwoViewGeometries();
  EXPECT_EQ(two_view_geometries.size(), 1);
  EXPECT_EQ(two_view_geometries[0].first,
            ImagePairToPairId(image_id1, image_id2));
  EXPECT_EQ(two_view_geometry.config, two_view_geometries[0].second.config);
  EXPECT_EQ(two_view_geometry.F, two_view_geometries[0].second.F);
  EXPECT_EQ(two_view_geometry.E, two_view_geometries[0].second.E);
  EXPECT_EQ(two_view_geometry.H, two_view_geometries[0].second.H);
  EXPECT_TRUE(two_view_geometries[0].second.cam2_from_cam1.has_value());
  EXPECT_EQ(two_view_geometry.cam2_from_cam1->rotation().coeffs(),
            two_view_geometries[0].second.cam2_from_cam1->rotation().coeffs());
  EXPECT_EQ(two_view_geometry.cam2_from_cam1->translation(),
            two_view_geometries[0].second.cam2_from_cam1->translation());
  EXPECT_EQ(two_view_geometry.inlier_matches.size(),
            two_view_geometries[0].second.inlier_matches.size());
  const std::vector<std::pair<image_pair_t, int>> pair_ids_and_num_inliers =
      database->ReadTwoViewGeometryNumInliers();
  EXPECT_EQ(pair_ids_and_num_inliers.size(), 1);
  EXPECT_EQ(pair_ids_and_num_inliers[0].first,
            ImagePairToPairId(image_id1, image_id2));
  EXPECT_EQ(pair_ids_and_num_inliers[0].second,
            two_view_geometry.inlier_matches.size());
  EXPECT_EQ(database->NumInlierMatches(), 1000);
  database->DeleteInlierMatches(image_id1, image_id2);
  EXPECT_TRUE(database->ExistsTwoViewGeometry(image_id1, image_id2));
  EXPECT_EQ(database->NumInlierMatches(), 0);
  database->DeleteTwoViewGeometry(image_id1, image_id2);
  EXPECT_FALSE(database->ExistsTwoViewGeometry(image_id1, image_id2));
  EXPECT_EQ(database->NumInlierMatches(), 0);
  database->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
  EXPECT_ANY_THROW(
      database->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry));
  EXPECT_EQ(database->NumInlierMatches(), 1000);
  database->ClearTwoViewGeometries();
  EXPECT_EQ(database->NumInlierMatches(), 0);
  two_view_geometry.inlier_matches.clear();
  database->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
  EXPECT_EQ(two_view_geometry.cam2_from_cam1,
            database->ReadTwoViewGeometry(image_id1, image_id2).cam2_from_cam1);

  // Test with E and F set, but H missing.
  database->ClearTwoViewGeometries();
  TwoViewGeometry two_view_geometry_no_h;
  two_view_geometry_no_h.inlier_matches = FeatureMatches(10);
  two_view_geometry_no_h.config =
      TwoViewGeometry::ConfigurationType::CALIBRATED;
  two_view_geometry_no_h.E = Eigen::Matrix3d::Random();
  two_view_geometry_no_h.F = Eigen::Matrix3d::Random();
  database->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry_no_h);
  const TwoViewGeometry two_view_geometry_no_h_read =
      database->ReadTwoViewGeometry(image_id1, image_id2);
  EXPECT_TRUE(two_view_geometry_no_h_read.E.has_value());
  EXPECT_TRUE(two_view_geometry_no_h_read.F.has_value());
  EXPECT_EQ(two_view_geometry_no_h.E, two_view_geometry_no_h_read.E);
  EXPECT_EQ(two_view_geometry_no_h.F, two_view_geometry_no_h_read.F);
  EXPECT_FALSE(two_view_geometry_no_h_read.H.has_value());
}

TEST_P(ParameterizedDatabaseTests, Merge) {
  std::shared_ptr<Database> database1 = GetParam()(kInMemorySqliteDatabasePath);
  std::shared_ptr<Database> database2 = GetParam()(kInMemorySqliteDatabasePath);

  // This test intentionally uses custom, large, partially overlapping IDs from
  // rigs/frames/images/cameras which then require remapping of the IDs. This is
  // to ensure that the database can handle this case.

  Camera camera1 = Camera::CreateFromModelName(
      kInvalidCameraId, "SIMPLE_PINHOLE", 1.0, 1, 1);
  camera1.camera_id = 50;
  database1->WriteCamera(camera1, /*use_camera_id=*/true);
  Camera camera2 = Camera::CreateFromModelName(
      kInvalidCameraId, "SIMPLE_PINHOLE", 1.0, 1, 1);
  camera2.camera_id = 60;
  database1->WriteCamera(camera2, /*use_camera_id=*/true);
  Camera camera3 = Camera::CreateFromModelName(
      kInvalidCameraId, "SIMPLE_PINHOLE", 1.0, 1, 1);
  camera3.camera_id = 55;
  database2->WriteCamera(camera3, /*use_camera_id=*/true);
  Camera camera4 = Camera::CreateFromModelName(
      kInvalidCameraId, "SIMPLE_PINHOLE", 1.0, 1, 1);
  camera4.camera_id = 60;
  database2->WriteCamera(camera4, /*use_camera_id=*/true);

  Rig rig1;
  rig1.SetRigId(100);
  rig1.AddRefSensor(camera1.SensorId());
  rig1.AddSensor(camera2.SensorId(), Rigid3d());
  database1->WriteRig(rig1, /*use_rig_id=*/true);

  Rig rig2;
  rig2.SetRigId(200);
  rig2.AddRefSensor(camera3.SensorId());
  rig2.AddSensor(camera4.SensorId(), Rigid3d());
  database2->WriteRig(rig2, /*use_rig_id=*/true);

  const image_t image_id1 = 300;
  const image_t image_id2 = 400;
  const image_t image_id3 = 350;
  const image_t image_id4 = 400;

  Image image;
  image.SetImageId(image_id1);
  image.SetCameraId(camera1.camera_id);
  image.SetName("test1");
  database1->WriteImage(image, /*use_image_id=*/true);
  image.SetImageId(image_id2);
  image.SetCameraId(camera2.camera_id);
  image.SetName("test2");
  database1->WriteImage(image, /*use_image_id=*/true);

  image.SetImageId(image_id3);
  image.SetCameraId(camera3.camera_id);
  image.SetName("test3");
  database2->WriteImage(image, /*use_image_id=*/true);
  image.SetImageId(image_id4);
  image.SetCameraId(camera4.camera_id);
  image.SetName("test4");
  database2->WriteImage(image, /*use_image_id=*/true);

  Frame frame1;
  frame1.SetRigId(rig1.RigId());
  frame1.AddDataId(data_t(camera1.SensorId(), image_id1));
  frame1.AddDataId(data_t(camera2.SensorId(), image_id2));
  frame1.SetFrameId(1000);
  database1->WriteFrame(frame1, /*use_frame_id=*/true);
  Frame frame2;
  frame2.SetRigId(rig2.RigId());
  frame2.AddDataId(data_t(camera3.SensorId(), image_id3));
  frame2.AddDataId(data_t(camera4.SensorId(), image_id4));
  frame2.SetFrameId(2000);
  database2->WriteFrame(frame2, /*use_frame_id=*/true);

  PosePrior pose_prior1;
  pose_prior1.corr_data_id = data_t(camera1.SensorId(), image_id1);
  pose_prior1.position = Eigen::Vector3d::Random();
  pose_prior1.pose_prior_id = database1->WritePosePrior(pose_prior1);

  PosePrior pose_prior2;
  pose_prior2.corr_data_id = data_t(camera3.SensorId(), image_id3);
  pose_prior2.position = Eigen::Vector3d::Random();
  pose_prior2.pose_prior_id = database2->WritePosePrior(pose_prior2);

  auto keypoints1 = FeatureKeypoints(10);
  keypoints1[0].x = 100;
  auto keypoints2 = FeatureKeypoints(20);
  keypoints2[0].x = 200;
  auto keypoints3 = FeatureKeypoints(30);
  keypoints3[0].x = 300;
  auto keypoints4 = FeatureKeypoints(40);
  keypoints4[0].x = 400;

  const FeatureDescriptors descriptors1(
      FeatureExtractorType::UNDEFINED, FeatureDescriptorsData::Random(10, 128));
  const FeatureDescriptors descriptors2(
      FeatureExtractorType::UNDEFINED, FeatureDescriptorsData::Random(20, 128));
  const FeatureDescriptors descriptors3(
      FeatureExtractorType::UNDEFINED, FeatureDescriptorsData::Random(30, 128));
  const FeatureDescriptors descriptors4(
      FeatureExtractorType::UNDEFINED, FeatureDescriptorsData::Random(40, 128));

  database1->WriteKeypoints(image_id1, keypoints1);
  database1->WriteKeypoints(image_id2, keypoints2);
  database2->WriteKeypoints(image_id3, keypoints3);
  database2->WriteKeypoints(image_id4, keypoints4);
  database1->WriteDescriptors(image_id1, descriptors1);
  database1->WriteDescriptors(image_id2, descriptors2);
  database2->WriteDescriptors(image_id3, descriptors3);
  database2->WriteDescriptors(image_id4, descriptors4);
  database1->WriteMatches(image_id1, image_id2, FeatureMatches(10));
  database2->WriteMatches(image_id3, image_id4, FeatureMatches(10));
  database1->WriteTwoViewGeometry(image_id1, image_id2, TwoViewGeometry());
  database2->WriteTwoViewGeometry(image_id3, image_id4, TwoViewGeometry());

  std::shared_ptr<Database> merged_database =
      GetParam()(kInMemorySqliteDatabasePath);
  Database::Merge(*database1, *database2, merged_database.get());
  EXPECT_EQ(merged_database->NumRigs(), 2);
  EXPECT_EQ(merged_database->NumCameras(), 4);
  EXPECT_EQ(merged_database->NumFrames(), 2);
  EXPECT_EQ(merged_database->NumImages(), 4);
  EXPECT_EQ(merged_database->NumPosePriors(), 2);
  EXPECT_EQ(merged_database->NumKeypoints(), 100);
  EXPECT_EQ(merged_database->NumDescriptors(), 100);
  EXPECT_EQ(merged_database->NumMatches(), 20);
  EXPECT_EQ(merged_database->NumInlierMatches(), 0);
  EXPECT_EQ(merged_database->ReadAllFrames()[0].NumDataIds(),
            frame1.NumDataIds());
  EXPECT_EQ(merged_database->ReadAllFrames()[1].NumDataIds(),
            frame2.NumDataIds());
  for (const auto& frame : merged_database->ReadAllFrames()) {
    for (const auto& data_id : frame.DataIds()) {
      switch (data_id.sensor_id.type) {
        case SensorType::CAMERA:
          EXPECT_TRUE(merged_database->ExistsCamera(data_id.sensor_id.id));
          EXPECT_TRUE(merged_database->ExistsImage(data_id.id));
          break;
        default:
          GTEST_FAIL() << "Unexpected sensor type: " << data_id.sensor_id.type;
          break;
      }
    }
  }
  for (const auto& pose_prior : merged_database->ReadAllPosePriors()) {
    switch (pose_prior.corr_data_id.sensor_id.type) {
      case SensorType::CAMERA:
        EXPECT_TRUE(merged_database->ExistsCamera(
            pose_prior.corr_data_id.sensor_id.id));
        EXPECT_TRUE(merged_database->ExistsImage(pose_prior.corr_data_id.id));
        break;
      default:
        GTEST_FAIL() << "Unexpected sensor type: "
                     << pose_prior.corr_data_id.sensor_id.type;
        break;
    }
  }

  EXPECT_EQ(merged_database->ReadAllImages()[0].CameraId(), 1);
  EXPECT_EQ(merged_database->ReadAllImages()[1].CameraId(), 2);
  EXPECT_EQ(merged_database->ReadAllImages()[2].CameraId(), 3);
  EXPECT_EQ(merged_database->ReadAllImages()[3].CameraId(), 4);
  EXPECT_EQ(merged_database->ReadKeypoints(1).size(), 10);
  EXPECT_EQ(merged_database->ReadKeypoints(2).size(), 20);
  EXPECT_EQ(merged_database->ReadKeypoints(3).size(), 30);
  EXPECT_EQ(merged_database->ReadKeypoints(4).size(), 40);
  EXPECT_EQ(merged_database->ReadKeypoints(1)[0].x, 100);
  EXPECT_EQ(merged_database->ReadKeypoints(2)[0].x, 200);
  EXPECT_EQ(merged_database->ReadKeypoints(3)[0].x, 300);
  EXPECT_EQ(merged_database->ReadKeypoints(4)[0].x, 400);
  EXPECT_EQ(merged_database->ReadDescriptors(1).type, descriptors1.type);
  EXPECT_EQ(merged_database->ReadDescriptors(1).data.size(),
            descriptors1.data.size());
  EXPECT_EQ(merged_database->ReadDescriptors(2).type, descriptors2.type);
  EXPECT_EQ(merged_database->ReadDescriptors(2).data.size(),
            descriptors2.data.size());
  EXPECT_EQ(merged_database->ReadDescriptors(3).type, descriptors3.type);
  EXPECT_EQ(merged_database->ReadDescriptors(3).data.size(),
            descriptors3.data.size());
  EXPECT_EQ(merged_database->ReadDescriptors(4).type, descriptors4.type);
  EXPECT_EQ(merged_database->ReadDescriptors(4).data.size(),
            descriptors4.data.size());
  EXPECT_TRUE(merged_database->ExistsMatches(1, 2));
  EXPECT_FALSE(merged_database->ExistsMatches(2, 3));
  EXPECT_FALSE(merged_database->ExistsMatches(2, 4));
  EXPECT_TRUE(merged_database->ExistsMatches(3, 4));

  merged_database->ClearAllTables();
  EXPECT_EQ(merged_database->NumRigs(), 0);
  EXPECT_EQ(merged_database->NumCameras(), 0);
  EXPECT_EQ(merged_database->NumFrames(), 0);
  EXPECT_EQ(merged_database->NumImages(), 0);
  EXPECT_EQ(merged_database->NumPosePriors(), 0);
  EXPECT_EQ(merged_database->NumKeypoints(), 0);
  EXPECT_EQ(merged_database->NumDescriptors(), 0);
  EXPECT_EQ(merged_database->NumMatches(), 0);
}

INSTANTIATE_TEST_SUITE_P(
    DatabaseTests,
    ParameterizedDatabaseTests,
    ::testing::Values([](const std::filesystem::path& path) {
      return Database::Open(path);
    }));

// Helper to create a database file with images and descriptors.
std::shared_ptr<Database> CreateDatabaseWithRandomDescriptors(
    const std::vector<int>& num_descriptors_per_image) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  const int num_images = num_descriptors_per_image.size();

  Camera camera;
  camera.camera_id = database->WriteCamera(camera);

  for (int i = 0; i < num_images; ++i) {
    Image image;
    image.SetName("image" + std::to_string(i));
    image.SetCameraId(camera.camera_id);
    image.SetImageId(database->WriteImage(image));
    database->WriteDescriptors(
        image.ImageId(),
        FeatureDescriptors(
            FeatureExtractorType::SIFT,
            FeatureDescriptorsData::Random(num_descriptors_per_image[i], 128)));
  }

  return database;
}

TEST(LoadRandomDatabaseDescriptorsTest, LoadEmpty) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  const auto result = LoadRandomDatabaseDescriptors(*database, -1);
  EXPECT_EQ(result.data.rows(), 0);
  EXPECT_EQ(result.data.cols(), 0);
  EXPECT_EQ(result.type, FeatureExtractorType::UNDEFINED);
}

TEST(LoadRandomDatabaseDescriptorsTest, LoadAll) {
  const auto database = CreateDatabaseWithRandomDescriptors({10, 20, 30});
  const auto result = LoadRandomDatabaseDescriptors(*database, -1);
  EXPECT_EQ(result.data.rows(), 60);
  EXPECT_EQ(result.data.cols(), 128);
  EXPECT_EQ(result.type, FeatureExtractorType::SIFT);
}

TEST(LoadRandomDatabaseDescriptorsTest, LoadAllWithLargeMax) {
  const auto database = CreateDatabaseWithRandomDescriptors({15, 15});
  const auto result = LoadRandomDatabaseDescriptors(*database, 1000);
  EXPECT_EQ(result.data.rows(), 30);
  EXPECT_EQ(result.data.cols(), 128);
}

TEST(LoadRandomDatabaseDescriptorsTest, LoadSubset) {
  const auto database = CreateDatabaseWithRandomDescriptors({10, 20, 30});
  const auto result = LoadRandomDatabaseDescriptors(*database, 10);
  EXPECT_EQ(result.data.rows(), 10);
  EXPECT_EQ(result.data.cols(), 128);
  EXPECT_EQ(result.type, FeatureExtractorType::SIFT);
}

TEST(LoadRandomDatabaseDescriptorsTest, LoadSubsetWithSomeEmpty) {
  const auto database =
      CreateDatabaseWithRandomDescriptors({0, 10, 0, 15, 0, 20, 0});
  const auto result = LoadRandomDatabaseDescriptors(*database, 15);
  EXPECT_EQ(result.data.rows(), 15);
  EXPECT_EQ(result.data.cols(), 128);
  EXPECT_EQ(result.type, FeatureExtractorType::SIFT);
}

TEST(LoadRandomDatabaseDescriptorsTest, LoadExactTotal) {
  const auto database = CreateDatabaseWithRandomDescriptors({0, 10, 0, 10, 0});
  const auto result = LoadRandomDatabaseDescriptors(*database, 20);
  EXPECT_EQ(result.data.rows(), 20);
  EXPECT_EQ(result.data.cols(), 128);
}

}  // namespace
}  // namespace colmap

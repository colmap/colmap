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

#include "colmap/geometry/pose.h"
#include "colmap/util/eigen_alignment.h"

#include <thread>

#include <Eigen/Geometry>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(Database, OpenCloseConstructorDestructor) {
  Database database(Database::kInMemoryDatabasePath);
}

TEST(Database, OpenClose) {
  Database database(Database::kInMemoryDatabasePath);
  database.Close();
}

TEST(Database, Transaction) {
  Database database(Database::kInMemoryDatabasePath);
  DatabaseTransaction database_transaction(&database);
}

TEST(Database, TransactionMultiThreaded) {
  Database database(Database::kInMemoryDatabasePath);

  std::thread thread1([&database]() {
    DatabaseTransaction database_transaction(&database);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  });

  std::thread thread2([&database]() {
    DatabaseTransaction database_transaction(&database);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  });

  thread1.join();
  thread2.join();
}

TEST(Database, Empty) {
  Database database(Database::kInMemoryDatabasePath);
  EXPECT_EQ(database.NumCameras(), 0);
  EXPECT_EQ(database.NumFrames(), 0);
  EXPECT_EQ(database.NumImages(), 0);
  EXPECT_EQ(database.NumKeypoints(), 0);
  EXPECT_EQ(database.MaxNumKeypoints(), 0);
  EXPECT_EQ(database.NumDescriptors(), 0);
  EXPECT_EQ(database.MaxNumDescriptors(), 0);
  EXPECT_EQ(database.NumMatches(), 0);
  EXPECT_EQ(database.NumMatchedImagePairs(), 0);
  EXPECT_EQ(database.NumVerifiedImagePairs(), 0);
}

TEST(Database, ImagePairToPairId) {
  EXPECT_EQ(Database::ImagePairToPairId(0, 0), 0);
  EXPECT_EQ(Database::ImagePairToPairId(0, 1), 1);
  EXPECT_EQ(Database::ImagePairToPairId(0, 2), 2);
  EXPECT_EQ(Database::ImagePairToPairId(0, 3), 3);
  EXPECT_EQ(Database::ImagePairToPairId(1, 2), Database::kMaxNumImages + 2);
  for (image_t i = 0; i < 20; ++i) {
    for (image_t j = 0; j < 20; ++j) {
      const image_pair_t pair_id = Database::ImagePairToPairId(i, j);
      image_t image_id1;
      image_t image_id2;
      std::tie(image_id1, image_id2) = Database::PairIdToImagePair(pair_id);
      if (i < j) {
        EXPECT_EQ(i, image_id1);
        EXPECT_EQ(j, image_id2);
      } else {
        EXPECT_EQ(i, image_id2);
        EXPECT_EQ(j, image_id1);
      }
    }
  }
}

TEST(Database, SwapImagePair) {
  EXPECT_FALSE(Database::SwapImagePair(0, 0));
  EXPECT_FALSE(Database::SwapImagePair(0, 1));
  EXPECT_TRUE(Database::SwapImagePair(1, 0));
  EXPECT_FALSE(Database::SwapImagePair(1, 1));
}

TEST(Database, Rig) {
  Database database(Database::kInMemoryDatabasePath);
  EXPECT_EQ(database.NumRigs(), 0);
  Rig rig;
  rig.AddRefSensor(sensor_t(SensorType::CAMERA, 1));
  rig.AddSensor(
      sensor_t(SensorType::CAMERA, 2),
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random()));
  rig.AddSensor(sensor_t(SensorType::IMU, 3));
  rig.AddSensor(
      sensor_t(SensorType::IMU, 4),
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random()));
  rig.SetRigId(database.WriteRig(rig));
  EXPECT_EQ(database.NumRigs(), 1);
  EXPECT_TRUE(database.ExistsRig(rig.RigId()));
  EXPECT_EQ(database.ReadRig(rig.RigId()), rig);
  EXPECT_EQ(database.ReadRigWithSensor(sensor_t(SensorType::CAMERA, 1)), rig);
  EXPECT_EQ(database.ReadRigWithSensor(sensor_t(SensorType::IMU, 4)), rig);
  EXPECT_EQ(database.ReadRigWithSensor(sensor_t(SensorType::IMU, 42)),
            std::nullopt);
  rig.SensorFromRig(sensor_t(SensorType::CAMERA, 2)) =
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
  database.UpdateRig(rig);
  EXPECT_EQ(database.ReadRig(rig.RigId()), rig);
  Rig rig2;
  rig2.AddRefSensor(sensor_t(SensorType::IMU, 10));
  rig2.SetRigId(rig.RigId() + 1);
  database.WriteRig(rig2, /*use_rig_id=*/true);
  EXPECT_EQ(database.NumRigs(), 2);
  EXPECT_TRUE(database.ExistsRig(rig.RigId()));
  EXPECT_TRUE(database.ExistsRig(rig2.RigId()));
  EXPECT_EQ(database.ReadAllRigs().size(), 2);
  EXPECT_EQ(database.ReadAllRigs()[0].RigId(), rig.RigId());
  EXPECT_EQ(database.ReadAllRigs()[1].RigId(), rig2.RigId());
  database.ClearRigs();
  EXPECT_EQ(database.NumRigs(), 0);
}

TEST(Database, Camera) {
  Database database(Database::kInMemoryDatabasePath);
  EXPECT_EQ(database.NumCameras(), 0);
  Camera camera = Camera::CreateFromModelName(
      kInvalidCameraId, "SIMPLE_PINHOLE", 1.0, 1, 1);
  camera.camera_id = database.WriteCamera(camera);
  EXPECT_EQ(database.NumCameras(), 1);
  EXPECT_TRUE(database.ExistsCamera(camera.camera_id));
  EXPECT_EQ(database.ReadCamera(camera.camera_id).camera_id, camera.camera_id);
  EXPECT_EQ(database.ReadCamera(camera.camera_id).model_id, camera.model_id);
  EXPECT_EQ(database.ReadCamera(camera.camera_id), camera);
  camera.SetFocalLength(2 * camera.FocalLength());
  database.UpdateCamera(camera);
  EXPECT_EQ(database.ReadCamera(camera.camera_id), camera);
  Camera camera2 = camera;
  camera2.camera_id = camera.camera_id + 1;
  database.WriteCamera(camera2, true);
  EXPECT_EQ(database.NumCameras(), 2);
  EXPECT_TRUE(database.ExistsCamera(camera.camera_id));
  EXPECT_TRUE(database.ExistsCamera(camera2.camera_id));
  EXPECT_EQ(database.ReadAllCameras().size(), 2);
  EXPECT_THAT(database.ReadAllCameras(), testing::ElementsAre(camera, camera2));
  database.ClearCameras();
  EXPECT_EQ(database.NumCameras(), 0);
}

TEST(Database, Frame) {
  Database database(Database::kInMemoryDatabasePath);
  Rig rig;
  rig.AddRefSensor(sensor_t(SensorType::CAMERA, 1));
  rig.SetRigId(database.WriteRig(rig));
  EXPECT_EQ(database.NumFrames(), 0);
  Frame frame;
  frame.SetRigId(rig.RigId());
  frame.AddDataId(data_t(sensor_t(SensorType::IMU, 1), 2));
  frame.AddDataId(data_t(sensor_t(SensorType::CAMERA, 1), 3));
  frame.SetFrameId(database.WriteFrame(frame));
  EXPECT_EQ(database.NumFrames(), 1);
  EXPECT_TRUE(database.ExistsFrame(frame.FrameId()));
  EXPECT_EQ(database.ReadFrame(frame.FrameId()), frame);
  frame.AddDataId(data_t(sensor_t(SensorType::CAMERA, 2), 4));
  database.UpdateFrame(frame);
  EXPECT_EQ(database.ReadFrame(frame.FrameId()), frame);
  Frame frame2;
  frame2.SetRigId(rig.RigId());
  frame2.AddDataId(data_t(sensor_t(SensorType::CAMERA, 2), 5));
  frame2.SetFrameId(frame.FrameId() + 1);
  database.WriteFrame(frame2, /*use_frame_id=*/true);
  EXPECT_EQ(database.NumFrames(), 2);
  EXPECT_TRUE(database.ExistsFrame(frame.FrameId()));
  EXPECT_TRUE(database.ExistsFrame(frame2.FrameId()));
  EXPECT_EQ(database.ReadAllFrames().size(), 2);
  EXPECT_EQ(database.ReadAllFrames()[0].FrameId(), frame.FrameId());
  EXPECT_EQ(database.ReadAllFrames()[1].FrameId(), frame2.FrameId());
  database.ClearFrames();
  EXPECT_EQ(database.NumFrames(), 0);
}

TEST(Database, Image) {
  Database database(Database::kInMemoryDatabasePath);
  Camera camera = Camera::CreateFromModelName(
      kInvalidCameraId, "SIMPLE_PINHOLE", 1.0, 1, 1);
  camera.camera_id = database.WriteCamera(camera);
  EXPECT_EQ(database.NumImages(), 0);
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.camera_id);
  image.SetImageId(database.WriteImage(image));
  EXPECT_EQ(database.NumImages(), 1);
  EXPECT_TRUE(database.ExistsImage(image.ImageId()));
  EXPECT_EQ(database.ReadImage(image.ImageId()), image);
  EXPECT_EQ(database.ReadImageWithName(image.Name()), image);
  EXPECT_EQ(database.ReadImageWithName("foobar"), std::nullopt);
  image.SetName("test_changed");
  database.UpdateImage(image);
  EXPECT_EQ(database.ReadImage(image.ImageId()), image);
  Image image2 = image;
  image2.SetName("test2");
  image2.SetImageId(image.ImageId() + 1);
  database.WriteImage(image2, true);
  EXPECT_EQ(database.NumImages(), 2);
  EXPECT_TRUE(database.ExistsImage(image.ImageId()));
  EXPECT_TRUE(database.ExistsImage(image2.ImageId()));
  EXPECT_THAT(database.ReadAllImages(), testing::ElementsAre(image, image2));
  database.ClearImages();
  EXPECT_EQ(database.NumImages(), 0);
}

TEST(Database, PosePrior) {
  Database database(Database::kInMemoryDatabasePath);
  Camera camera;
  camera.camera_id = database.WriteCamera(camera);
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.camera_id);
  image.SetImageId(database.WriteImage(image));
  EXPECT_EQ(database.NumPosePriors(), 0);
  PosePrior pose_prior(Eigen::Vector3d(0.1, 0.2, 0.3),
                       PosePrior::CoordinateSystem::CARTESIAN);
  EXPECT_TRUE(pose_prior.IsValid());
  EXPECT_FALSE(pose_prior.IsCovarianceValid());
  database.WritePosePrior(image.ImageId(), pose_prior);
  EXPECT_EQ(database.NumPosePriors(), 1);
  auto read_pose_prior = database.ReadPosePrior(image.ImageId());
  EXPECT_EQ(read_pose_prior.position, pose_prior.position);
  EXPECT_EQ(read_pose_prior.coordinate_system, pose_prior.coordinate_system);
  EXPECT_TRUE(read_pose_prior.IsValid());
  EXPECT_FALSE(read_pose_prior.IsCovarianceValid());
  pose_prior.position_covariance = Eigen::Matrix3d::Identity();
  EXPECT_TRUE(pose_prior.IsCovarianceValid());
  database.UpdatePosePrior(image.ImageId(), pose_prior);
  read_pose_prior = database.ReadPosePrior(image.ImageId());
  EXPECT_EQ(read_pose_prior.position, pose_prior.position);
  EXPECT_EQ(read_pose_prior.position_covariance,
            pose_prior.position_covariance);
  EXPECT_EQ(read_pose_prior.coordinate_system, pose_prior.coordinate_system);
  EXPECT_TRUE(read_pose_prior.IsValid());
  EXPECT_TRUE(read_pose_prior.IsCovarianceValid());
  database.ClearPosePriors();
  EXPECT_EQ(database.NumPosePriors(), 0);
}

TEST(Database, Keypoints) {
  Database database(Database::kInMemoryDatabasePath);
  Camera camera;
  camera.camera_id = database.WriteCamera(camera);
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.camera_id);
  image.SetImageId(database.WriteImage(image));
  EXPECT_EQ(database.NumKeypoints(), 0);
  EXPECT_EQ(database.NumKeypointsForImage(image.ImageId()), 0);
  const FeatureKeypoints keypoints = FeatureKeypoints(10);
  database.WriteKeypoints(image.ImageId(), keypoints);
  const FeatureKeypoints keypoints_read =
      database.ReadKeypoints(image.ImageId());
  EXPECT_EQ(keypoints.size(), keypoints_read.size());
  for (size_t i = 0; i < keypoints.size(); ++i) {
    EXPECT_EQ(keypoints[i].x, keypoints_read[i].x);
    EXPECT_EQ(keypoints[i].y, keypoints_read[i].y);
    EXPECT_EQ(keypoints[i].a11, keypoints_read[i].a11);
    EXPECT_EQ(keypoints[i].a12, keypoints_read[i].a12);
    EXPECT_EQ(keypoints[i].a21, keypoints_read[i].a21);
    EXPECT_EQ(keypoints[i].a22, keypoints_read[i].a22);
  }
  EXPECT_EQ(database.NumKeypoints(), 10);
  EXPECT_EQ(database.MaxNumKeypoints(), 10);
  EXPECT_EQ(database.NumKeypointsForImage(image.ImageId()), 10);
  const FeatureKeypoints keypoints2 = FeatureKeypoints(20);
  image.SetName("test2");
  image.SetImageId(database.WriteImage(image));
  database.WriteKeypoints(image.ImageId(), keypoints2);
  EXPECT_EQ(database.NumKeypoints(), 30);
  EXPECT_EQ(database.MaxNumKeypoints(), 20);
  EXPECT_EQ(database.NumKeypointsForImage(image.ImageId()), 20);
  database.ClearKeypoints();
  EXPECT_EQ(database.NumKeypoints(), 0);
  EXPECT_EQ(database.MaxNumKeypoints(), 0);
  EXPECT_EQ(database.NumKeypointsForImage(image.ImageId()), 0);
}

TEST(Database, Descriptors) {
  Database database(Database::kInMemoryDatabasePath);
  Camera camera;
  camera.camera_id = database.WriteCamera(camera);
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.camera_id);
  image.SetImageId(database.WriteImage(image));
  EXPECT_EQ(database.NumDescriptors(), 0);
  EXPECT_EQ(database.NumDescriptorsForImage(image.ImageId()), 0);
  const FeatureDescriptors descriptors = FeatureDescriptors::Random(10, 128);
  database.WriteDescriptors(image.ImageId(), descriptors);
  const FeatureDescriptors descriptors_read =
      database.ReadDescriptors(image.ImageId());
  EXPECT_EQ(descriptors.rows(), descriptors_read.rows());
  EXPECT_EQ(descriptors.cols(), descriptors_read.cols());
  for (FeatureDescriptors::Index r = 0; r < descriptors.rows(); ++r) {
    for (FeatureDescriptors::Index c = 0; c < descriptors.cols(); ++c) {
      EXPECT_EQ(descriptors(r, c), descriptors_read(r, c));
    }
  }
  EXPECT_EQ(database.NumDescriptors(), 10);
  EXPECT_EQ(database.MaxNumDescriptors(), 10);
  EXPECT_EQ(database.NumDescriptorsForImage(image.ImageId()), 10);
  const FeatureDescriptors descriptors2 = FeatureDescriptors(20, 128);
  image.SetName("test2");
  image.SetImageId(database.WriteImage(image));
  database.WriteDescriptors(image.ImageId(), descriptors2);
  EXPECT_EQ(database.NumDescriptors(), 30);
  EXPECT_EQ(database.MaxNumDescriptors(), 20);
  EXPECT_EQ(database.NumDescriptorsForImage(image.ImageId()), 20);
  database.ClearDescriptors();
  EXPECT_EQ(database.NumDescriptors(), 0);
  EXPECT_EQ(database.MaxNumDescriptors(), 0);
  EXPECT_EQ(database.NumDescriptorsForImage(image.ImageId()), 0);
}

TEST(Database, Matches) {
  Database database(Database::kInMemoryDatabasePath);
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
    EXPECT_EQ(database.NumMatchedImagePairs(), 1);
    const FeatureMatches matches_read12 =
        database.ReadMatches(image_id1, image_id2);
    EXPECT_EQ(matches12.size(), matches_read12.size());
    for (size_t i = 0; i < matches12.size(); ++i) {
      EXPECT_EQ(matches12[i].point2D_idx1, matches_read12[i].point2D_idx1);
      EXPECT_EQ(matches12[i].point2D_idx2, matches_read12[i].point2D_idx2);
    }
    const FeatureMatches matches_read21 =
        database.ReadMatches(image_id2, image_id1);
    EXPECT_EQ(matches12.size(), matches_read21.size());
    for (size_t i = 0; i < matches12.size(); ++i) {
      EXPECT_EQ(matches12[i].point2D_idx1, matches_read21[i].point2D_idx2);
      EXPECT_EQ(matches12[i].point2D_idx2, matches_read21[i].point2D_idx1);
    }
  };

  EXPECT_EQ(database.NumMatchedImagePairs(), 0);
  database.WriteMatches(image_id1, image_id2, matches12);
  expectValidMatches();
  database.DeleteMatches(image_id1, image_id2);
  EXPECT_EQ(database.NumMatchedImagePairs(), 0);
  database.WriteMatches(image_id2, image_id1, matches21);
  expectValidMatches();

  EXPECT_EQ(database.ReadAllMatchesBlob().size(), 1);
  EXPECT_EQ(database.ReadAllMatchesBlob()[0].first,
            Database::ImagePairToPairId(image_id1, image_id2));
  EXPECT_EQ(database.ReadAllMatches().size(), 1);
  EXPECT_EQ(database.ReadAllMatches()[0].first,
            Database::ImagePairToPairId(image_id1, image_id2));
  EXPECT_EQ(database.NumMatches(), kNumMatches);
  database.DeleteMatches(image_id1, image_id2);
  EXPECT_EQ(database.NumMatches(), 0);
  database.WriteMatches(image_id1, image_id2, matches12);
  EXPECT_EQ(database.NumMatches(), kNumMatches);
  database.ClearMatches();
  EXPECT_EQ(database.NumMatches(), 0);
}

TEST(Database, TwoViewGeometry) {
  Database database(Database::kInMemoryDatabasePath);
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
  database.WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
  const TwoViewGeometry two_view_geometry_read =
      database.ReadTwoViewGeometry(image_id1, image_id2);
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
  EXPECT_EQ(two_view_geometry.cam2_from_cam1.rotation.coeffs(),
            two_view_geometry_read.cam2_from_cam1.rotation.coeffs());
  EXPECT_EQ(two_view_geometry.cam2_from_cam1.translation,
            two_view_geometry_read.cam2_from_cam1.translation);

  const TwoViewGeometry two_view_geometry_read_inv =
      database.ReadTwoViewGeometry(image_id2, image_id1);
  EXPECT_EQ(two_view_geometry_read_inv.inlier_matches.size(),
            two_view_geometry_read.inlier_matches.size());
  for (size_t i = 0; i < two_view_geometry_read.inlier_matches.size(); ++i) {
    EXPECT_EQ(two_view_geometry_read_inv.inlier_matches[i].point2D_idx2,
              two_view_geometry_read.inlier_matches[i].point2D_idx1);
    EXPECT_EQ(two_view_geometry_read_inv.inlier_matches[i].point2D_idx1,
              two_view_geometry_read.inlier_matches[i].point2D_idx2);
  }

  EXPECT_EQ(two_view_geometry_read_inv.config, two_view_geometry_read.config);
  EXPECT_EQ(two_view_geometry_read_inv.F.transpose(), two_view_geometry_read.F);
  EXPECT_EQ(two_view_geometry_read_inv.E.transpose(), two_view_geometry_read.E);
  EXPECT_TRUE(two_view_geometry_read_inv.H.inverse().eval().isApprox(
      two_view_geometry_read.H));
  EXPECT_TRUE(two_view_geometry_read_inv.cam2_from_cam1.rotation.isApprox(
      Inverse(two_view_geometry_read.cam2_from_cam1).rotation));
  EXPECT_TRUE(two_view_geometry_read_inv.cam2_from_cam1.translation.isApprox(
      Inverse(two_view_geometry_read.cam2_from_cam1).translation));

  const std::vector<std::pair<image_pair_t, TwoViewGeometry>>
      two_view_geometries = database.ReadTwoViewGeometries();
  EXPECT_EQ(two_view_geometries.size(), 1);
  EXPECT_EQ(two_view_geometries[0].first,
            Database::ImagePairToPairId(image_id1, image_id2));
  EXPECT_EQ(two_view_geometry.config, two_view_geometries[0].second.config);
  EXPECT_EQ(two_view_geometry.F, two_view_geometries[0].second.F);
  EXPECT_EQ(two_view_geometry.E, two_view_geometries[0].second.E);
  EXPECT_EQ(two_view_geometry.H, two_view_geometries[0].second.H);
  EXPECT_EQ(two_view_geometry.cam2_from_cam1.rotation.coeffs(),
            two_view_geometries[0].second.cam2_from_cam1.rotation.coeffs());
  EXPECT_EQ(two_view_geometry.cam2_from_cam1.translation,
            two_view_geometries[0].second.cam2_from_cam1.translation);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(),
            two_view_geometries[0].second.inlier_matches.size());
  const std::vector<std::pair<image_pair_t, int>> pair_ids_and_num_inliers =
      database.ReadTwoViewGeometryNumInliers();
  EXPECT_EQ(pair_ids_and_num_inliers.size(), 1);
  EXPECT_EQ(pair_ids_and_num_inliers[0].first,
            Database::ImagePairToPairId(image_id1, image_id2));
  EXPECT_EQ(pair_ids_and_num_inliers[0].second,
            two_view_geometry.inlier_matches.size());
  EXPECT_EQ(database.NumInlierMatches(), 1000);
  database.DeleteInlierMatches(image_id1, image_id2);
  EXPECT_EQ(database.NumInlierMatches(), 0);
  database.WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
  EXPECT_EQ(database.NumInlierMatches(), 1000);
  database.ClearTwoViewGeometries();
  EXPECT_EQ(database.NumInlierMatches(), 0);
}

TEST(Database, Merge) {
  Database database1(Database::kInMemoryDatabasePath);
  Database database2(Database::kInMemoryDatabasePath);

  Camera camera1 = Camera::CreateFromModelName(
      kInvalidCameraId, "SIMPLE_PINHOLE", 1.0, 1, 1);
  camera1.camera_id = database1.WriteCamera(camera1);
  Camera camera2 = Camera::CreateFromModelName(
      kInvalidCameraId, "SIMPLE_PINHOLE", 1.0, 1, 1);
  camera2.camera_id = database2.WriteCamera(camera2);

  Rig rig1;
  rig1.AddRefSensor(camera1.SensorId());
  rig1.SetRigId(database1.WriteRig(rig1));
  Rig rig2;
  rig2.AddRefSensor(camera2.SensorId());
  rig2.SetRigId(database2.WriteRig(rig2));

  Image image;
  image.SetCameraId(camera1.camera_id);

  image.SetName("test1");
  const image_t image_id1 = database1.WriteImage(image);
  image.SetName("test2");
  const image_t image_id2 = database1.WriteImage(image);

  image.SetCameraId(camera2.camera_id);
  image.SetName("test3");
  const image_t image_id3 = database2.WriteImage(image);
  image.SetName("test4");
  const image_t image_id4 = database2.WriteImage(image);

  Frame frame1;
  frame1.SetRigId(rig1.RigId());
  frame1.AddDataId(data_t(camera1.SensorId(), image_id1));
  frame1.AddDataId(data_t(camera2.SensorId(), image_id2));
  frame1.SetFrameId(database1.WriteFrame(frame1));
  Frame frame2;
  frame2.SetRigId(rig2.RigId());
  frame2.AddDataId(data_t(camera1.SensorId(), image_id3));
  frame2.AddDataId(data_t(camera2.SensorId(), image_id4));
  frame2.SetFrameId(database2.WriteFrame(frame2));

  database1.WritePosePrior(image_id1,
                           PosePrior(Eigen::Vector3d::Constant(0.1)));
  database2.WritePosePrior(image_id3,
                           PosePrior(Eigen::Vector3d::Constant(0.2)));

  auto keypoints1 = FeatureKeypoints(10);
  keypoints1[0].x = 100;
  auto keypoints2 = FeatureKeypoints(20);
  keypoints2[0].x = 200;
  auto keypoints3 = FeatureKeypoints(30);
  keypoints3[0].x = 300;
  auto keypoints4 = FeatureKeypoints(40);
  keypoints4[0].x = 400;

  const auto descriptors1 = FeatureDescriptors::Random(10, 128);
  const auto descriptors2 = FeatureDescriptors::Random(20, 128);
  const auto descriptors3 = FeatureDescriptors::Random(30, 128);
  const auto descriptors4 = FeatureDescriptors::Random(40, 128);

  database1.WriteKeypoints(image_id1, keypoints1);
  database1.WriteKeypoints(image_id2, keypoints2);
  database2.WriteKeypoints(image_id3, keypoints3);
  database2.WriteKeypoints(image_id4, keypoints4);
  database1.WriteDescriptors(image_id1, descriptors1);
  database1.WriteDescriptors(image_id2, descriptors2);
  database2.WriteDescriptors(image_id3, descriptors3);
  database2.WriteDescriptors(image_id4, descriptors4);
  database1.WriteMatches(image_id1, image_id2, FeatureMatches(10));
  database2.WriteMatches(image_id3, image_id4, FeatureMatches(10));
  database1.WriteTwoViewGeometry(image_id1, image_id2, TwoViewGeometry());
  database2.WriteTwoViewGeometry(image_id3, image_id4, TwoViewGeometry());

  Database merged_database(Database::kInMemoryDatabasePath);
  Database::Merge(database1, database2, &merged_database);
  EXPECT_EQ(merged_database.NumRigs(), 2);
  EXPECT_EQ(merged_database.NumCameras(), 2);
  EXPECT_EQ(merged_database.NumFrames(), 2);
  EXPECT_EQ(merged_database.NumImages(), 4);
  EXPECT_EQ(merged_database.NumPosePriors(), 2);
  EXPECT_EQ(merged_database.NumKeypoints(), 100);
  EXPECT_EQ(merged_database.NumDescriptors(), 100);
  EXPECT_EQ(merged_database.NumMatches(), 20);
  EXPECT_EQ(merged_database.NumInlierMatches(), 0);
  EXPECT_EQ(merged_database.ReadAllFrames()[0].DataIds().size(),
            frame1.DataIds().size());
  EXPECT_EQ(merged_database.ReadAllFrames()[1].DataIds().size(),
            frame2.DataIds().size());
  EXPECT_EQ(merged_database.ReadAllImages()[0].CameraId(), 1);
  EXPECT_EQ(merged_database.ReadAllImages()[1].CameraId(), 1);
  EXPECT_EQ(merged_database.ReadAllImages()[2].CameraId(), 2);
  EXPECT_EQ(merged_database.ReadAllImages()[3].CameraId(), 2);
  EXPECT_EQ(merged_database.ReadPosePrior(1).position.x(), 0.1);
  EXPECT_FALSE(merged_database.ExistsPosePrior(2));
  EXPECT_EQ(merged_database.ReadPosePrior(3).position.x(), 0.2);
  EXPECT_FALSE(merged_database.ExistsPosePrior(4));
  EXPECT_EQ(merged_database.ReadKeypoints(1).size(), 10);
  EXPECT_EQ(merged_database.ReadKeypoints(2).size(), 20);
  EXPECT_EQ(merged_database.ReadKeypoints(3).size(), 30);
  EXPECT_EQ(merged_database.ReadKeypoints(4).size(), 40);
  EXPECT_EQ(merged_database.ReadKeypoints(1)[0].x, 100);
  EXPECT_EQ(merged_database.ReadKeypoints(2)[0].x, 200);
  EXPECT_EQ(merged_database.ReadKeypoints(3)[0].x, 300);
  EXPECT_EQ(merged_database.ReadKeypoints(4)[0].x, 400);
  EXPECT_EQ(merged_database.ReadDescriptors(1).size(), descriptors1.size());
  EXPECT_EQ(merged_database.ReadDescriptors(2).size(), descriptors2.size());
  EXPECT_EQ(merged_database.ReadDescriptors(3).size(), descriptors3.size());
  EXPECT_EQ(merged_database.ReadDescriptors(4).size(), descriptors4.size());
  EXPECT_TRUE(merged_database.ExistsMatches(1, 2));
  EXPECT_FALSE(merged_database.ExistsMatches(2, 3));
  EXPECT_FALSE(merged_database.ExistsMatches(2, 4));
  EXPECT_TRUE(merged_database.ExistsMatches(3, 4));

  merged_database.ClearAllTables();
  EXPECT_EQ(merged_database.NumRigs(), 0);
  EXPECT_EQ(merged_database.NumCameras(), 0);
  EXPECT_EQ(merged_database.NumFrames(), 0);
  EXPECT_EQ(merged_database.NumImages(), 0);
  EXPECT_EQ(merged_database.NumPosePriors(), 0);
  EXPECT_EQ(merged_database.NumKeypoints(), 0);
  EXPECT_EQ(merged_database.NumDescriptors(), 0);
  EXPECT_EQ(merged_database.NumMatches(), 0);
}

}  // namespace
}  // namespace colmap

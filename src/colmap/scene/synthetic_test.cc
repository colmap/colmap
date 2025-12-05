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

#include "colmap/scene/synthetic.h"

#include "colmap/geometry/triangulation.h"
#include "colmap/math/random.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/projection.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/eigen_matchers.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(SynthesizeDataset, Nominal) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 2;
  options.num_cameras_per_rig = 3;
  options.num_frames_per_rig = 3;
  SynthesizeDataset(options, &reconstruction, database.get());

  const std::string test_dir = CreateTestDir();
  const std::string sparse_path = test_dir + "/sparse";
  CreateDirIfNotExists(sparse_path);
  reconstruction.Write(sparse_path);

  EXPECT_EQ(database->NumRigs(), options.num_rigs);
  EXPECT_EQ(reconstruction.NumRigs(), options.num_rigs);
  for (const auto& rig : reconstruction.Rigs()) {
    EXPECT_GE(rig.second.NumSensors(), options.num_cameras_per_rig);
  }

  EXPECT_EQ(database->NumCameras(),
            options.num_rigs * options.num_cameras_per_rig);
  EXPECT_EQ(reconstruction.NumCameras(),
            options.num_rigs * options.num_cameras_per_rig);
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    EXPECT_EQ(camera, database->ReadCamera(camera_id));
    EXPECT_EQ(camera.model_id, options.camera_model_id);
  }

  EXPECT_EQ(database->NumFrames(),
            options.num_rigs * options.num_frames_per_rig);
  EXPECT_EQ(reconstruction.NumFrames(),
            options.num_rigs * options.num_frames_per_rig);
  for (auto& [frame_id, frame] : reconstruction.Frames()) {
    Frame reconstruction_frame = frame;
    EXPECT_TRUE(reconstruction_frame.HasPose());
    reconstruction_frame.ResetPose();
    EXPECT_EQ(reconstruction_frame, database->ReadFrame(frame_id));
    EXPECT_EQ(reconstruction_frame.NumDataIds(),
              reconstruction_frame.RigPtr()->NumSensors());
  }

  EXPECT_EQ(database->NumImages(),
            options.num_rigs * options.num_cameras_per_rig *
                options.num_frames_per_rig);
  EXPECT_EQ(reconstruction.NumImages(),
            options.num_rigs * options.num_cameras_per_rig *
                options.num_frames_per_rig);
  EXPECT_EQ(reconstruction.NumRegFrames(),
            options.num_rigs * options.num_frames_per_rig);
  std::set<std::string> image_names;
  for (const auto& image : reconstruction.Images()) {
    EXPECT_EQ(image.second.Name(), database->ReadImage(image.first).Name());
    image_names.insert(image.second.Name());
    EXPECT_EQ(image.second.NumPoints2D(),
              database->ReadKeypoints(image.first).size());
    EXPECT_EQ(image.second.NumPoints2D(),
              database->ReadDescriptors(image.first).rows());
    EXPECT_EQ(database->ReadDescriptors(image.first).cols(), 128);
    EXPECT_EQ(image.second.NumPoints2D(),
              options.num_points3D + options.num_points2D_without_point3D);
    EXPECT_EQ(image.second.NumPoints3D(), options.num_points3D);
  }
  EXPECT_EQ(image_names.size(), reconstruction.NumImages());

  const int num_image_pairs =
      reconstruction.NumImages() * (reconstruction.NumImages() - 1) / 2;
  EXPECT_EQ(database->NumVerifiedImagePairs(), num_image_pairs);
  EXPECT_EQ(database->NumInlierMatches(),
            num_image_pairs * options.num_points3D);

  EXPECT_NEAR(reconstruction.ComputeMeanReprojectionError(), 0, 1e-6);
  EXPECT_NEAR(reconstruction.ComputeCentroid(0, 1).norm(), 0, 0.2);
  EXPECT_NEAR(
      reconstruction.ComputeMeanTrackLength(), reconstruction.NumImages(), 0.1);
  EXPECT_EQ(reconstruction.ComputeNumObservations(),
            reconstruction.NumImages() * options.num_points3D);

  EXPECT_EQ(reconstruction.NumPoints3D(), options.num_points3D);

  // All observations should be perfect and have sufficient triangulation
  // angle. No points or observations should be filtered.
  const double kMaxReprojError = 1e-3;
  const double kMinTriAngleDeg = 0.4;
  std::unordered_map<image_t, Eigen::Vector3d> proj_centers;
  for (const auto& point3D_id : reconstruction.Point3DIds()) {
    Point3D& point3D = reconstruction.Point3D(point3D_id);

    // Make sure all descriptors of the same 3D point have identical features.
    const FeatureDescriptor descriptors =
        database->ReadDescriptors(point3D.track.Element(0).image_id)
            .row(point3D.track.Element(0).point2D_idx);

    double max_tri_angle = 0;
    for (size_t i1 = 0; i1 < point3D.track.Length(); ++i1) {
      const auto& track_el = point3D.track.Element(i1);
      const image_t image_id1 = track_el.image_id;
      const Image& image1 = reconstruction.Image(image_id1);
      const Camera& camera1 = reconstruction.Camera(image1.CameraId());
      const Point2D& point2D = image1.Point2D(track_el.point2D_idx);
      const double squared_reproj_error = CalculateSquaredReprojectionError(
          point2D.xy, point3D.xyz, image1.CamFromWorld(), camera1);
      EXPECT_LE(squared_reproj_error, kMaxReprojError * kMaxReprojError);
      EXPECT_EQ(descriptors,
                database->ReadDescriptors(point3D.track.Element(i1).image_id)
                    .row(point3D.track.Element(i1).point2D_idx));

      Eigen::Vector3d proj_center1;
      if (proj_centers.count(image_id1) == 0) {
        proj_center1 = image1.ProjectionCenter();
        proj_centers.emplace(image_id1, proj_center1);
      } else {
        proj_center1 = proj_centers.at(image_id1);
      }

      for (size_t i2 = 0; i2 < i1; ++i2) {
        const image_t image_id2 = point3D.track.Element(i2).image_id;
        const Eigen::Vector3d& proj_center2 = proj_centers.at(image_id2);
        max_tri_angle = std::max(max_tri_angle,
                                 CalculateTriangulationAngle(
                                     proj_center1, proj_center2, point3D.xyz));
      }
    }

    EXPECT_GE(max_tri_angle, DegToRad(kMinTriAngleDeg));
  }
}

TEST(SynthesizeDataset, MultipleTimes) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 2;
  options.num_cameras_per_rig = 3;
  options.num_frames_per_rig = 3;
  SynthesizeDataset(options, &reconstruction, database.get());
  SynthesizeDataset(options, &reconstruction, database.get());

  EXPECT_EQ(database->NumRigs(), 2 * options.num_rigs);
  EXPECT_EQ(reconstruction.NumRigs(), database->NumRigs());

  EXPECT_EQ(database->NumCameras(),
            2 * options.num_rigs * options.num_cameras_per_rig);
  EXPECT_EQ(reconstruction.NumCameras(), database->NumCameras());

  EXPECT_EQ(database->NumFrames(),
            2 * options.num_rigs * options.num_frames_per_rig);
  EXPECT_EQ(database->NumFrames(), reconstruction.NumFrames());

  EXPECT_EQ(database->NumImages(),
            2 * options.num_rigs * options.num_cameras_per_rig *
                options.num_frames_per_rig);
  EXPECT_EQ(database->NumImages(), reconstruction.NumImages());

  const int num_image_pairs =
      reconstruction.NumImages() * (reconstruction.NumImages() - 1) / 2;
  EXPECT_EQ(database->NumVerifiedImagePairs(), num_image_pairs);

  EXPECT_EQ(reconstruction.NumPoints3D(), 2 * options.num_points3D);
}

TEST(SynthesizeDataset, WithPriors) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.prior_position = true;
  options.prior_gravity = true;
  options.prior_gravity_in_world = Eigen::Vector3d::Random().normalized();
  SynthesizeDataset(options, &reconstruction, database.get());

  const std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
  std::unordered_map<image_t, const PosePrior*> image_to_prior;
  for (const auto& pose_prior : pose_priors) {
    EXPECT_EQ(pose_prior.corr_data_id.sensor_id.type, SensorType::CAMERA);
    image_to_prior[pose_prior.corr_data_id.id] = &pose_prior;
  }

  for (const auto& [image_id, image] : reconstruction.Images()) {
    EXPECT_TRUE(image_to_prior.count(image_id));
    const PosePrior& pose_prior = *image_to_prior.at(image_id);
    EXPECT_THAT(image.ProjectionCenter(),
                EigenMatrixNear(pose_prior.position, 1e-9));
    EXPECT_THAT(image.CamFromWorld().rotation * options.prior_gravity_in_world,
                EigenMatrixNear(pose_prior.gravity, 1e-9));
  }
}

TEST(SynthesizeDataset, MultiReconstruction) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction1;
  Reconstruction reconstruction2;
  SyntheticDatasetOptions options;
  SynthesizeDataset(options, &reconstruction1, database.get());
  SynthesizeDataset(options, &reconstruction2, database.get());

  const int num_cameras = options.num_rigs * options.num_cameras_per_rig;
  EXPECT_EQ(database->NumCameras(), 2 * num_cameras);
  EXPECT_EQ(reconstruction1.NumCameras(), num_cameras);
  EXPECT_EQ(reconstruction1.NumCameras(), num_cameras);
  const int num_images = num_cameras * options.num_frames_per_rig;
  EXPECT_EQ(database->NumImages(), 2 * num_images);
  EXPECT_EQ(reconstruction1.NumImages(), num_images);
  EXPECT_EQ(reconstruction2.NumImages(), num_images);
  EXPECT_EQ(reconstruction1.NumRegFrames(), num_images);
  EXPECT_EQ(reconstruction2.NumRegFrames(), num_images);
  const int num_image_pairs = num_images * (num_images - 1) / 2;
  EXPECT_EQ(database->NumVerifiedImagePairs(), 2 * num_image_pairs);
  EXPECT_EQ(database->NumInlierMatches(),
            2 * num_image_pairs * options.num_points3D);
}

TEST(SynthesizeDataset, ExhaustiveMatches) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.match_config = SyntheticDatasetOptions::MatchConfig::EXHAUSTIVE;
  SynthesizeDataset(options, &reconstruction, database.get());

  const int num_images = options.num_rigs * options.num_cameras_per_rig *
                         options.num_frames_per_rig;
  const int num_image_pairs = num_images * (num_images - 1) / 2;
  EXPECT_EQ(database->NumMatchedImagePairs(), num_image_pairs);
  EXPECT_EQ(database->NumVerifiedImagePairs(), num_image_pairs);
  EXPECT_EQ(database->NumInlierMatches(),
            num_image_pairs * options.num_points3D);
}

TEST(SynthesizeDataset, ChainedMatches) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.match_config = SyntheticDatasetOptions::MatchConfig::CHAINED;
  SynthesizeDataset(options, &reconstruction, database.get());

  const int num_images = options.num_rigs * options.num_cameras_per_rig *
                         options.num_frames_per_rig;
  const int num_image_pairs = num_images - 1;
  EXPECT_EQ(database->NumMatchedImagePairs(), num_image_pairs);
  EXPECT_EQ(database->NumVerifiedImagePairs(), num_image_pairs);
  EXPECT_EQ(database->NumInlierMatches(),
            num_image_pairs * options.num_points3D);
  for (const auto& [pair_id, _] : database->ReadAllMatches()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    EXPECT_EQ(image_id1 + 1, image_id2);
  }
  for (const auto& [pair_id, _] : database->ReadTwoViewGeometries()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    EXPECT_EQ(image_id1 + 1, image_id2);
  }
}

TEST(SynthesizeDataset, NoDatabase) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  SyntheticDatasetOptions options;
  Reconstruction reconstruction;
  SynthesizeDataset(options, &reconstruction);
}

TEST(SynthesizeDataset, Determinism) {
  SyntheticDatasetOptions options;

  Reconstruction reconstruction1;
  SetPRNGSeed(42);
  SynthesizeDataset(options, &reconstruction1);

  Reconstruction reconstruction2;
  SetPRNGSeed(42);
  SynthesizeDataset(options, &reconstruction2);

  EXPECT_EQ(reconstruction1.Rigs(), reconstruction2.Rigs());
  EXPECT_EQ(reconstruction1.Frames(), reconstruction2.Frames());
  EXPECT_EQ(reconstruction1.Cameras(), reconstruction2.Cameras());
  EXPECT_EQ(reconstruction1.Images(), reconstruction2.Images());
  EXPECT_EQ(reconstruction1.Points3D(), reconstruction2.Points3D());
}

TEST(SynthesizeNoise, Point2DNoise) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  SynthesizeDataset(options, &reconstruction, database.get());
  EXPECT_LT(reconstruction.ComputeMeanReprojectionError(), 1e-3);

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction, database.get());
  EXPECT_GT(reconstruction.ComputeMeanReprojectionError(), 1e-3);

  for (const auto& [image_id, image] : reconstruction.Images()) {
    const auto& keypoints = database->ReadKeypoints(image_id);
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      EXPECT_THAT(
          Eigen::Vector2d(keypoints[point2D_idx].x, keypoints[point2D_idx].y),
          EigenMatrixNear(image.Point2D(point2D_idx).xy, 1e-6));
    }
  }
}

TEST(SynthesizeNoise, Point3DNoise) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;

  SynthesizeDataset(options, &reconstruction, database.get());
  EXPECT_LT(reconstruction.ComputeMeanReprojectionError(), 1e-3);

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point3D_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction, database.get());
  EXPECT_GT(reconstruction.ComputeMeanReprojectionError(), 1e-3);
}

TEST(SynthesizeNoise, RigFromWorldNoise) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;

  SynthesizeDataset(options, &reconstruction, database.get());
  EXPECT_LT(reconstruction.ComputeMeanReprojectionError(), 1e-3);

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.1;
  synthetic_noise_options.rig_from_world_rotation_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction, database.get());
  EXPECT_GT(reconstruction.ComputeMeanReprojectionError(), 1e-3);
}

std::unordered_map<pose_prior_t, PosePrior> ReadPosePriors(Database& database) {
  std::unordered_map<pose_prior_t, PosePrior> pose_priors;
  for (auto& pose_prior : database.ReadAllPosePriors()) {
    pose_priors.emplace(pose_prior.pose_prior_id, std::move(pose_prior));
  }
  return pose_priors;
}

TEST(SynthesizeNoise, PriorPositionNoise) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.prior_position = true;
  options.prior_position_coordinate_system = PosePrior::CoordinateSystem::WGS84;

  SynthesizeDataset(options, &reconstruction, database.get());
  const std::unordered_map<pose_prior_t, PosePrior> orig_pose_priors =
      ReadPosePriors(*database);

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.prior_position_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction, database.get());
  for (const auto& pose_prior : database->ReadAllPosePriors()) {
    // Check that some noise was added.
    EXPECT_THAT(pose_prior.position,
                testing::Not(EigenMatrixEq(
                    orig_pose_priors.at(pose_prior.pose_prior_id).position)));
    // Check that the noisy positions are somewhat close to the original ones.
    EXPECT_THAT(
        pose_prior.position,
        EigenMatrixNear(orig_pose_priors.at(pose_prior.pose_prior_id).position,
                        10 * synthetic_noise_options.prior_position_stddev));
    EXPECT_TRUE(pose_prior.HasPositionCov());
    EXPECT_NEAR(pose_prior.position_covariance.trace() / 3.0,
                synthetic_noise_options.prior_position_stddev *
                    synthetic_noise_options.prior_position_stddev,
                1e-6);
  }
}

TEST(SynthesizeNoise, PriorGravityNoise) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.prior_gravity = true;

  SynthesizeDataset(options, &reconstruction, database.get());
  const std::unordered_map<pose_prior_t, PosePrior> orig_pose_priors =
      ReadPosePriors(*database);

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.prior_gravity_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction, database.get());
  for (const auto& pose_prior : database->ReadAllPosePriors()) {
    const double angle = std::acos(pose_prior.gravity.dot(
        orig_pose_priors.at(pose_prior.pose_prior_id).gravity));
    // Check that some noise was added.
    EXPECT_GT(angle, 0);
    // Check that the noisy directions are somewhat close to the original ones.
    EXPECT_LT(angle, 10 * synthetic_noise_options.prior_gravity_stddev);
  }
}

TEST(SynthesizeImages, Nominal) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 80;
  synthetic_dataset_options.num_points2D_without_point3D = 20;
  synthetic_dataset_options.camera_width = 320;
  synthetic_dataset_options.camera_height = 240;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const std::string test_dir = CreateTestDir();
  const std::string image_path = test_dir + "/images";
  CreateDirIfNotExists(image_path);
  SynthesizeImages(SyntheticImageOptions(), reconstruction, image_path);

  for (const auto& [image_id, image] : reconstruction.Images()) {
    Bitmap bitmap;
    EXPECT_TRUE(bitmap.Read(JoinPaths(image_path, image.Name())));
    EXPECT_EQ(bitmap.Width(), image.CameraPtr()->width);
    EXPECT_EQ(bitmap.Height(), image.CameraPtr()->height);
  }
}

}  // namespace
}  // namespace colmap

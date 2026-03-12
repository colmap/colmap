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

#include "colmap/estimators/rotation_averaging.h"

#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/synthetic.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

void LoadReconstructionAndPoseGraph(const Database& database,
                                    Reconstruction* reconstruction,
                                    PoseGraph* pose_graph) {
  DatabaseCache database_cache;
  DatabaseCache::Options options;
  database_cache.Load(database, options);
  reconstruction->Load(database_cache);
  pose_graph->Load(*database_cache.CorrespondenceGraph());
}

struct TestData {
  std::shared_ptr<Database> database;
  Reconstruction gt_reconstruction;
  Reconstruction reconstruction;
  PoseGraph pose_graph;
  std::vector<PosePrior> pose_priors;
};

TestData CreateTestData(const SyntheticDatasetOptions& dataset_options,
                        const SyntheticNoiseOptions* noise_options = nullptr) {
  TestData data;
  data.database = Database::Open(kInMemorySqliteDatabasePath);
  SynthesizeDataset(
      dataset_options, &data.gt_reconstruction, data.database.get());
  if (noise_options) {
    SynthesizeNoise(
        *noise_options, &data.gt_reconstruction, data.database.get());
  }
  LoadReconstructionAndPoseGraph(
      *data.database, &data.reconstruction, &data.pose_graph);
  data.pose_priors = data.database->ReadAllPosePriors();
  return data;
}

RotationEstimatorOptions CreateRATestOptions(bool use_gravity = false) {
  RotationEstimatorOptions options;
  options.skip_initialization = false;
  options.use_gravity = use_gravity;
  options.use_stratified = true;
  return options;
}

void ExpectEqualRotations(const Reconstruction& gt,
                          const Reconstruction& computed,
                          const double max_rotation_error_deg) {
  const double max_rotation_error_rad = DegToRad(max_rotation_error_deg);
  const std::vector<image_t> reg_image_ids = gt.RegImageIds();
  for (size_t i = 0; i < reg_image_ids.size(); i++) {
    const image_t image_id1 = reg_image_ids[i];
    for (size_t j = 0; j < i; j++) {
      const image_t image_id2 = reg_image_ids[j];
      const Eigen::Quaterniond cam2_from_cam1 =
          computed.Image(image_id2).CamFromWorld().rotation() *
          computed.Image(image_id1).CamFromWorld().rotation().inverse();
      const Eigen::Quaterniond cam2_from_cam1_gt =
          gt.Image(image_id2).CamFromWorld().rotation() *
          gt.Image(image_id1).CamFromWorld().rotation().inverse();
      EXPECT_LE(cam2_from_cam1.angularDistance(cam2_from_cam1_gt),
                max_rotation_error_rad);
    }
  }
}

void ResetSensorsFromRig(Reconstruction& reconstruction) {
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor.has_value()) {
        reconstruction.Rig(rig_id).ResetSensorFromRig(sensor_id);
      }
    }
  }
}

void RunAndVerifyRotationAveraging(const Reconstruction& gt_reconstruction,
                                   const Reconstruction& reconstruction,
                                   const PoseGraph& pose_graph,
                                   const std::vector<PosePrior>& pose_priors,
                                   const std::vector<bool>& use_gravity_values,
                                   const double max_rotation_error_deg) {
  for (const bool use_gravity : use_gravity_values) {
    Reconstruction reconstruction_copy = reconstruction;
    PoseGraph pose_graph_copy = pose_graph;
    RunRotationAveraging(CreateRATestOptions(use_gravity),
                         pose_graph_copy,
                         reconstruction_copy,
                         pose_priors);

    ExpectEqualRotations(
        gt_reconstruction, reconstruction_copy, max_rotation_error_deg);
  }
}

TEST(RotationAveraging, WithoutNoise) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  RunAndVerifyRotationAveraging(data.gt_reconstruction,
                                data.reconstruction,
                                data.pose_graph,
                                data.pose_priors,
                                {true, false},
                                /*max_rotation_error_deg=*/1e-2);
}

TEST(RotationAveraging, WithoutNoiseWithNonTrivialKnownRig) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  RunAndVerifyRotationAveraging(data.gt_reconstruction,
                                data.reconstruction,
                                data.pose_graph,
                                data.pose_priors,
                                {true, false},
                                /*max_rotation_error_deg=*/1e-2);
}

TEST(RotationAveraging, WithoutNoiseWithNonTrivialUnknownRig) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  ResetSensorsFromRig(data.reconstruction);

  // For unknown rigs, it is not supported to use gravity.
  RunAndVerifyRotationAveraging(data.gt_reconstruction,
                                data.reconstruction,
                                data.pose_graph,
                                data.pose_priors,
                                {false},
                                /*max_rotation_error_deg=*/1e-2);
}

TEST(RotationAveraging, WithNoiseAndOutliers) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  synthetic_noise_options.prior_gravity_stddev = 3e-1;
  auto data =
      CreateTestData(synthetic_dataset_options, &synthetic_noise_options);

  RunAndVerifyRotationAveraging(data.gt_reconstruction,
                                data.reconstruction,
                                data.pose_graph,
                                data.pose_priors,
                                {true, false},
                                /*max_rotation_error_deg=*/3);
}

TEST(RotationAveraging, WithNoiseAndOutliersWithNonTrivialKnownRigs) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  synthetic_noise_options.prior_gravity_stddev = 3e-1;
  auto data =
      CreateTestData(synthetic_dataset_options, &synthetic_noise_options);

  RunAndVerifyRotationAveraging(data.gt_reconstruction,
                                data.reconstruction,
                                data.pose_graph,
                                data.pose_priors,
                                {true, false},
                                /*max_rotation_error_deg=*/2.);
}

TEST(RotationAveraging, DeterministicRandomSeed) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  RotationEstimatorOptions options = CreateRATestOptions();
  options.random_seed = 42;

  // Run twice with the same seed and verify identical results.
  Reconstruction reconstruction1 = data.reconstruction;
  PoseGraph pose_graph1 = data.pose_graph;
  EXPECT_TRUE(RunRotationAveraging(
      options, pose_graph1, reconstruction1, data.pose_priors));

  Reconstruction reconstruction2 = data.reconstruction;
  PoseGraph pose_graph2 = data.pose_graph;
  EXPECT_TRUE(RunRotationAveraging(
      options, pose_graph2, reconstruction2, data.pose_priors));

  ExpectEqualRotations(
      reconstruction1, reconstruction2, /*max_rotation_error_deg=*/0);
}

TEST(RotationAveraging, EmptyPoseGraph) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 20;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  // Invalidate all edges so connected components are empty.
  for (auto& [pair_id, edge] : data.pose_graph.Edges()) {
    edge.valid = false;
  }

  RotationEstimatorOptions options = CreateRATestOptions();
  EXPECT_FALSE(RunRotationAveraging(
      options, data.pose_graph, data.reconstruction, data.pose_priors));
}

TEST(RotationAveraging, GravityWithUnknownRigSensorsReturnsFalse) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  ResetSensorsFromRig(data.reconstruction);

  // With gravity enabled and unknown rig sensors, EstimateRotations should
  // fail inside RunRotationAveraging because AllSensorsFromRigKnown returns
  // false. However, RunRotationAveraging takes the HasUnknownCamsFromRig path
  // which creates an expanded reconstruction (singleton rigs) that avoids the
  // AllSensorsFromRigKnown check. To directly hit the
  // AllSensorsFromRigKnown check, we use RotationEstimator directly.
  RotationEstimatorOptions options = CreateRATestOptions(/*use_gravity=*/true);

  std::unordered_set<image_t> active_image_ids;
  for (const auto& [image_id, image] : data.reconstruction.Images()) {
    active_image_ids.insert(image_id);
  }

  RotationEstimator estimator(options);
  EXPECT_FALSE(estimator.EstimateRotations(data.pose_graph,
                                           data.pose_priors,
                                           active_image_ids,
                                           data.reconstruction));
}

// Covers: InitializeRigRotationsFromImages standalone (lines 465-564) with
// multi-camera rig to exercise cam_from_rig estimation and rig_from_world
// averaging.
TEST(RotationAveraging, InitializeSensorFromRigUsingCamsFromWorld) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  // Build cams_from_world from the ground truth.
  std::unordered_map<image_t, Rigid3d> cams_from_world;
  for (const auto& [image_id, image] : data.gt_reconstruction.Images()) {
    if (image.HasPose()) {
      cams_from_world[image_id] = image.CamFromWorld();
    }
  }

  ResetSensorsFromRig(data.reconstruction);

  EXPECT_TRUE(
      InitializeRigRotationsFromImages(cams_from_world, data.reconstruction));

  for (const auto& [rig_id, rig] : data.reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
      EXPECT_LT(sensor_from_rig->rotation().angularDistance(
                    data.gt_reconstruction.Rig(rig_id)
                        .SensorFromRig(sensor_id)
                        .rotation()),
                1e-6);
    }
  }
}

}  // namespace
}  // namespace colmap

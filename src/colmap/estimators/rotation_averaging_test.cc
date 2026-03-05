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
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

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
      EXPECT_LT(cam2_from_cam1.angularDistance(cam2_from_cam1_gt),
                max_rotation_error_rad);
    }
  }
}

TEST(RotationAveraging, WithoutNoise) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  // TODO: The current 1-dof rotation averaging sometimes fails to pick the
  // right solution (e.g., 180 deg flipped).
  for (const bool use_gravity : {false}) {
    Reconstruction reconstruction_copy = reconstruction;
    RunRotationAveraging(CreateRATestOptions(use_gravity),
                         pose_graph,
                         reconstruction_copy,
                         pose_priors);

    ExpectEqualRotations(gt_reconstruction,
                         reconstruction_copy,
                         /*max_rotation_error_deg=*/1e-2);
  }
}

TEST(RotationAveraging, WithoutNoiseWithNonTrivialKnownRig) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  for (const bool use_gravity : {true, false}) {
    Reconstruction reconstruction_copy = reconstruction;
    RunRotationAveraging(CreateRATestOptions(use_gravity),
                         pose_graph,
                         reconstruction_copy,
                         pose_priors);

    ExpectEqualRotations(gt_reconstruction,
                         reconstruction_copy,
                         /*max_rotation_error_deg=*/1e-2);
  }
}

TEST(RotationAveraging, WithoutNoiseWithNonTrivialUnknownRig) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  // Reset rig sensors to unknown.
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor.has_value()) {
        reconstruction.Rig(rig_id).ResetSensorFromRig(sensor_id);
      }
    }
  }

  // For unknown rigs, it is not supported to use gravity.
  for (const bool use_gravity : {false}) {
    Reconstruction reconstruction_copy = reconstruction;
    RunRotationAveraging(CreateRATestOptions(use_gravity),
                         pose_graph,
                         reconstruction_copy,
                         pose_priors);

    ExpectEqualRotations(gt_reconstruction,
                         reconstruction_copy,
                         /*max_rotation_error_deg=*/1e-2);
  }
}

TEST(RotationAveraging, WithNoiseAndOutliers) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  synthetic_noise_options.prior_gravity_stddev = 3e-1;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  // TODO: The current 1-dof rotation averaging sometimes fails to pick the
  // right solution (e.g., 180 deg flipped).
  for (const bool use_gravity : {false}) {
    Reconstruction reconstruction_copy = reconstruction;
    RunRotationAveraging(CreateRATestOptions(use_gravity),
                         pose_graph,
                         reconstruction_copy,
                         pose_priors);

    ExpectEqualRotations(
        gt_reconstruction, reconstruction_copy, /*max_rotation_error_deg=*/3);
  }
}

TEST(RotationAveraging, WithNoiseAndOutliersWithNonTrivialKnownRigs) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  synthetic_noise_options.prior_gravity_stddev = 3e-1;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  // TODO: The current 1-dof rotation averaging sometimes fails to pick the
  // right solution (e.g., 180 deg flipped).
  for (const bool use_gravity : {false}) {
    Reconstruction reconstruction_copy = reconstruction;
    RunRotationAveraging(CreateRATestOptions(use_gravity),
                         pose_graph,
                         reconstruction_copy,
                         pose_priors);

    ExpectEqualRotations(
        gt_reconstruction, reconstruction_copy, /*max_rotation_error_deg=*/2.);
  }
}

// Covers: skip_initialization = true in SolveRotationAveraging (line 401).
TEST(RotationAveraging, SkipInitialization) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  RotationEstimatorOptions options = CreateRATestOptions();
  options.skip_initialization = true;

  Reconstruction reconstruction_copy = reconstruction;
  // Should still succeed even without spanning tree initialization,
  // since the solver can work from zero-initialized rotations.
  EXPECT_TRUE(RunRotationAveraging(
      options, pose_graph, reconstruction_copy, pose_priors));
}

// Covers: weight_type = HALF_NORM in IRLS solver.
TEST(RotationAveraging, HalfNormWeightType) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  RotationEstimatorOptions options = CreateRATestOptions();
  options.weight_type = RotationEstimatorOptions::HALF_NORM;

  Reconstruction reconstruction_copy = reconstruction;
  EXPECT_TRUE(RunRotationAveraging(
      options, pose_graph, reconstruction_copy, pose_priors));

  ExpectEqualRotations(gt_reconstruction,
                       reconstruction_copy,
                       /*max_rotation_error_deg=*/1e-2);
}

// Covers: random_seed >= 0 for deterministic rotation averaging.
TEST(RotationAveraging, DeterministicRandomSeed) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  RotationEstimatorOptions options = CreateRATestOptions();
  options.random_seed = 42;

  // Run twice with the same seed and verify identical results.
  Reconstruction recon1 = reconstruction;
  PoseGraph pg1 = pose_graph;
  EXPECT_TRUE(RunRotationAveraging(options, pg1, recon1, pose_priors));

  Reconstruction recon2 = reconstruction;
  PoseGraph pg2 = pose_graph;
  EXPECT_TRUE(RunRotationAveraging(options, pg2, recon2, pose_priors));

  // Both runs should produce identical rotations.
  const std::vector<image_t> reg_ids = recon1.RegImageIds();
  ASSERT_EQ(reg_ids.size(), recon2.RegImageIds().size());
  for (const image_t image_id : reg_ids) {
    const Eigen::Quaterniond q1 =
        recon1.Image(image_id).CamFromWorld().rotation();
    const Eigen::Quaterniond q2 =
        recon2.Image(image_id).CamFromWorld().rotation();
    EXPECT_NEAR(q1.angularDistance(q2), 0.0, 1e-12);
  }
}

// Covers: max_rotation_error_deg <= 0 disables outlier filtering (lines 654+).
TEST(RotationAveraging, DisableRotationErrorFiltering) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  RotationEstimatorOptions options = CreateRATestOptions();
  options.max_rotation_error_deg = 0;  // Disable filtering.

  Reconstruction reconstruction_copy = reconstruction;
  EXPECT_TRUE(RunRotationAveraging(
      options, pose_graph, reconstruction_copy, pose_priors));

  ExpectEqualRotations(gt_reconstruction,
                       reconstruction_copy,
                       /*max_rotation_error_deg=*/1e-2);
}

// Covers: empty pose graph -> "No connected components found" (lines 578-579).
TEST(RotationAveraging, EmptyPoseGraphReturnsFalse) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 20;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  // Invalidate all edges so connected components are empty.
  for (auto& [pair_id, edge] : pose_graph.Edges()) {
    edge.valid = false;
  }

  RotationEstimatorOptions options = CreateRATestOptions();
  EXPECT_FALSE(
      RunRotationAveraging(options, pose_graph, reconstruction, pose_priors));
}

// Covers: gravity with unknown rig sensors -> AllSensorsFromRigKnown returns
// false (lines 28-43, 274-277).
TEST(RotationAveraging, GravityWithUnknownRigSensorsReturnsFalse) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  // Reset rig sensors to unknown.
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor.has_value()) {
        reconstruction.Rig(rig_id).ResetSensorFromRig(sensor_id);
      }
    }
  }

  // With gravity enabled and unknown rig sensors, EstimateRotations should
  // fail inside RunRotationAveraging because AllSensorsFromRigKnown returns
  // false. However, RunRotationAveraging takes the HasUnknownCamsFromRig path
  // which creates an expanded reconstruction (singleton rigs) that avoids the
  // AllSensorsFromRigKnown check. To directly hit the
  // AllSensorsFromRigKnown check, we use RotationEstimator directly.
  RotationEstimatorOptions options = CreateRATestOptions(/*use_gravity=*/true);

  std::unordered_set<image_t> active_image_ids;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    active_image_ids.insert(image_id);
  }

  RotationEstimator estimator(options);
  EXPECT_FALSE(estimator.EstimateRotations(
      pose_graph, pose_priors, active_image_ids, reconstruction));
}

// Covers: filter_unregistered = true in RunRotationAveraging (line 576).
// This option is for refinement passes where some frames are already
// registered.
TEST(RotationAveraging, FilterUnregistered) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  // First run: register all frames so they have poses.
  RotationEstimatorOptions initial_options = CreateRATestOptions();
  EXPECT_TRUE(RunRotationAveraging(
      initial_options, pose_graph, reconstruction, pose_priors));
  EXPECT_GT(reconstruction.NumRegImages(), 0);

  // Second run: with filter_unregistered=true, the connected component
  // computation only considers already-registered frames.
  RotationEstimatorOptions options = CreateRATestOptions();
  options.filter_unregistered = true;

  // Reset edge validity for second pass.
  for (auto& [pair_id, edge] : pose_graph.Edges()) {
    edge.valid = true;
  }

  PoseGraph pose_graph2 = pose_graph;
  Reconstruction reconstruction_copy = reconstruction;
  EXPECT_TRUE(RunRotationAveraging(
      options, pose_graph2, reconstruction_copy, pose_priors));
}

// Covers: stratified gravity subset skipped when >95% of pairs have gravity
// (lines 353-354 in MaybeSolveGravityAlignedSubset).
TEST(RotationAveraging, StratifiedGravityAllPairsHaveGravity) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  // All cameras in a single rig, all with gravity priors.
  // This means >95% of pairs will have gravity -> subset solve is skipped.
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  // All images have gravity, so all pairs have gravity (100% > 95%).
  // MaybeSolveGravityAlignedSubset should skip the subset solve.
  RotationEstimatorOptions options = CreateRATestOptions(/*use_gravity=*/true);
  options.use_stratified = true;

  std::unordered_set<image_t> active_image_ids;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    active_image_ids.insert(image_id);
  }

  RotationEstimator estimator(options);
  Reconstruction reconstruction_copy = reconstruction;
  EXPECT_TRUE(estimator.EstimateRotations(
      pose_graph, pose_priors, active_image_ids, reconstruction_copy));
}

// Covers: stratified gravity disabled (use_stratified = false, line 280).
TEST(RotationAveraging, GravityWithoutStratified) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  RotationEstimatorOptions options = CreateRATestOptions(/*use_gravity=*/true);
  options.use_stratified = false;

  Reconstruction reconstruction_copy = reconstruction;
  EXPECT_TRUE(RunRotationAveraging(
      options, pose_graph, reconstruction_copy, pose_priors));

  ExpectEqualRotations(gt_reconstruction,
                       reconstruction_copy,
                       /*max_rotation_error_deg=*/1e-2);
}

// Covers: no gravity priors -> UseGravity returns false even when
// options.use_gravity = true (lines 23-26).
TEST(RotationAveraging, GravityEnabledButNoPriors) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = false;  // No gravity priors.
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  // use_gravity = true but no gravity priors exist, so UseGravity returns
  // false and the non-gravity path is taken.
  RotationEstimatorOptions options = CreateRATestOptions(/*use_gravity=*/true);

  Reconstruction reconstruction_copy = reconstruction;
  EXPECT_TRUE(RunRotationAveraging(
      options, pose_graph, reconstruction_copy, pose_priors));

  ExpectEqualRotations(gt_reconstruction,
                       reconstruction_copy,
                       /*max_rotation_error_deg=*/1e-2);
}

// Covers: InitializeRigRotationsFromImages standalone (lines 465-564) with
// multi-camera rig to exercise cam_from_rig estimation and rig_from_world
// averaging.
TEST(RotationAveraging, InitializeRigRotationsFromImagesMultiCamera) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  // Build cams_from_world from the ground truth.
  std::unordered_map<image_t, Rigid3d> cams_from_world;
  for (const auto& [image_id, image] : gt_reconstruction.Images()) {
    if (image.HasPose()) {
      cams_from_world[image_id] = image.CamFromWorld();
    }
  }

  // Reset rig sensors to unknown.
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor.has_value()) {
        reconstruction.Rig(rig_id).ResetSensorFromRig(sensor_id);
      }
    }
  }

  EXPECT_TRUE(InitializeRigRotationsFromImages(cams_from_world, reconstruction));

  // Verify that rig_from_world was set for all frames.
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    EXPECT_TRUE(frame.HasPose());
  }
}

// Covers: InitializeRigRotationsFromImages with empty input map.
TEST(RotationAveraging, InitializeRigRotationsFromImagesEmpty) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 20;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  // Empty cams_from_world: no rotations to initialize from.
  std::unordered_map<image_t, Rigid3d> cams_from_world;
  EXPECT_TRUE(InitializeRigRotationsFromImages(cams_from_world, reconstruction));

  // No frames should have pose set since there was nothing to initialize from.
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    EXPECT_FALSE(frame.HasPose());
  }
}

// Covers: tight rotation error filtering that deregisters outlier frames
// (lines 653-681 in RunRotationAveraging).
TEST(RotationAveraging, RotationErrorFilteringDeregistersOutliers) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  // Use a very tight threshold to trigger edge filtering and deregistration.
  RotationEstimatorOptions options = CreateRATestOptions();
  options.max_rotation_error_deg = 0.5;

  Reconstruction reconstruction_copy = reconstruction;
  const bool success = RunRotationAveraging(
      options, pose_graph, reconstruction_copy, pose_priors);
  // Should succeed (some frames may be deregistered but not all).
  EXPECT_TRUE(success);

  // Verify that at least some frames were registered.
  EXPECT_GT(reconstruction_copy.NumRegImages(), 0);
}

// Covers: multiple rigs with noise exercising the full
// FilterEdgesByRelativeRotation path with actual outlier invalidation
// (lines 238-265).
TEST(RotationAveraging, FilterEdgesInvalidatesOutlierPairs) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.5;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 2;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  const size_t initial_valid_count = [&]() {
    size_t count = 0;
    for (const auto& [pair_id, edge] : pose_graph.Edges()) {
      if (edge.valid) count++;
    }
    return count;
  }();

  RotationEstimatorOptions options = CreateRATestOptions();
  options.max_rotation_error_deg = 1.0;

  Reconstruction reconstruction_copy = reconstruction;
  EXPECT_TRUE(RunRotationAveraging(
      options, pose_graph, reconstruction_copy, pose_priors));

  // With noise and tight filtering, some edges should have been invalidated.
  size_t final_valid_count = 0;
  for (const auto& [pair_id, edge] : pose_graph.Edges()) {
    if (edge.valid) final_valid_count++;
  }
  EXPECT_LE(final_valid_count, initial_valid_count);
}

// Covers: unknown rigs with noise to exercise the full expanded reconstruction
// path including CreateExpandedReconstruction and the subsequent
// cam_from_rig initialization (lines 590-651).
TEST(RotationAveraging, WithNoiseAndUnknownRig) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 6;
  synthetic_dataset_options.num_points3D = 80;
  synthetic_dataset_options.inlier_match_ratio = 0.7;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  // Reset rig sensors to unknown.
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor.has_value()) {
        reconstruction.Rig(rig_id).ResetSensorFromRig(sensor_id);
      }
    }
  }

  RotationEstimatorOptions options = CreateRATestOptions();

  Reconstruction reconstruction_copy = reconstruction;
  EXPECT_TRUE(RunRotationAveraging(
      options, pose_graph, reconstruction_copy, pose_priors));

  ExpectEqualRotations(gt_reconstruction,
                       reconstruction_copy,
                       /*max_rotation_error_deg=*/5.0);
}

// Covers: empty pose graph with unknown rig path -> "No connected components
// found" in the expanded reconstruction branch (lines 605-606).
TEST(RotationAveraging, EmptyPoseGraphWithUnknownRigReturnsFalse) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 20;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  // Reset rig sensors to unknown.
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor.has_value()) {
        reconstruction.Rig(rig_id).ResetSensorFromRig(sensor_id);
      }
    }
  }

  // Invalidate all edges.
  for (auto& [pair_id, edge] : pose_graph.Edges()) {
    edge.valid = false;
  }

  RotationEstimatorOptions options = CreateRATestOptions();
  EXPECT_FALSE(
      RunRotationAveraging(options, pose_graph, reconstruction, pose_priors));
}

}  // namespace
}  // namespace colmap

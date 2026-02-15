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

}  // namespace
}  // namespace colmap

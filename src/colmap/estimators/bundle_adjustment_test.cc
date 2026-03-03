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

#include "colmap/estimators/bundle_adjustment.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(BundleAdjustmentOptions, Copy) {
  BundleAdjustmentOptions options;
  options.refine_focal_length = false;
  options.refine_principal_point = true;
  options.min_track_length = 5;
  options.ceres->solver_options.max_num_iterations = 42;

  BundleAdjustmentOptions copy = options;

  // Verify fields are copied
  EXPECT_EQ(copy.refine_focal_length, false);
  EXPECT_EQ(copy.refine_principal_point, true);
  EXPECT_EQ(copy.min_track_length, 5);
  EXPECT_EQ(copy.ceres->solver_options.max_num_iterations, 42);

  // Verify deep copy of shared_ptr (different pointer instances)
  EXPECT_NE(options.ceres.get(), copy.ceres.get());
}

TEST(PosePriorBundleAdjustmentOptions, Copy) {
  PosePriorBundleAdjustmentOptions options;
  options.prior_position_fallback_stddev = 2.5;
  options.alignment_ransac_options.max_error = 1.0;
  options.ceres->prior_position_loss_scale = 0.42;

  PosePriorBundleAdjustmentOptions copy = options;

  // Verify fields are copied
  EXPECT_EQ(copy.prior_position_fallback_stddev, 2.5);
  EXPECT_EQ(copy.alignment_ransac_options.max_error, 1.0);
  EXPECT_EQ(copy.ceres->prior_position_loss_scale, 0.42);

  // Verify deep copy of shared_ptr (different pointer instances)
  EXPECT_NE(options.ceres.get(), copy.ceres.get());
}

TEST(BundleAdjustmentSummary, IsSolutionUsable) {
  BundleAdjustmentSummary summary;

  summary.termination_type = BundleAdjustmentTerminationType::CONVERGENCE;
  EXPECT_TRUE(summary.IsSolutionUsable());

  summary.termination_type = BundleAdjustmentTerminationType::NO_CONVERGENCE;
  EXPECT_TRUE(summary.IsSolutionUsable());

  summary.termination_type = BundleAdjustmentTerminationType::USER_SUCCESS;
  EXPECT_TRUE(summary.IsSolutionUsable());

  summary.termination_type = BundleAdjustmentTerminationType::FAILURE;
  EXPECT_FALSE(summary.IsSolutionUsable());

  summary.termination_type = BundleAdjustmentTerminationType::USER_FAILURE;
  EXPECT_FALSE(summary.IsSolutionUsable());
}

TEST(BundleAdjustmentConfig, NumResiduals) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 4;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const std::vector<image_t> image_ids = reconstruction.RegImageIds();
  CHECK_EQ(image_ids.size(), 4);

  BundleAdjustmentConfig config;

  config.AddImage(image_ids[0]);
  config.AddImage(image_ids[1]);
  EXPECT_EQ(config.NumResiduals(reconstruction), 400);

  config.AddVariablePoint(1);
  EXPECT_EQ(config.NumResiduals(reconstruction), 404);

  config.AddConstantPoint(2);
  EXPECT_EQ(config.NumResiduals(reconstruction), 408);

  config.AddImage(image_ids[2]);
  EXPECT_EQ(config.NumResiduals(reconstruction), 604);

  config.AddImage(image_ids[3]);
  EXPECT_EQ(config.NumResiduals(reconstruction), 800);

  config.IgnorePoint(3);
  EXPECT_EQ(config.NumResiduals(reconstruction), 792);
}

TEST(BundleAdjustmentConfig, AddRemoveImage) {
  BundleAdjustmentConfig config;
  EXPECT_EQ(config.NumImages(), 0);

  config.AddImage(1);
  config.AddImage(2);
  config.AddImage(3);
  EXPECT_EQ(config.NumImages(), 3);
  EXPECT_TRUE(config.HasImage(1));
  EXPECT_TRUE(config.HasImage(2));
  EXPECT_TRUE(config.HasImage(3));
  EXPECT_FALSE(config.HasImage(4));

  config.RemoveImage(2);
  EXPECT_EQ(config.NumImages(), 2);
  EXPECT_TRUE(config.HasImage(1));
  EXPECT_FALSE(config.HasImage(2));
  EXPECT_TRUE(config.HasImage(3));

  // Removing non-existent image is a no-op
  config.RemoveImage(99);
  EXPECT_EQ(config.NumImages(), 2);
}

TEST(BundleAdjustmentConfig, ConstantVariableCamIntrinsics) {
  BundleAdjustmentConfig config;
  EXPECT_EQ(config.NumConstantCamIntrinsics(), 0);

  config.SetConstantCamIntrinsics(1);
  config.SetConstantCamIntrinsics(2);
  EXPECT_EQ(config.NumConstantCamIntrinsics(), 2);
  EXPECT_TRUE(config.HasConstantCamIntrinsics(1));
  EXPECT_TRUE(config.HasConstantCamIntrinsics(2));
  EXPECT_FALSE(config.HasConstantCamIntrinsics(3));

  config.SetVariableCamIntrinsics(1);
  EXPECT_EQ(config.NumConstantCamIntrinsics(), 1);
  EXPECT_FALSE(config.HasConstantCamIntrinsics(1));
  EXPECT_TRUE(config.HasConstantCamIntrinsics(2));

  const auto& constant_cams = config.ConstantCamIntrinsics();
  EXPECT_EQ(constant_cams.size(), 1);
  EXPECT_EQ(constant_cams.count(2), 1);
}

TEST(BundleAdjustmentConfig, ConstantVariableSensorFromRigPose) {
  BundleAdjustmentConfig config;
  EXPECT_EQ(config.NumConstantSensorFromRigPoses(), 0);

  sensor_t sensor1(SensorType::CAMERA, 1);
  sensor_t sensor2(SensorType::CAMERA, 2);

  config.SetConstantSensorFromRigPose(sensor1);
  config.SetConstantSensorFromRigPose(sensor2);
  EXPECT_EQ(config.NumConstantSensorFromRigPoses(), 2);
  EXPECT_TRUE(config.HasConstantSensorFromRigPose(sensor1));
  EXPECT_TRUE(config.HasConstantSensorFromRigPose(sensor2));

  config.SetVariableSensorFromRigPose(sensor1);
  EXPECT_EQ(config.NumConstantSensorFromRigPoses(), 1);
  EXPECT_FALSE(config.HasConstantSensorFromRigPose(sensor1));
  EXPECT_TRUE(config.HasConstantSensorFromRigPose(sensor2));

  const auto& constant_poses = config.ConstantSensorFromRigPoses();
  EXPECT_EQ(constant_poses.size(), 1);
  EXPECT_EQ(constant_poses.count(sensor2), 1);
}

TEST(BundleAdjustmentConfig, ConstantVariableRigFromWorldPose) {
  BundleAdjustmentConfig config;
  EXPECT_EQ(config.NumConstantRigFromWorldPoses(), 0);

  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);
  EXPECT_EQ(config.NumConstantRigFromWorldPoses(), 2);
  EXPECT_TRUE(config.HasConstantRigFromWorldPose(1));
  EXPECT_TRUE(config.HasConstantRigFromWorldPose(2));

  config.SetVariableRigFromWorldPose(1);
  EXPECT_EQ(config.NumConstantRigFromWorldPoses(), 1);
  EXPECT_FALSE(config.HasConstantRigFromWorldPose(1));
  EXPECT_TRUE(config.HasConstantRigFromWorldPose(2));

  const auto& constant_rig_poses = config.ConstantRigFromWorldPoses();
  EXPECT_EQ(constant_rig_poses.size(), 1);
  EXPECT_EQ(constant_rig_poses.count(2), 1);
}

TEST(BundleAdjustmentConfig, ConstantVariablePoints) {
  BundleAdjustmentConfig config;
  EXPECT_EQ(config.NumPoints(), 0);
  EXPECT_EQ(config.NumVariablePoints(), 0);
  EXPECT_EQ(config.NumConstantPoints(), 0);

  config.AddVariablePoint(1);
  config.AddVariablePoint(2);
  EXPECT_EQ(config.NumPoints(), 2);
  EXPECT_EQ(config.NumVariablePoints(), 2);
  EXPECT_EQ(config.NumConstantPoints(), 0);
  EXPECT_TRUE(config.HasPoint(1));
  EXPECT_TRUE(config.HasVariablePoint(1));
  EXPECT_FALSE(config.HasConstantPoint(1));

  config.AddConstantPoint(3);
  EXPECT_EQ(config.NumPoints(), 3);
  EXPECT_EQ(config.NumVariablePoints(), 2);
  EXPECT_EQ(config.NumConstantPoints(), 1);
  EXPECT_TRUE(config.HasPoint(3));
  EXPECT_FALSE(config.HasVariablePoint(3));
  EXPECT_TRUE(config.HasConstantPoint(3));

  config.RemoveVariablePoint(1);
  EXPECT_EQ(config.NumVariablePoints(), 1);
  EXPECT_FALSE(config.HasPoint(1));

  config.RemoveConstantPoint(3);
  EXPECT_EQ(config.NumConstantPoints(), 0);
  EXPECT_FALSE(config.HasPoint(3));

  const auto& var_points = config.VariablePoints();
  EXPECT_EQ(var_points.size(), 1);
  EXPECT_EQ(var_points.count(2), 1);
  const auto& const_points = config.ConstantPoints();
  EXPECT_TRUE(const_points.empty());
}

TEST(BundleAdjustmentConfig, IgnoredPoints) {
  BundleAdjustmentConfig config;
  EXPECT_FALSE(config.IsIgnoredPoint(1));

  config.IgnorePoint(1);
  EXPECT_TRUE(config.IsIgnoredPoint(1));
  EXPECT_FALSE(config.IsIgnoredPoint(2));
}

TEST(BundleAdjustmentConfig, FixGauge) {
  BundleAdjustmentConfig config;
  EXPECT_EQ(config.FixedGauge(), BundleAdjustmentGauge::UNSPECIFIED);

  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  EXPECT_EQ(config.FixedGauge(), BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);
  EXPECT_EQ(config.FixedGauge(), BundleAdjustmentGauge::THREE_POINTS);
}

TEST(BundleAdjustmentConfig, Images) {
  BundleAdjustmentConfig config;
  config.AddImage(5);
  config.AddImage(10);

  const auto& images = config.Images();
  EXPECT_EQ(images.size(), 2);
  EXPECT_EQ(images.count(5), 1);
  EXPECT_EQ(images.count(10), 1);
}

TEST(BundleAdjustmentSummary, BriefReport) {
  BundleAdjustmentSummary summary;
  summary.termination_type = BundleAdjustmentTerminationType::CONVERGENCE;
  summary.num_residuals = 42;

  const std::string report = summary.BriefReport();
  EXPECT_NE(report.find("CONVERGENCE"), std::string::npos);
  EXPECT_NE(report.find("42"), std::string::npos);
}

// Parameterized test for generic BundleAdjuster interface across backends.
class BundleAdjusterBackendTest
    : public ::testing::TestWithParam<BundleAdjustmentBackend> {};

TEST_P(BundleAdjusterBackendTest, Nominal) {
  SetPRNGSeed(0);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 200;
  SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction);

  Reconstruction reconstruction = gt_reconstruction;

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.point3D_stddev = 0.1;
  synthetic_noise_options.rig_from_world_rotation_stddev = 0.5;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.backend = GetParam();

  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultBundleAdjuster(options, config, reconstruction);

  // Test abstract interface accessors
  EXPECT_EQ(bundle_adjuster->Options().backend, GetParam());
  EXPECT_EQ(bundle_adjuster->Config().NumImages(), 10);

  // Solve and verify through abstract interface
  const auto summary = bundle_adjuster->Solve();
  EXPECT_TRUE(summary->IsSolutionUsable());
  EXPECT_GT(summary->num_residuals, 0);

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
}

INSTANTIATE_TEST_SUITE_P(BundleAdjusterBackends,
                         BundleAdjusterBackendTest,
                         ::testing::Values(BundleAdjustmentBackend::CERES));

// Parameterized test for generic PosePriorBundleAdjuster interface across
// backends.
class PosePriorBundleAdjusterBackendTest
    : public ::testing::TestWithParam<BundleAdjustmentBackend> {};

TEST_P(PosePriorBundleAdjusterBackendTest, Nominal) {
  SetPRNGSeed(0);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.prior_position = true;
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction = gt_reconstruction;

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.point3D_stddev = 0.1;
  synthetic_noise_options.rig_from_world_rotation_stddev = 0.5;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.1;
  synthetic_noise_options.prior_position_stddev = 0.05;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  BundleAdjustmentConfig config;
  for (const frame_t frame_id : reconstruction.RegFrameIds()) {
    const Frame& frame = reconstruction.Frame(frame_id);
    for (const data_t& data_id : frame.ImageIds()) {
      config.AddImage(data_id.id);
    }
  }

  BundleAdjustmentOptions options;
  options.backend = GetParam();

  PosePriorBundleAdjustmentOptions prior_options;
  prior_options.alignment_ransac_options.random_seed = 0;

  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreatePosePriorBundleAdjuster(
          options, prior_options, config, pose_priors, reconstruction);

  // Test abstract interface accessors
  EXPECT_EQ(bundle_adjuster->Options().backend, GetParam());
  EXPECT_EQ(bundle_adjuster->Config().NumImages(), 7);

  // Solve and verify through abstract interface
  const auto summary = bundle_adjuster->Solve();
  EXPECT_TRUE(summary->IsSolutionUsable());
  EXPECT_GT(summary->num_residuals, 0);

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

INSTANTIATE_TEST_SUITE_P(PosePriorBundleAdjusterBackends,
                         PosePriorBundleAdjusterBackendTest,
                         ::testing::Values(BundleAdjustmentBackend::CERES));

}  // namespace
}  // namespace colmap

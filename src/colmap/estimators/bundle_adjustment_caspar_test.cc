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

#include "colmap/estimators/bundle_adjustment_caspar.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/models.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

// Due to pose normalization operations, constant variables may not be perfectly
// fixed during bundle adjustment.
constexpr double kConstantPoseVarEps = 1e-9;

#define CheckVariableCamera(camera, orig_camera)       \
  {                                                    \
    const size_t focal_length_idx =                    \
        SimpleRadialCameraModel::focal_length_idxs[0]; \
    const size_t extra_param_idx =                     \
        SimpleRadialCameraModel::extra_params_idxs[0]; \
    EXPECT_NE((camera).params[focal_length_idx],       \
              (orig_camera).params[focal_length_idx]); \
    EXPECT_NE((camera).params[extra_param_idx],        \
              (orig_camera).params[extra_param_idx]);  \
  }

#define CheckConstantCamera(camera, orig_camera)       \
  {                                                    \
    const size_t focal_length_idx =                    \
        SimpleRadialCameraModel::focal_length_idxs[0]; \
    const size_t extra_param_idx =                     \
        SimpleRadialCameraModel::extra_params_idxs[0]; \
    EXPECT_EQ((camera).params[focal_length_idx],       \
              (orig_camera).params[focal_length_idx]); \
    EXPECT_EQ((camera).params[extra_param_idx],        \
              (orig_camera).params[extra_param_idx]);  \
  }

#define CheckVariableCamFromWorld(image, orig_image)                   \
  {                                                                    \
    EXPECT_THAT((image).CamFromWorld(),                                \
                testing::Not(Rigid3dEq((orig_image).CamFromWorld()))); \
  }

#define CheckConstantCamFromWorld(image, orig_image)     \
  {                                                      \
    EXPECT_THAT((image).CamFromWorld(),                  \
                Rigid3dNear((orig_image).CamFromWorld(), \
                            kConstantPoseVarEps,         \
                            kConstantPoseVarEps));       \
  }

#define CheckConstantCamFromWorldTranslationCoord(image, orig_image) \
  {                                                                  \
    size_t num_constant_coords = 0;                                  \
    for (int i = 0; i < 3; ++i) {                                    \
      if (std::abs((image).CamFromWorld().translation()(i) -         \
                   (orig_image).CamFromWorld().translation()(i)) <   \
          kConstantPoseVarEps) {                                     \
        ++num_constant_coords;                                       \
      }                                                              \
    }                                                                \
    EXPECT_EQ(num_constant_coords, 1);                               \
  }

#define CheckVariablePoint(point, orig_point) \
  { EXPECT_NE((point).xyz, (orig_point).xyz); }

#define CheckConstantPoint(point, orig_point) \
  { EXPECT_EQ((point).xyz, (orig_point).xyz); }

namespace colmap {
namespace {

TEST(DefaultBundleAdjuster, Nominal) {
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

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

#ifdef CASPAR_USE_DOUBLE
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
#else
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.3,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
#endif
}

TEST(DefaultBundleAdjuster, NominalMultiCameraRig) {
  SetPRNGSeed(0);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 5;
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
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

#ifdef CASPAR_USE_DOUBLE
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
#else
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.3,
                                 /*max_proj_center_error=*/0.3,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
#endif
}

TEST(DefaultBundleAdjuster, TwoView) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  // Caspar does not implement complete gauge fixing
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(summary->num_residuals, 400);

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckConstantCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));

  CheckVariableCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
  CheckConstantCamFromWorld(reconstruction.Image(2),
                            orig_reconstruction.Image(2));

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    CheckVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
}

TEST(DefaultBundleAdjuster, PartiallyContainedTracks) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 3;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const auto variable_point3D_id =
      reconstruction.Image(3).Point2D(0).point3D_id;
  reconstruction.DeleteObservation(3, 0);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  // Caspar does not implement complete gauge fixing
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckConstantCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));

  CheckVariableCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
  CheckConstantCamFromWorld(reconstruction.Image(2),
                            orig_reconstruction.Image(2));

  CheckConstantCamera(reconstruction.Camera(3), orig_reconstruction.Camera(3));
  CheckConstantCamFromWorld(reconstruction.Image(3),
                            orig_reconstruction.Image(3));

  for (const auto& point3D : reconstruction.Points3D()) {
    if (point3D.first == variable_point3D_id) {
      CheckVariablePoint(point3D.second,
                         orig_reconstruction.Point3D(point3D.first));
    } else {
      CheckConstantPoint(point3D.second,
                         orig_reconstruction.Point3D(point3D.first));
    }
  }
}

TEST(DefaultBundleAdjuster, ConstantPoints) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const auto orig_reconstruction = reconstruction;

  const point3D_t constant_point3D_id1 = 1;
  const point3D_t constant_point3D_id2 = 2;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.AddConstantPoint(constant_point3D_id1);
  config.AddConstantPoint(constant_point3D_id2);
  // Caspar does not implement complete gauge fixing
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(summary->num_residuals, 400);

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckConstantCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));

  CheckVariableCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
  CheckConstantCamFromWorld(reconstruction.Image(2),
                            orig_reconstruction.Image(2));

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D_id == constant_point3D_id1 ||
        point3D_id == constant_point3D_id2) {
      CheckConstantPoint(point3D, orig_reconstruction.Point3D(point3D_id));
    } else {
      CheckVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
    }
  }
}

TEST(DefaultBundleAdjuster, VariableImage) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 3;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.AddImage(3);
  // Caspar does not implement complete gauge fixing
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // 100 points, 3 images, 2 residuals per point per image
  EXPECT_EQ(summary->num_residuals, 600);

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckConstantCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));

  CheckVariableCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
  CheckConstantCamFromWorld(reconstruction.Image(2),
                            orig_reconstruction.Image(2));

  CheckVariableCamera(reconstruction.Camera(3), orig_reconstruction.Camera(3));
  CheckVariableCamFromWorld(reconstruction.Image(3),
                            orig_reconstruction.Image(3));

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    CheckVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
}

TEST(DefaultBundleAdjuster, ConstantFocalLengthAndExtraParams) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  // Caspar does not implement complete gauge fixing
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);

  BundleAdjustmentOptions options;
  options.refine_focal_length = false;
  options.refine_extra_params = false;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(summary->num_residuals, 400);

  CheckConstantCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));
  CheckConstantCamFromWorld(reconstruction.Image(2),
                            orig_reconstruction.Image(2));

  const size_t focal_length_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t extra_param_idx = SimpleRadialCameraModel::extra_params_idxs[0];

  const auto& camera0 = reconstruction.Camera(1);
  const auto& orig_camera0 = orig_reconstruction.Camera(1);
  EXPECT_TRUE(camera0.params[focal_length_idx] ==
              orig_camera0.params[focal_length_idx]);
  EXPECT_TRUE(camera0.params[extra_param_idx] ==
              orig_camera0.params[extra_param_idx]);

  const auto& camera1 = reconstruction.Camera(2);
  const auto& orig_camera1 = orig_reconstruction.Camera(2);
  EXPECT_TRUE(camera1.params[focal_length_idx] ==
              orig_camera1.params[focal_length_idx]);
  EXPECT_TRUE(camera1.params[extra_param_idx] ==
              orig_camera1.params[extra_param_idx]);

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    CheckVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
}

TEST(DefaultBundleAdjuster, VariablePrincipalPoint) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  // Caspar does not implement complete gauge fixing
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);

  BundleAdjustmentOptions options;
  options.refine_principal_point = true;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(summary->num_residuals, 400);

  CheckConstantCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));
  CheckConstantCamFromWorld(reconstruction.Image(2),
                            orig_reconstruction.Image(2));

  const size_t focal_length_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t principal_point_idx_x =
      SimpleRadialCameraModel::principal_point_idxs[0];
  const size_t principal_point_idx_y =
      SimpleRadialCameraModel::principal_point_idxs[0];
  const size_t extra_param_idx = SimpleRadialCameraModel::extra_params_idxs[0];

  const auto& camera0 = reconstruction.Camera(1);
  const auto& orig_camera0 = orig_reconstruction.Camera(1);
  EXPECT_TRUE(camera0.params[focal_length_idx] !=
              orig_camera0.params[focal_length_idx]);
  EXPECT_TRUE(camera0.params[principal_point_idx_x] !=
              orig_camera0.params[principal_point_idx_x]);
  EXPECT_TRUE(camera0.params[principal_point_idx_y] !=
              orig_camera0.params[principal_point_idx_y]);
  EXPECT_TRUE(camera0.params[extra_param_idx] !=
              orig_camera0.params[extra_param_idx]);

  const auto& camera1 = reconstruction.Camera(2);
  const auto& orig_camera1 = orig_reconstruction.Camera(2);
  EXPECT_TRUE(camera1.params[focal_length_idx] !=
              orig_camera1.params[focal_length_idx]);
  EXPECT_TRUE(camera1.params[principal_point_idx_x] !=
              orig_camera1.params[principal_point_idx_x]);
  EXPECT_TRUE(camera1.params[principal_point_idx_y] !=
              orig_camera1.params[principal_point_idx_y]);
  EXPECT_TRUE(camera1.params[extra_param_idx] !=
              orig_camera1.params[extra_param_idx]);

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    CheckVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
}

TEST(DefaultBundleAdjuster, MergedCalibConvergence) {
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

  BundleAdjustmentOptions options;
  options.refine_principal_point = true;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

#ifdef CASPAR_USE_DOUBLE
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
#else
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.3,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
#endif
}

TEST(DefaultBundleAdjuster, MergedCalibFixedPose) {
  // Verifies that all four intrinsic parameters change.
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);

  BundleAdjustmentOptions options;
  options.refine_principal_point = true;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  const size_t focal_length_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t principal_point_idx_x =
      SimpleRadialCameraModel::principal_point_idxs[0];
  const size_t principal_point_idx_y =
      SimpleRadialCameraModel::principal_point_idxs[1];
  const size_t extra_param_idx = SimpleRadialCameraModel::extra_params_idxs[0];

  for (const camera_t cam_id : {camera_t{1}, camera_t{2}}) {
    const auto& cam = reconstruction.Camera(cam_id);
    const auto& orig_cam = orig_reconstruction.Camera(cam_id);
    EXPECT_NE(cam.params[focal_length_idx], orig_cam.params[focal_length_idx]);
    EXPECT_NE(cam.params[extra_param_idx], orig_cam.params[extra_param_idx]);
    EXPECT_NE(cam.params[principal_point_idx_x],
              orig_cam.params[principal_point_idx_x]);
    EXPECT_NE(cam.params[principal_point_idx_y],
              orig_cam.params[principal_point_idx_y]);
  }

  CheckConstantCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));
  CheckConstantCamFromWorld(reconstruction.Image(2),
                            orig_reconstruction.Image(2));
}

TEST(DefaultBundleAdjuster, MergedCalibFixedPoint) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  // Fix all 3D points; only pose and calib are free.
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    config.AddConstantPoint(point3D_id);
  }

  BundleAdjustmentOptions options;
  options.refine_principal_point = true;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    CheckConstantPoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
}

TEST(DefaultBundleAdjuster, IgnorePoint) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.IgnorePoint(42);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // 99 points (point 42 ignored), 2 images, 2 residuals per point per image
  EXPECT_EQ(summary->num_residuals, 396);
}

TEST(DefaultBundleAdjuster, ExternalImagePoseIsInvariant) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 3;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  opts.num_points2D_without_point3D = 0;
  SynthesizeDataset(opts, &reconstruction);

  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.rig_from_world_rotation_stddev = 0.5;
  noise_opts.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    config.AddVariablePoint(point3D_id);
  }

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  CheckConstantCamFromWorld(reconstruction.Image(3),
                            orig_reconstruction.Image(3));
}

TEST(DefaultBundleAdjuster, ExternalCameraIntrinsicsOrderingIsConsistent) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 3;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  opts.num_points2D_without_point3D = 0;
  SynthesizeDataset(opts, &reconstruction);

  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    config.AddVariablePoint(point3D_id);
  }

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  CheckConstantCamera(reconstruction.Camera(3), orig_reconstruction.Camera(3));
}

TEST(DefaultBundleAdjuster, ExternalImageViaConstantPointsIsInvariant) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 3;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  opts.num_points2D_without_point3D = 0;
  SynthesizeDataset(opts, &reconstruction);

  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.rig_from_world_rotation_stddev = 0.5;
  noise_opts.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    config.AddConstantPoint(point3D_id);
  }

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  CheckConstantCamFromWorld(reconstruction.Image(3),
                            orig_reconstruction.Image(3));
}

TEST(DefaultBundleAdjuster, MultipleExternalImagesAreInvariant) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 4;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  opts.num_points2D_without_point3D = 0;
  SynthesizeDataset(opts, &reconstruction);

  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.rig_from_world_rotation_stddev = 0.5;
  noise_opts.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    config.AddVariablePoint(point3D_id);
  }

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  CheckConstantCamFromWorld(reconstruction.Image(3),
                            orig_reconstruction.Image(3));
  CheckConstantCamFromWorld(reconstruction.Image(4),
                            orig_reconstruction.Image(4));
}

TEST(DefaultBundleAdjuster, MergedCalibMatchesCeres) {
  SetPRNGSeed(0);
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.point3D_stddev = 0.1;
  synthetic_noise_options.rig_from_world_rotation_stddev = 0.3;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.05;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);

  BundleAdjustmentOptions options;
  options.refine_principal_point = true;

  Reconstruction reconstruction_ceres = reconstruction;
  Reconstruction reconstruction_caspar = reconstruction;

  std::unique_ptr<BundleAdjuster> ceres_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction_ceres);
  ASSERT_NE(ceres_adjuster->Solve()->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  std::unique_ptr<BundleAdjuster> caspar_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction_caspar);
  ASSERT_NE(caspar_adjuster->Solve()->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // Layout bugs in the merged Calib kernel cause 100+ unit errors; float32 vs
  // double accumulation should be well under these thresholds.
#ifdef CASPAR_USE_DOUBLE
  constexpr double kFocalTol = 1.0;
  constexpr double kPPTol = 1.0;
  constexpr double kExtraTol = 1e-4;
#else
  constexpr double kFocalTol = 20.0;
  constexpr double kPPTol = 10.0;
  constexpr double kExtraTol = 5e-3;
#endif

  const size_t f_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t cx_idx = SimpleRadialCameraModel::principal_point_idxs[0];
  const size_t cy_idx = SimpleRadialCameraModel::principal_point_idxs[1];
  const size_t k_idx = SimpleRadialCameraModel::extra_params_idxs[0];

  for (const auto& [cam_id, _] : reconstruction.Cameras()) {
    const auto& cam_ceres = reconstruction_ceres.Camera(cam_id);
    const auto& cam_caspar = reconstruction_caspar.Camera(cam_id);
    EXPECT_NEAR(cam_caspar.params[f_idx], cam_ceres.params[f_idx], kFocalTol)
        << "focal length mismatch for camera " << cam_id;
    EXPECT_NEAR(cam_caspar.params[cx_idx], cam_ceres.params[cx_idx], kPPTol)
        << "cx mismatch for camera " << cam_id;
    EXPECT_NEAR(cam_caspar.params[cy_idx], cam_ceres.params[cy_idx], kPPTol)
        << "cy mismatch for camera " << cam_id;
    EXPECT_NEAR(cam_caspar.params[k_idx], cam_ceres.params[k_idx], kExtraTol)
        << "radial distortion mismatch for camera " << cam_id;
  }
}

bool PoseExactlyUnchanged(const Image& a, const Image& b) {
  return a.CamFromWorld().rotation().coeffs() ==
             b.CamFromWorld().rotation().coeffs() &&
         a.CamFromWorld().translation() == b.CamFromWorld().translation();
}

TEST(DefaultBundleAdjuster, GaugeFixingWithOneFrameFromWorld) {
  SetPRNGSeed(0);
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 2;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  SynthesizeDataset(opts, &reconstruction);
  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.point3D_stddev = 0.1;
  noise_opts.rig_from_world_rotation_stddev = 0.3;
  noise_opts.rig_from_world_translation_stddev = 0.05;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  auto adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  ASSERT_NE(adjuster->Solve()->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // Exactly one of the two frames must be pinned by gauge fixing.
  const int n_fixed =
      static_cast<int>(PoseExactlyUnchanged(reconstruction.Image(1),
                                            orig_reconstruction.Image(1))) +
      static_cast<int>(PoseExactlyUnchanged(reconstruction.Image(2),
                                            orig_reconstruction.Image(2)));
  EXPECT_EQ(n_fixed, 1);
}

TEST(DefaultBundleAdjuster,
     GaugeFixingWithOneFrameFromWorld_SkipsWhenAlreadyFixed) {
  SetPRNGSeed(0);
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 2;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  SynthesizeDataset(opts, &reconstruction);
  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.point3D_stddev = 0.1;
  noise_opts.rig_from_world_rotation_stddev = 0.3;
  noise_opts.rig_from_world_translation_stddev = 0.05;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.SetConstantRigFromWorldPose(1);  // frame 1 explicitly constant
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  auto adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  ASSERT_NE(adjuster->Solve()->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // Frame 1 is explicitly constant — must be unchanged.
  CheckConstantCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));
  // Frame 2 is not gauge-fixed (gauge fixer saw frame 1 already fixed) — must
  // change.
  CheckVariableCamFromWorld(reconstruction.Image(2),
                            orig_reconstruction.Image(2));
}

TEST(DefaultBundleAdjuster, GaugeFixingWithThreePoints_PinsExactlyThreePoints) {
  SetPRNGSeed(0);
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 2;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  SynthesizeDataset(opts, &reconstruction);
  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.point3D_stddev = 0.1;
  noise_opts.rig_from_world_rotation_stddev = 0.3;
  noise_opts.rig_from_world_translation_stddev = 0.05;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

  BundleAdjustmentOptions options;
  auto adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  ASSERT_NE(adjuster->Solve()->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  int n_unchanged = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.xyz == orig_reconstruction.Point3D(point3D_id).xyz) {
      ++n_unchanged;
    }
  }
  EXPECT_EQ(n_unchanged, 3);

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckVariableCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
}

TEST(DefaultBundleAdjuster,
     GaugeFixingWithThreePoints_CountsExistingConstantPoints) {
  SetPRNGSeed(0);
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 2;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  SynthesizeDataset(opts, &reconstruction);
  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.point3D_stddev = 0.1;
  noise_opts.rig_from_world_rotation_stddev = 0.3;
  noise_opts.rig_from_world_translation_stddev = 0.05;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.AddConstantPoint(1);  // 1 existing constant; gauge fixer adds 2 more
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

  BundleAdjustmentOptions options;
  auto adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  ASSERT_NE(adjuster->Solve()->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // Total unchanged = 1 config-constant + 2 gauge-fixed = 3.
  int n_unchanged = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.xyz == orig_reconstruction.Point3D(point3D_id).xyz) {
      ++n_unchanged;
    }
  }
  EXPECT_EQ(n_unchanged, 3);

  CheckConstantPoint(reconstruction.Point3D(1), orig_reconstruction.Point3D(1));
}

}  // namespace
}  // namespace colmap

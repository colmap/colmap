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

// Helper to get Problem from BundleAdjuster (requires casting to Ceres impl)
inline ceres::Problem& GetCeresProblem(BundleAdjuster& bundle_adjuster) {
  auto* ceres_ba = dynamic_cast<CeresBundleAdjuster*>(&bundle_adjuster);
  CHECK_NOTNULL(ceres_ba);
  return *ceres_ba->Problem();
}

// Helper to get ceres::Solver::Summary from base summary
inline const ceres::Solver::Summary& GetCeresSummary(
    const BundleAdjustmentSummary* summary) {
  auto* ceres_summary =
      dynamic_cast<const CeresBundleAdjustmentSummary*>(summary);
  CHECK_NOTNULL(ceres_summary);
  return ceres_summary->ceres_summary;
}

TEST(DefaultBundleAdjuster, Nominal) {
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
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
}

TEST(DefaultBundleAdjuster, NominalMultiCameraRig) {
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
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
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
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            GetCeresProblem(*bundle_adjuster).NumResiduals());

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(GetCeresSummary(summary.get()).num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 5 rig_from_world parameters (pose of second image)
  // + 2 x 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            309);

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckConstantCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));

  CheckVariableCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
  CheckConstantCamFromWorldTranslationCoord(reconstruction.Image(2),
                                            orig_reconstruction.Image(2));

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    CheckVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
}

TEST(DefaultBundleAdjuster, ThreeViewSpherical) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 3;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_model_id =
      EquirectangularCameraModel::model_id;
  synthetic_dataset_options.camera_width = 1000;
  synthetic_dataset_options.camera_height = 500;
  synthetic_dataset_options.camera_params = {1000, 500};
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  ASSERT_TRUE(reconstruction.Camera(1).IsSpherical());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.AddImage(3);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            GetCeresProblem(*bundle_adjuster).NumResiduals());

  // The spherical model has no focal length; its (w, h) parameters are held
  // constant during bundle adjustment.
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    EXPECT_EQ(camera.params, orig_reconstruction.Camera(camera_id).params);
  }

  CheckConstantCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));
  CheckConstantCamFromWorldTranslationCoord(reconstruction.Image(2),
                                            orig_reconstruction.Image(2));
  CheckVariableCamFromWorld(reconstruction.Image(3),
                            orig_reconstruction.Image(3));
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    CheckVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
}

TEST(DefaultBundleAdjuster, TwoViewRig) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            GetCeresProblem(*bundle_adjuster).NumResiduals());

  // 100 points, 4 images, 2 residuals per point per image
  EXPECT_EQ(GetCeresSummary(summary.get()).num_residuals_reduced, 800);
  // 97 x 3 point parameters (3 fixed for gauge)
  // + 2 x 6 rig_from_world parameters
  // + 1 x 6 sensor_from_rig parameters
  // + 2 x 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            313);

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckVariableCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));

  CheckVariableCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
  CheckVariableCamFromWorld(reconstruction.Image(2),
                            orig_reconstruction.Image(2));

  size_t num_variable_points = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D != orig_reconstruction.Point3D(point3D_id)) {
      ++num_variable_points;
    }
  }
  EXPECT_EQ(num_variable_points, 97);
}

TEST(DefaultBundleAdjuster, ManyViewRig) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            GetCeresProblem(*bundle_adjuster).NumResiduals());

  // 100 points, 30 images, 2 residuals per point per image
  EXPECT_EQ(GetCeresSummary(summary.get()).num_residuals_reduced, 6000);
  // 97 x 3 point parameters (3 fixed for gauge)
  // + 10 x 6 rig_from_world parameters
  // + 4 x 6 sensor_from_rig parameters
  // + 6 x 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            387);

  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    CheckVariableCamera(camera, orig_reconstruction.Camera(camera_id));
  }

  for (const image_t image_id : reconstruction.RegImageIds()) {
    CheckVariableCamFromWorld(reconstruction.Image(image_id),
                              orig_reconstruction.Image(image_id));
  }

  size_t num_variable_points = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D != orig_reconstruction.Point3D(point3D_id)) {
      ++num_variable_points;
    }
  }
  EXPECT_EQ(num_variable_points, 97);
}

TEST(DefaultBundleAdjuster, ManyViewRigConstantSensorFromRig) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.SetConstantSensorFromRigPose(reconstruction.Camera(2).SensorId());
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            GetCeresProblem(*bundle_adjuster).NumResiduals());

  // 100 points, 30 images, 2 residuals per point per image
  EXPECT_EQ(GetCeresSummary(summary.get()).num_residuals_reduced, 6000);
  // 97 x 3 point parameters (3 fixed for gauge)
  // + 10 x 6 rig_from_world parameters
  // + 3 x 6 sensor_from_rig parameters
  // + 6 x 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            381);

  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    CheckVariableCamera(camera, orig_reconstruction.Camera(camera_id));
  }

  for (const image_t image_id : reconstruction.RegImageIds()) {
    CheckVariableCamFromWorld(reconstruction.Image(image_id),
                              orig_reconstruction.Image(image_id));
  }

  size_t num_variable_points = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D != orig_reconstruction.Point3D(point3D_id)) {
      ++num_variable_points;
    }
  }
  EXPECT_EQ(num_variable_points, 97);
}

TEST(DefaultBundleAdjuster, ManyViewRigConstantRigFromWorld) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  const frame_t constant_frame_id = 1;
  config.SetConstantRigFromWorldPose(constant_frame_id);
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            GetCeresProblem(*bundle_adjuster).NumResiduals());

  // 100 points, 30 images, 2 residuals per point per image
  EXPECT_EQ(GetCeresSummary(summary.get()).num_residuals_reduced, 6000);
  // 97 x 3 point parameters (3 fixed for gauge)
  // + 9 x 6 rig_from_world parameters
  // + 4 x 6 sensor_from_rig parameters
  // + 6 x 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            381);

  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    CheckVariableCamera(camera, orig_reconstruction.Camera(camera_id));
  }

  for (const image_t image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);
    if (image.FrameId() == constant_frame_id &&
        image.FramePtr()->RigPtr()->IsRefSensor(
            image.CameraPtr()->SensorId())) {
      CheckConstantCamFromWorld(image, orig_reconstruction.Image(image_id));
    } else {
      CheckVariableCamFromWorld(image, orig_reconstruction.Image(image_id));
    }
  }

  size_t num_variable_points = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D != orig_reconstruction.Point3D(point3D_id)) {
      ++num_variable_points;
    }
  }
  EXPECT_EQ(num_variable_points, 97);
}

TEST(DefaultBundleAdjuster, ConstantRigFromWorldRotation) {
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
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.AddImage(3);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.constant_rig_from_world_rotation = true;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            GetCeresProblem(*bundle_adjuster).NumResiduals());

  // 100 points, 3 images, 2 residuals per point per image
  EXPECT_EQ(GetCeresSummary(summary.get()).num_residuals_reduced, 600);
  // 100 x 3 point parameters
  // + 2 translation parameters (second image, one coord fixed for gauge)
  // + 3 translation parameters (third image)
  // + 3 x 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            311);

  // Check rotations are constant for all images
  for (const image_t image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);
    const auto& orig_image = orig_reconstruction.Image(image_id);
    // Rotation should be nearly unchanged (use angular distance)
    EXPECT_LE(image.CamFromWorld().rotation().angularDistance(
                  orig_image.CamFromWorld().rotation()),
              kConstantPoseVarEps);
  }

  // Check translations are variable (except for gauge-fixed parts)
  // At least one image should have changed translation
  bool has_variable_translation = false;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);
    const auto& orig_image = orig_reconstruction.Image(image_id);
    if ((image.CamFromWorld().translation() -
         orig_image.CamFromWorld().translation())
            .norm() > kConstantPoseVarEps) {
      has_variable_translation = true;
      break;
    }
  }
  EXPECT_TRUE(has_variable_translation);

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    CheckVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
}

TEST(DefaultBundleAdjuster, TwoViewConstantCamera) {
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
  config.SetConstantCamIntrinsics(1);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            GetCeresProblem(*bundle_adjuster).NumResiduals());

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(GetCeresSummary(summary.get()).num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            302);

  CheckConstantCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
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
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            GetCeresProblem(*bundle_adjuster).NumResiduals());

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(GetCeresSummary(summary.get()).num_residuals_reduced, 400);
  // 1 x 3 point parameters
  // 2 x 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced, 7);

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

TEST(DefaultBundleAdjuster, PartiallyContainedTracksForceToOptimizePoint) {
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

  const point3D_t variable_point3D_id =
      reconstruction.Image(3).Point2D(0).point3D_id;
  const point3D_t add_variable_point3D_id =
      reconstruction.Image(3).Point2D(1).point3D_id;
  const point3D_t add_constant_point3D_id =
      reconstruction.Image(3).Point2D(2).point3D_id;
  reconstruction.DeleteObservation(3, 0);

  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);
  config.AddVariablePoint(add_variable_point3D_id);
  config.AddConstantPoint(add_constant_point3D_id);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            GetCeresProblem(*bundle_adjuster).NumResiduals());

  // 100 points, 2 images, 2 residuals per point per image
  // + 2 residuals in 3rd image for added variable 3D point
  // (added constant point does not add residuals since the image/camera
  // is also constant).
  EXPECT_EQ(GetCeresSummary(summary.get()).num_residuals_reduced, 402);
  // 2 x 3 point parameters
  // 2 x 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            10);

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckConstantCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));

  CheckVariableCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
  CheckConstantCamFromWorld(reconstruction.Image(2),
                            orig_reconstruction.Image(2));

  CheckConstantCamera(reconstruction.Camera(3), orig_reconstruction.Camera(3));
  CheckConstantCamFromWorld(reconstruction.Image(3),
                            orig_reconstruction.Image(3));

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D_id == variable_point3D_id ||
        point3D_id == add_variable_point3D_id) {
      CheckVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
    } else {
      CheckConstantPoint(point3D, orig_reconstruction.Point3D(point3D_id));
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
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);
  config.AddConstantPoint(constant_point3D_id1);
  config.AddConstantPoint(constant_point3D_id2);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            GetCeresProblem(*bundle_adjuster).NumResiduals());

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(GetCeresSummary(summary.get()).num_residuals_reduced, 400);
  // 98 x 3 point parameters
  // + 2 x 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            298);

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
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            GetCeresProblem(*bundle_adjuster).NumResiduals());

  // 100 points, 3 images, 2 residuals per point per image
  EXPECT_EQ(GetCeresSummary(summary.get()).num_residuals_reduced, 600);
  // 100 x 3 point parameters
  // + 5 rig_from_world parameters (pose of second image)
  // + 6 rig_from_world parameters (pose of third image)
  // + 3 x 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            317);

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckConstantCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));

  CheckVariableCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
  CheckConstantCamFromWorldTranslationCoord(reconstruction.Image(2),
                                            orig_reconstruction.Image(2));

  CheckVariableCamera(reconstruction.Camera(3), orig_reconstruction.Camera(3));
  CheckVariableCamFromWorld(reconstruction.Image(3),
                            orig_reconstruction.Image(3));

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    CheckVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
}

TEST(DefaultBundleAdjuster, ConstantFocalLength) {
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
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.refine_focal_length = false;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            GetCeresProblem(*bundle_adjuster).NumResiduals());

  // 100 points, 3 images, 2 residuals per point per image
  EXPECT_EQ(GetCeresSummary(summary.get()).num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 5 rig_from_world parameters (pose of second image)
  // + 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            307);

  CheckConstantCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));
  CheckConstantCamFromWorldTranslationCoord(reconstruction.Image(2),
                                            orig_reconstruction.Image(2));

  const size_t focal_length_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t extra_param_idx = SimpleRadialCameraModel::extra_params_idxs[0];

  const auto& camera0 = reconstruction.Camera(1);
  const auto& orig_camera0 = orig_reconstruction.Camera(1);
  EXPECT_TRUE(camera0.params[focal_length_idx] ==
              orig_camera0.params[focal_length_idx]);
  EXPECT_TRUE(camera0.params[extra_param_idx] !=
              orig_camera0.params[extra_param_idx]);

  const auto& camera1 = reconstruction.Camera(2);
  const auto& orig_camera1 = orig_reconstruction.Camera(2);
  EXPECT_TRUE(camera1.params[focal_length_idx] ==
              orig_camera1.params[focal_length_idx]);
  EXPECT_TRUE(camera1.params[extra_param_idx] !=
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
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.refine_principal_point = true;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            GetCeresProblem(*bundle_adjuster).NumResiduals());

  // 100 points, 3 images, 2 residuals per point per image
  EXPECT_EQ(GetCeresSummary(summary.get()).num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 5 rig_from_world parameters (pose of second image)
  // + 8 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            313);

  CheckConstantCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));
  CheckConstantCamFromWorldTranslationCoord(reconstruction.Image(2),
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

TEST(DefaultBundleAdjuster, ConstantExtraParam) {
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
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.refine_extra_params = false;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            GetCeresProblem(*bundle_adjuster).NumResiduals());

  // 100 points, 3 images, 2 residuals per point per image
  EXPECT_EQ(GetCeresSummary(summary.get()).num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 5 rig_from_world parameters (pose of second image)
  // + 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            307);

  CheckConstantCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));
  CheckConstantCamFromWorldTranslationCoord(reconstruction.Image(2),
                                            orig_reconstruction.Image(2));

  const size_t focal_length_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t extra_param_idx = SimpleRadialCameraModel::extra_params_idxs[0];

  const auto& camera0 = reconstruction.Camera(1);
  const auto& orig_camera0 = orig_reconstruction.Camera(1);
  EXPECT_TRUE(camera0.params[focal_length_idx] !=
              orig_camera0.params[focal_length_idx]);
  EXPECT_TRUE(camera0.params[extra_param_idx] ==
              orig_camera0.params[extra_param_idx]);

  const auto& camera1 = reconstruction.Camera(2);
  const auto& orig_camera1 = orig_reconstruction.Camera(2);
  EXPECT_TRUE(camera1.params[focal_length_idx] !=
              orig_camera1.params[focal_length_idx]);
  EXPECT_TRUE(camera1.params[extra_param_idx] ==
              orig_camera1.params[extra_param_idx]);

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    CheckVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
}

TEST(DefaultBundleAdjuster, ConstantPoints3D) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 20;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);

  BundleAdjustmentOptions options;
  options.refine_points3D = false;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            GetCeresProblem(*bundle_adjuster).NumResiduals());

  // 20 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(GetCeresSummary(summary.get()).num_residuals_reduced, 80);
  // 0 point parameters (all constant due to refine_points3D=false)
  // + 2 x 6 rig_from_world parameters
  // + 2 x 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            16);

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckVariableCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));

  CheckVariableCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
  CheckVariableCamFromWorld(reconstruction.Image(2),
                            orig_reconstruction.Image(2));

  // All 3D points should remain constant.
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    CheckConstantPoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
}

TEST(DefaultBundleAdjuster, FixGaugeWithThreePoints) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);

  auto ExpectValidSolve = [&config, &reconstruction](
                              const int num_effective_parameters_reduced) {
    const auto summary1 = CreateDefaultCeresBundleAdjuster(
                              BundleAdjustmentOptions(), config, reconstruction)
                              ->Solve();
    ASSERT_NE(summary1->termination_type,
              BundleAdjustmentTerminationType::FAILURE);
    EXPECT_EQ(GetCeresSummary(summary1.get()).num_effective_parameters_reduced,
              num_effective_parameters_reduced);
  };

  ExpectValidSolve(316);

  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);
  ExpectValidSolve(307);

  config.AddConstantPoint(1);
  ExpectValidSolve(307);

  config.AddConstantPoint(2);
  config.AddConstantPoint(3);
  ExpectValidSolve(307);

  config.AddConstantPoint(4);
  ExpectValidSolve(304);
}

TEST(DefaultBundleAdjuster, FixGaugeWithTwoCamsFromWorld) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentOptions options;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.AddImage(3);
  config.AddImage(4);

  auto ExpectValidSolve = [&options, &config, &reconstruction](
                              const int num_effective_parameters_reduced) {
    const auto summary1 =
        CreateDefaultCeresBundleAdjuster(options, config, reconstruction)
            ->Solve();
    ASSERT_NE(summary1->termination_type,
              BundleAdjustmentTerminationType::FAILURE);
    EXPECT_EQ(GetCeresSummary(summary1.get()).num_effective_parameters_reduced,
              num_effective_parameters_reduced);
  };

  options.refine_rig_from_world = false;
  ExpectValidSolve(320);

  options.refine_rig_from_world = true;
  ExpectValidSolve(332);

  options.refine_rig_from_world = false;
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  ExpectValidSolve(320);

  options.refine_rig_from_world = true;
  ExpectValidSolve(325);

  config.SetConstantRigFromWorldPose(1);
  ExpectValidSolve(325);

  config.SetConstantRigFromWorldPose(2);
  ExpectValidSolve(320);
}

TEST(DefaultBundleAdjuster, FixGaugeWithTwoCamsFromWorldFixSensorFromRig) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentOptions options;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.AddImage(3);
  config.AddImage(4);

  auto ExpectValidSolve = [&options, &config, &reconstruction](
                              const int num_effective_parameters_reduced) {
    const auto summary1 =
        CreateDefaultCeresBundleAdjuster(options, config, reconstruction)
            ->Solve();
    ASSERT_NE(summary1->termination_type,
              BundleAdjustmentTerminationType::FAILURE);
    EXPECT_EQ(GetCeresSummary(summary1.get()).num_effective_parameters_reduced,
              num_effective_parameters_reduced);
  };

  options.refine_rig_from_world = false;
  options.refine_sensor_from_rig = false;
  ExpectValidSolve(308);

  options.refine_rig_from_world = true;
  ExpectValidSolve(320);

  options.refine_rig_from_world = false;
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  ExpectValidSolve(308);

  options.refine_rig_from_world = true;
  ExpectValidSolve(313);

  config.SetConstantRigFromWorldPose(1);
  ExpectValidSolve(313);

  config.SetConstantRigFromWorldPose(2);
  ExpectValidSolve(308);
}

TEST(DefaultBundleAdjuster, FixGaugeWithTwoCamsFromWorldNoReferenceSensor) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  // Delete observations from the two reference images.
  THROW_CHECK(reconstruction.Image(1).IsRefInFrame());
  THROW_CHECK(reconstruction.Image(3).IsRefInFrame());
  for (point2D_t i = 0; i < reconstruction.Image(1).NumPoints2D(); ++i) {
    if (reconstruction.Image(1).Point2D(i).HasPoint3D()) {
      reconstruction.DeleteObservation(1, i);
    }
  }
  for (point2D_t i = 0; i < reconstruction.Image(3).NumPoints2D(); ++i) {
    if (reconstruction.Image(3).Point2D(i).HasPoint3D()) {
      reconstruction.DeleteObservation(3, i);
    }
  }

  // Only add two non-reference images.
  BundleAdjustmentOptions options;
  BundleAdjustmentConfig config;
  config.AddImage(2);
  config.AddImage(4);

  auto ExpectValidSolve = [&options, &config, &reconstruction](
                              const int num_effective_parameters_reduced) {
    const auto summary1 =
        CreateDefaultCeresBundleAdjuster(options, config, reconstruction)
            ->Solve();
    THROW_CHECK_NE(summary1->termination_type,
                   BundleAdjustmentTerminationType::FAILURE);
    THROW_CHECK_EQ(
        GetCeresSummary(summary1.get()).num_effective_parameters_reduced,
        num_effective_parameters_reduced);
  };

  // refine_sensor_from_rig should have no effect when there are no reference
  // sensors
  options.refine_rig_from_world = true;
  options.refine_sensor_from_rig = true;
  ExpectValidSolve(316);

  options.refine_rig_from_world = false;
  ExpectValidSolve(304);

  options.refine_rig_from_world = true;
  ExpectValidSolve(316);

  options.refine_rig_from_world = false;
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  ExpectValidSolve(304);

  options.refine_sensor_from_rig = false;
  ExpectValidSolve(304);

  options.refine_rig_from_world = true;
  ExpectValidSolve(309);

  config.SetConstantRigFromWorldPose(1);
  ExpectValidSolve(309);
  options.refine_rig_from_world = false;
  ExpectValidSolve(304);

  config.SetConstantRigFromWorldPose(2);
  options.refine_rig_from_world = true;
  ExpectValidSolve(304);
  options.refine_rig_from_world = false;
  ExpectValidSolve(304);
}

TEST(DefaultBundleAdjuster, FixGaugeWithTwoCamsFromWorldFallback) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentOptions options;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);

  // The current implementation needs two reference cameras in different frames
  // to fix the gauge. If there are none, it falls back to fixing the gauge with
  // three points.
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  const auto summary =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction)
          ->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters, 316);
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            307);
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
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            GetCeresProblem(*bundle_adjuster).NumResiduals());

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(GetCeresSummary(summary.get()).num_residuals_reduced, 396);
  // 99 x 3 point parameters
  // + 5 rig_from_world parameters (pose of second image)
  // + 2 x 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            306);
}

TEST(PosePriorBundleAdjuster, LossScaleDefaultsAreSqrtOfChiSquareQuantiles) {
  // Regression test for the robust-loss unit-conversion fix: Ceres scales a
  // robust loss as rho(s, a) = a^2 * rho(s / a^2), where `a` is in
  // residual-norm ("number of standard deviations") units, so the 95%-
  // confidence-radius threshold for a whitened residual is
  // sqrt(chi-square_k_dof_95), not the raw chi-square quantile itself.
  // Regressing to the raw quantile here would silently make the robust
  // loss ~2.8x (3-DOF) / ~2.4x (2-DOF) too permissive.
  CeresPosePriorBundleAdjustmentOptions ceres_options;
  EXPECT_DOUBLE_EQ(ceres_options.prior_position_loss_scale,
                   std::sqrt(kChiSquare95ThreeDof));
  EXPECT_DOUBLE_EQ(ceres_options.prior_gravity_loss_scale,
                   std::sqrt(kChiSquare95TwoDof));
  // Sanity check against the literal quantile values, not just internal
  // self-consistency with std::sqrt.
  EXPECT_NEAR(ceres_options.prior_position_loss_scale, 2.7955, 1e-4);
  EXPECT_NEAR(ceres_options.prior_gravity_loss_scale, 2.4477, 1e-4);
}

TEST(PosePriorBundleAdjuster, AlignmentRobustToOutliers) {
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 7;
  synthetic_options.num_points3D = 50;
  synthetic_options.prior_position = true;
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction = gt_reconstruction;

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point3D_stddev = 0.2;
  synthetic_noise_options.rig_from_world_rotation_stddev = 1.0;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.2;
  synthetic_noise_options.prior_position_stddev = 0.05;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
  // Add 2 outlier priors with very large covariance
  pose_priors.at(0).position += Eigen::Vector3d::Constant(10);
  pose_priors.at(0).position_covariance = Eigen::Matrix3d::Identity() * 1e6;
  pose_priors.at(1).position += Eigen::Vector3d::Constant(1);
  pose_priors.at(1).position_covariance = Eigen::Matrix3d::Identity() * 1e2;

  PosePriorBundleAdjustmentOptions prior_ba_options;
  prior_ba_options.alignment_ransac_options.random_seed = 0;
  prior_ba_options.alignment_ransac_options.max_error = 0.0;

  BundleAdjustmentOptions ba_options;
  BundleAdjustmentConfig ba_config;

  for (const frame_t frame_id : reconstruction.RegFrameIds()) {
    const Frame& frame = reconstruction.Frame(frame_id);
    for (const data_t& data_id : frame.ImageIds()) {
      ba_config.AddImage(data_id.id);
    }
  }

  auto adjuster = CreatePosePriorBundleAdjuster(
      ba_options, prior_ba_options, ba_config, pose_priors, reconstruction);
  auto summary = adjuster->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(PosePriorBundleAdjuster, InsufficientPriorsUseTwoCameraGauge) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 50;
  synthetic_options.prior_position = true;
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
  pose_priors.resize(2);
  auto adjuster =
      CreatePosePriorBundleAdjuster(BundleAdjustmentOptions(),
                                    PosePriorBundleAdjustmentOptions(),
                                    config,
                                    std::move(pose_priors),
                                    reconstruction);

  EXPECT_EQ(adjuster->Config().FixedGauge(),
            BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  EXPECT_TRUE(adjuster->Solve()->IsSolutionUsable());
}

TEST(PosePriorBundleAdjuster, MissingPositionCov) {
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 7;
  synthetic_options.num_points3D = 100;
  synthetic_options.prior_position = true;
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction = gt_reconstruction;

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
  for (PosePrior& pose_prior : pose_priors) {
    EXPECT_FALSE(pose_prior.HasPositionCov());
  }

  PosePriorBundleAdjustmentOptions prior_ba_options;
  prior_ba_options.alignment_ransac_options.random_seed = 0;
  prior_ba_options.ceres->prior_position_loss_function_type =
      CeresBundleAdjustmentOptions::LossFunctionType::CAUCHY;

  BundleAdjustmentOptions ba_options;
  BundleAdjustmentConfig ba_config;

  for (const frame_t frame_id : reconstruction.RegFrameIds()) {
    const Frame& frame = reconstruction.Frame(frame_id);
    for (const data_t& data_id : frame.ImageIds()) {
      ba_config.AddImage(data_id.id);
    }
  }

  auto adjuster = CreatePosePriorBundleAdjuster(
      ba_options, prior_ba_options, ba_config, pose_priors, reconstruction);
  auto summary = adjuster->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(PosePriorBundleAdjuster, InvalidCovarianceFallsBackSafely) {
  // A declared-but-degenerate per-row covariance (zero, singular, or
  // non-symmetric) must not reach CovarianceWeightedCostFunctor's
  // cov.inverse().llt() whitening: that row should instead fall back to
  // prior_position_fallback_stddev, exactly as if HasPositionCov() were
  // false. Before this validation existed, a singular matrix here could
  // produce a non-PD "square root information" and corrupt the whole solve
  // with NaNs/Infs.
  SetPRNGSeed(0);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 7;
  synthetic_options.num_points3D = 50;
  synthetic_options.prior_position = true;
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction = gt_reconstruction;

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point3D_stddev = 0.2;
  synthetic_noise_options.rig_from_world_rotation_stddev = 1.0;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.2;
  synthetic_noise_options.prior_position_stddev = 0.05;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
  ASSERT_GE(pose_priors.size(), 3u);
  // Zero covariance: finite (passes HasPositionCov()), but singular.
  pose_priors.at(0).position_covariance = Eigen::Matrix3d::Zero();
  // Non-symmetric: finite, nonsingular diagonal, but not a valid covariance.
  pose_priors.at(1).position_covariance = Eigen::Matrix3d::Identity();
  pose_priors.at(1).position_covariance(0, 1) = 5.0;
  // Negative diagonal: finite, symmetric, but not positive definite.
  pose_priors.at(2).position_covariance = -Eigen::Matrix3d::Identity();

  PosePriorBundleAdjustmentOptions prior_ba_options;
  prior_ba_options.alignment_ransac_options.random_seed = 0;
  prior_ba_options.prior_position_fallback_stddev = 1.0;

  BundleAdjustmentOptions ba_options;
  BundleAdjustmentConfig ba_config;
  for (const frame_t frame_id : reconstruction.RegFrameIds()) {
    const Frame& frame = reconstruction.Frame(frame_id);
    for (const data_t& data_id : frame.ImageIds()) {
      ba_config.AddImage(data_id.id);
    }
  }

  auto adjuster = CreatePosePriorBundleAdjuster(
      ba_options, prior_ba_options, ba_config, pose_priors, reconstruction);
  auto summary = adjuster->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());

  for (const frame_t frame_id : reconstruction.RegFrameIds()) {
    EXPECT_TRUE(
        reconstruction.Frame(frame_id).RigFromWorld().params.allFinite());
  }
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));

  prior_ba_options.require_valid_position_covariance = true;
  EXPECT_ANY_THROW(CreatePosePriorBundleAdjuster(
      ba_options, prior_ba_options, ba_config, pose_priors, reconstruction));
}

TEST(PosePriorBundleAdjuster, OptimizationRobustToOutliers) {
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 7;
  synthetic_options.num_points3D = 100;
  synthetic_options.prior_position = true;
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction = gt_reconstruction;

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point3D_stddev = 0.2;
  synthetic_noise_options.rig_from_world_rotation_stddev = 1.0;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.2;
  synthetic_noise_options.prior_position_stddev = 0.05;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
  // Add 2 confident but wrong priors.
  pose_priors[0].position_covariance = Eigen::Matrix3d::Identity() * 0.01;
  pose_priors[0].position += Eigen::Vector3d::Constant(10);
  pose_priors[1].position_covariance = Eigen::Matrix3d::Identity() * 1.01;
  pose_priors[1].position += Eigen::Vector3d::Constant(10);

  PosePriorBundleAdjustmentOptions prior_ba_options;
  prior_ba_options.alignment_ransac_options.random_seed = 0;
  prior_ba_options.ceres->prior_position_loss_function_type =
      CeresBundleAdjustmentOptions::LossFunctionType::CAUCHY;

  BundleAdjustmentOptions ba_options;
  BundleAdjustmentConfig ba_config;

  for (const frame_t frame_id : reconstruction.RegFrameIds()) {
    const Frame& frame = reconstruction.Frame(frame_id);
    for (const data_t& data_id : frame.ImageIds()) {
      ba_config.AddImage(data_id.id);
    }
  }

  auto adjuster = CreatePosePriorBundleAdjuster(
      ba_options, prior_ba_options, ba_config, pose_priors, reconstruction);
  auto summary = adjuster->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

// Characterizes how much the gravity residual actually moves a joint solve,
// and pins that measurement so a future weighting change cannot pass silently.
//
// It does not assert that gravity improves roll/pitch, because measured here
// it does not: from a 6.3 degree initial tilt, bundle adjustment reaches the
// same accuracy with gravity on or off, and even a deliberately 20-degree-wrong
// gravity barely perturbs the result. Each camera contributes hundreds of
// reprojection residuals and one gravity residual weighted at a several-degree
// sigma, so the angular term is outvoted long before it can steer anything.
// A sensor accurate to 2-5 degrees also cannot sharpen an estimate that
// reprojection and position priors have already driven below a degree.
//
// That makes gravity insurance rather than a contributor in this
// configuration: correct (see AbsoluteGravityPriorCostFunctor's own tests),
// wired into every pose-prior BA stage, harmless, and currently not decisive.
// The numbers below are the evidence for that claim; if a weighting change
// makes gravity matter, this test is where it will show up first.
//
// Tilt is measured as the angle between each camera's predicted down direction
// and ground truth's -- exactly the roll/pitch error, blind to yaw.
TEST(PosePriorBundleAdjuster, GravityInfluenceOnRollAndPitchIsBounded) {
  SetPRNGSeed(0);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 10;
  // Sparse structure with noisy observations, so reprojection alone leaves
  // real orientation freedom. With dense well-spread points, bundle adjustment
  // recovers the tilt perfectly on its own and there is nothing for gravity to
  // contribute -- which is a statement about that fixture, not about gravity.
  synthetic_options.num_points3D = 12;
  synthetic_options.prior_position = true;
  synthetic_options.prior_gravity = true;
  synthetic_options.prior_gravity_in_world = Eigen::Vector3d(0, 0, -1);
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  Reconstruction tilted = gt_reconstruction;
  SyntheticNoiseOptions noise_options;
  noise_options.point3D_stddev = 1.5;
  noise_options.point2D_stddev = 2.0;
  noise_options.rig_from_world_translation_stddev = 0.3;
  noise_options.prior_position_stddev = 0.05;
  SynthesizeNoise(noise_options, &tilted);

  // SynthesizeNoise only rotates about the world vertical, which is yaw and is
  // exactly what this test's metric ignores. Tilt has to be injected directly:
  // rotate each frame about a horizontal world axis, which is the roll/pitch
  // error gravity is supposed to remove.
  for (const frame_t frame_id : tilted.RegFrameIds()) {
    Rigid3d& rig_from_world = tilted.Frame(frame_id).RigFromWorld();
    const double angle_deg = RandomGaussian<double>(0.0, 6.0);
    const Eigen::Vector3d axis =
        Eigen::Vector3d(RandomGaussian<double>(0.0, 1.0),
                        RandomGaussian<double>(0.0, 1.0),
                        0.0)
            .normalized();
    rig_from_world.rotation() *=
        Eigen::Quaterniond(Eigen::AngleAxisd(DegToRad(angle_deg), axis));
  }

  // The strict path requires a declared covariance on every prior, so state
  // the one the noise above was actually drawn from.
  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
  for (PosePrior& prior : pose_priors) {
    prior.position_covariance =
        Eigen::Matrix3d::Identity() * (noise_options.prior_position_stddev *
                                       noise_options.prior_position_stddev);
  }

  BundleAdjustmentConfig ba_config;
  for (const frame_t frame_id : tilted.RegFrameIds()) {
    for (const data_t& data_id : tilted.Frame(frame_id).ImageIds()) {
      ba_config.AddImage(data_id.id);
    }
  }

  // Mean angle between the solved and ground-truth down directions, in the
  // camera frame. Yaw-free by construction.
  const auto mean_tilt_error_deg =
      [&gt_reconstruction](const Reconstruction& solved) {
        const Eigen::Vector3d world_down(0.0, 0.0, -1.0);
        double sum = 0.0;
        int count = 0;
        for (const image_t image_id : solved.RegImageIds()) {
          const Eigen::Vector3d solved_down =
              solved.Image(image_id).CamFromWorld().rotation() * world_down;
          const Eigen::Vector3d gt_down =
              gt_reconstruction.Image(image_id).CamFromWorld().rotation() *
              world_down;
          sum += RadToDeg(std::acos(std::clamp(
              solved_down.normalized().dot(gt_down.normalized()), -1.0, 1.0)));
          ++count;
        }
        return count > 0 ? sum / count : 0.0;
      };

  const auto solve_with_gravity = [&](bool use_gravity) {
    Reconstruction reconstruction = tilted;
    PosePriorBundleAdjustmentOptions prior_ba_options;
    prior_ba_options.alignment_ransac_options.random_seed = 0;
    prior_ba_options.use_prior_gravity = use_gravity;
    prior_ba_options.prior_gravity_stddev_deg = 2.0;
    BundleAdjustmentOptions ba_options;
    auto adjuster = CreatePosePriorBundleAdjuster(
        ba_options, prior_ba_options, ba_config, pose_priors, reconstruction);
    auto summary = adjuster->Solve();
    EXPECT_TRUE(summary->IsSolutionUsable());
    return mean_tilt_error_deg(reconstruction);
  };

  const double tilt_before = mean_tilt_error_deg(tilted);
  const double tilt_without_gravity = solve_with_gravity(false);
  const double tilt_with_gravity = solve_with_gravity(true);

  // How far a deliberately wrong gravity can drag the solve. This is the
  // measurement that says "outvoted", not "ignored": the residual is present
  // and finite, it simply cannot overcome the reprojection terms.
  double tilt_with_wrong_gravity = 0.0;
  {
    std::vector<PosePrior> wrong = pose_priors;
    for (PosePrior& prior : wrong) {
      prior.gravity =
          Eigen::AngleAxisd(DegToRad(20.0), Eigen::Vector3d::UnitX()) *
          prior.gravity;
    }
    Reconstruction reconstruction = tilted;
    PosePriorBundleAdjustmentOptions prior_ba_options;
    prior_ba_options.alignment_ransac_options.random_seed = 0;
    prior_ba_options.use_prior_gravity = true;
    prior_ba_options.prior_gravity_stddev_deg = 2.0;
    BundleAdjustmentOptions ba_options;
    auto adjuster = CreatePosePriorBundleAdjuster(
        ba_options, prior_ba_options, ba_config, wrong, reconstruction);
    adjuster->Solve();
    tilt_with_wrong_gravity = mean_tilt_error_deg(reconstruction);
  }

  // Printed so the margins are visible in the log rather than only implied by
  // a green test -- these numbers are the finding.
  std::cout << "  mean camera tilt error: " << tilt_before
            << " deg before solving, " << tilt_without_gravity
            << " deg without gravity, " << tilt_with_gravity
            << " deg with gravity, " << tilt_with_wrong_gravity
            << " deg with 20 deg wrong gravity" << std::endl;

  // The fixture must actually be tilted, or none of the comparisons mean
  // anything.
  ASSERT_GT(tilt_before, 2.0) << "fixture is not tilted enough to be a test";
  // ... and bundle adjustment must actually recover from it, or this would be
  // measuring a failed solve rather than gravity's contribution.
  ASSERT_LT(tilt_without_gravity, 1.0) << "the solve itself did not converge";

  // Gravity must never make a well-constrained solve worse. This is the
  // property the production pipeline depends on.
  EXPECT_LE(tilt_with_gravity, tilt_without_gravity + 0.01)
      << "gravity degraded roll/pitch: " << tilt_with_gravity << " deg with, "
      << tilt_without_gravity << " deg without";

  // And its influence is bounded: even a 20-degree-wrong reading moves the
  // solution by well under a tenth of a degree. If a weighting change makes
  // gravity decisive, this bound breaks first and the comment above needs
  // rewriting rather than the number relaxing.
  EXPECT_LT(std::abs(tilt_with_wrong_gravity - tilt_without_gravity), 0.1)
      << "a 20 deg gravity error moved the solve by "
      << std::abs(tilt_with_wrong_gravity - tilt_without_gravity)
      << " deg; gravity is now decisive in this configuration and the "
         "characterization above is stale";
}

TEST(PosePriorBundleAdjuster, GravityPriorSolveUsableWithOneOutlier) {
  SetPRNGSeed(0);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 10;
  synthetic_options.num_points3D = 100;
  synthetic_options.prior_position = true;
  synthetic_options.prior_gravity = true;
  // Match AbsoluteGravityPriorCostFunctor's fixed ENU-down convention (see
  // PosePriorBundleAdjuster::world_down_ in bundle_adjustment_ceres.cc):
  // gravity residuals are only added once position priors have established
  // the metric/ENU-aligned gauge, so ground-truth gravity in this test must
  // be expressed in that same convention for the residual to be zero at
  // ground truth.
  synthetic_options.prior_gravity_in_world = Eigen::Vector3d(0, 0, -1);
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction = gt_reconstruction;

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point3D_stddev = 0.2;
  synthetic_noise_options.rig_from_world_rotation_stddev = 3.0;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.2;
  synthetic_noise_options.prior_position_stddev = 0.05;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
  // Corrupt one image's gravity reading with a large (90 deg) tilt error --
  // a plausible single-frame IMU/CORI glitch -- while leaving its position
  // prior untouched.
  ASSERT_FALSE(pose_priors.empty());
  pose_priors[0].gravity =
      Eigen::AngleAxisd(DegToRad(90.0), Eigen::Vector3d::UnitX()) *
      pose_priors[0].gravity;

  PosePriorBundleAdjustmentOptions prior_ba_options;
  prior_ba_options.alignment_ransac_options.random_seed = 0;
  prior_ba_options.use_prior_gravity = true;
  prior_ba_options.prior_gravity_stddev_deg = 2.0;
  // Default gravity loss is already CAUCHY; explicit here to document intent
  // and keep this test resilient to a future default change elsewhere.
  prior_ba_options.ceres->prior_gravity_loss_function_type =
      CeresBundleAdjustmentOptions::LossFunctionType::CAUCHY;

  BundleAdjustmentOptions ba_options;
  BundleAdjustmentConfig ba_config;
  for (const frame_t frame_id : reconstruction.RegFrameIds()) {
    const Frame& frame = reconstruction.Frame(frame_id);
    for (const data_t& data_id : frame.ImageIds()) {
      ba_config.AddImage(data_id.id);
    }
  }

  auto adjuster = CreatePosePriorBundleAdjuster(
      ba_options, prior_ba_options, ba_config, pose_priors, reconstruction);
  auto summary = adjuster->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());

  // This is an integration guard that the gravity-enabled solve remains
  // usable and accurate with one corrupted reading. It does not attempt to
  // isolate a numerical Cauchy-vs-trivial delta; the exact loss selection is
  // wired and reported by PosePriorBundleAdjuster, while robust-loss behavior
  // itself is covered by Ceres.
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.5,
                                 /*max_proj_center_error=*/0.2,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

}  // namespace
}  // namespace colmap

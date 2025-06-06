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

#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/math/random.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/projection.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/models.h"
#include "colmap/util/eigen_matchers.h"

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
      if (std::abs((image).CamFromWorld().translation(i) -           \
                   (orig_image).CamFromWorld().translation(i)) <     \
          kConstantPoseVarEps) {                                     \
        ++num_constant_coords;                                       \
      }                                                              \
    }                                                                \
    EXPECT_EQ(num_constant_coords, 1);                               \
  }

#define CheckVariablePoint(point, orig_point) \
  {                                           \
    EXPECT_NE((point).xyz, (orig_point).xyz); \
  }

#define CheckConstantPoint(point, orig_point) \
  {                                           \
    EXPECT_EQ((point).xyz, (orig_point).xyz); \
  }

namespace colmap {
namespace {

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
}

TEST(DefaultBundleAdjuster, TwoView) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.point2D_stddev = 1;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary.termination_type, ceres::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            bundle_adjuster->Problem()->NumResiduals());

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 5 rig_from_world parameters (pose of second image)
  // + 2 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 309);

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

TEST(DefaultBundleAdjuster, TwoViewRig) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.point2D_stddev = 1;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary.termination_type, ceres::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            bundle_adjuster->Problem()->NumResiduals());

  // 100 points, 4 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 800);
  // 97 x 3 point parameters (3 fixed for gauge)
  // + 2 x 6 rig_from_world parameters
  // + 1 x 6 sensor_from_rig parameters
  // + 2 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 313);

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
  synthetic_dataset_options.point2D_stddev = 1;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary.termination_type, ceres::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            bundle_adjuster->Problem()->NumResiduals());

  // 100 points, 30 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 6000);
  // 97 x 3 point parameters (3 fixed for gauge)
  // + 10 x 6 rig_from_world parameters
  // + 4 x 6 sensor_from_rig parameters
  // + 6 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 387);

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
  synthetic_dataset_options.point2D_stddev = 1;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.SetConstantSensorFromRigPose(reconstruction.Camera(2).SensorId());
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary.termination_type, ceres::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            bundle_adjuster->Problem()->NumResiduals());

  // 100 points, 30 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 6000);
  // 97 x 3 point parameters (3 fixed for gauge)
  // + 10 x 6 rig_from_world parameters
  // + 3 x 6 sensor_from_rig parameters
  // + 6 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 381);

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
  synthetic_dataset_options.point2D_stddev = 1;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
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
      CreateDefaultBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary.termination_type, ceres::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            bundle_adjuster->Problem()->NumResiduals());

  // 100 points, 30 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 6000);
  // 97 x 3 point parameters (3 fixed for gauge)
  // + 9 x 6 rig_from_world parameters
  // + 4 x 6 sensor_from_rig parameters
  // + 6 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 381);

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

TEST(DefaultBundleAdjuster, TwoViewConstantCamera) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.point2D_stddev = 1;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);
  config.SetConstantCamIntrinsics(1);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary.termination_type, ceres::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            bundle_adjuster->Problem()->NumResiduals());

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 302);

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
  synthetic_dataset_options.point2D_stddev = 1;
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
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
      CreateDefaultBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary.termination_type, ceres::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            bundle_adjuster->Problem()->NumResiduals());

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 400);
  // 1 x 3 point parameters
  // 2 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 7);

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
  synthetic_dataset_options.point2D_stddev = 1;
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

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
      CreateDefaultBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary.termination_type, ceres::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            bundle_adjuster->Problem()->NumResiduals());

  // 100 points, 2 images, 2 residuals per point per image
  // + 2 residuals in 3rd image for added variable 3D point
  // (added constant point does not add residuals since the image/camera
  // is also constant).
  EXPECT_EQ(summary.num_residuals_reduced, 402);
  // 2 x 3 point parameters
  // 2 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 10);

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
  synthetic_dataset_options.point2D_stddev = 1;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
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
      CreateDefaultBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary.termination_type, ceres::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            bundle_adjuster->Problem()->NumResiduals());

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 400);
  // 98 x 3 point parameters
  // + 2 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 298);

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
  synthetic_dataset_options.point2D_stddev = 1;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.AddImage(3);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary.termination_type, ceres::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            bundle_adjuster->Problem()->NumResiduals());

  // 100 points, 3 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 600);
  // 100 x 3 point parameters
  // + 5 rig_from_world parameters (pose of second image)
  // + 6 rig_from_world parameters (pose of third image)
  // + 3 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 317);

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
  synthetic_dataset_options.point2D_stddev = 1;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.refine_focal_length = false;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary.termination_type, ceres::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            bundle_adjuster->Problem()->NumResiduals());

  // 100 points, 3 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 5 rig_from_world parameters (pose of second image)
  // + 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 307);

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
  synthetic_dataset_options.point2D_stddev = 1;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.refine_principal_point = true;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary.termination_type, ceres::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            bundle_adjuster->Problem()->NumResiduals());

  // 100 points, 3 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 5 rig_from_world parameters (pose of second image)
  // + 8 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 313);

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
  synthetic_dataset_options.point2D_stddev = 1;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.refine_extra_params = false;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary.termination_type, ceres::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            bundle_adjuster->Problem()->NumResiduals());

  // 100 points, 3 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 5 rig_from_world parameters (pose of second image)
  // + 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 307);

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

}  // namespace
}  // namespace colmap

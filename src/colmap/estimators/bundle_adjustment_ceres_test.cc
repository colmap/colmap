// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/bundle_adjustment_ceres.h"

#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/models.h"
#include "colmap/util/cuda.h"
#include "colmap/util/testing.h"

#include <limits>

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

#ifdef COLMAP_CUDA_ENABLED
TEST(CeresBundleAdjustmentOptions, FallsBackToCpuWithoutCudaDevice) {
  if (GetNumCudaDevices() > 0) {
    GTEST_SKIP() << "CUDA GPU is available";
  }

  CeresBundleAdjustmentOptions options;
  options.use_gpu = true;
  options.min_num_images_gpu_solver = 0;

  const ceres::Solver::Options solver_options =
      options.CreateSolverOptions(BundleAdjustmentConfig(), ceres::Problem());
  EXPECT_EQ(solver_options.dense_linear_algebra_library_type,
            options.solver_options.dense_linear_algebra_library_type);
  EXPECT_EQ(solver_options.sparse_linear_algebra_library_type,
            options.solver_options.sparse_linear_algebra_library_type);
}
#endif  // COLMAP_CUDA_ENABLED

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

TEST(DefaultBundleAdjuster, Cancellation) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 20;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }

  int num_checks = 0;
  BundleAdjustmentOptions options;
  options.check_if_stopped = [&num_checks]() {
    ++num_checks;
    return true;
  };
  const auto summary =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction)
          ->Solve();

  EXPECT_EQ(num_checks, 1);
  EXPECT_EQ(summary->termination_type,
            BundleAdjustmentTerminationType::USER_SUCCESS);
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

  CheckVariableCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));
  CheckVariableCamFromWorld(reconstruction.Image(2),
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
  // 100 x 3 point parameters
  // + 2 x 6 rig_from_world parameters
  // + 1 x 6 sensor_from_rig parameters
  // + 2 x 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            322);

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckVariableCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));

  CheckVariableCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
  CheckVariableCamFromWorld(reconstruction.Image(2),
                            orig_reconstruction.Image(2));

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    CheckVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
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
  // 100 x 3 point parameters
  // + 10 x 6 rig_from_world parameters
  // + 4 x 6 sensor_from_rig parameters
  // + 6 x 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            396);

  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    CheckVariableCamera(camera, orig_reconstruction.Camera(camera_id));
  }

  for (const image_t image_id : reconstruction.RegImageIds()) {
    CheckVariableCamFromWorld(reconstruction.Image(image_id),
                              orig_reconstruction.Image(image_id));
  }

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    CheckVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
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
  // 100 x 3 point parameters
  // + 10 x 6 rig_from_world parameters
  // + 3 x 6 sensor_from_rig parameters
  // + 6 x 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            390);

  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    CheckVariableCamera(camera, orig_reconstruction.Camera(camera_id));
  }

  for (const image_t image_id : reconstruction.RegImageIds()) {
    CheckVariableCamFromWorld(reconstruction.Image(image_id),
                              orig_reconstruction.Image(image_id));
  }

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    CheckVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
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
  // 100 x 3 point parameters
  // + 9 x 6 rig_from_world parameters
  // + 4 x 6 sensor_from_rig parameters
  // + 6 x 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            390);

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

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    CheckVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
}

TEST(DefaultBundleAdjuster, NoReferenceSensorConstantSensorFromRig) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  // Delete observations from the two reference images.
  THROW_CHECK(reconstruction.Image(1).IsRefInFrame());
  THROW_CHECK(reconstruction.Image(3).IsRefInFrame());
  for (const image_t ref_image_id : {1, 3}) {
    for (point2D_t i = 0; i < reconstruction.Image(ref_image_id).NumPoints2D();
         ++i) {
      if (reconstruction.Image(ref_image_id).Point2D(i).HasPoint3D()) {
        reconstruction.DeleteObservation(ref_image_id, i);
      }
    }
  }

  // Only add two non-reference images.
  BundleAdjustmentOptions options;
  BundleAdjustmentConfig config;
  config.AddImage(2);
  config.AddImage(4);

  auto ExpectValidSolve = [&options, &config, &reconstruction](
                              const int num_effective_parameters_reduced) {
    const auto summary =
        CreateDefaultCeresBundleAdjuster(options, config, reconstruction)
            ->Solve();
    ASSERT_NE(summary->termination_type,
              BundleAdjustmentTerminationType::FAILURE);
    EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
              num_effective_parameters_reduced);
  };

  // Without reference sensors, sensor_from_rig is held constant even when
  // refine_sensor_from_rig is true: 100 x 3 points + 2 x 6 rig_from_world +
  // 2 x 2 camera params = 316.
  options.refine_rig_from_world = true;
  options.refine_sensor_from_rig = true;
  ExpectValidSolve(316);

  options.refine_rig_from_world = false;
  ExpectValidSolve(304);

  options.refine_rig_from_world = true;
  options.refine_sensor_from_rig = false;
  ExpectValidSolve(316);

  config.SetConstantRigFromWorldPose(1);
  ExpectValidSolve(310);

  config.SetConstantRigFromWorldPose(2);
  ExpectValidSolve(304);
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
  // + 3 x 3 translation parameters
  // + 3 x 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            315);

  // Check rotations are constant and translations are variable for all images.
  for (const image_t image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);
    const auto& orig_image = orig_reconstruction.Image(image_id);
    EXPECT_LE(image.CamFromWorld().rotation().angularDistance(
                  orig_image.CamFromWorld().rotation()),
              kConstantPoseVarEps);
    EXPECT_GT((image.CamFromWorld().translation() -
               orig_image.CamFromWorld().translation())
                  .norm(),
              kConstantPoseVarEps);
  }

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    CheckVariablePoint(point3D, orig_reconstruction.Point3D(point3D_id));
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

  BundleAdjustmentOptions options;
  options.refine_focal_length = false;
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
  // + 2 x 6 rig_from_world parameters
  // + 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            314);

  CheckVariableCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));
  CheckVariableCamFromWorld(reconstruction.Image(2),
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

  BundleAdjustmentOptions options;
  options.refine_extra_params = false;
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
  // + 2 x 6 rig_from_world parameters
  // + 2 camera parameters
  EXPECT_EQ(GetCeresSummary(summary.get()).num_effective_parameters_reduced,
            314);

  CheckVariableCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));
  CheckVariableCamFromWorld(reconstruction.Image(2),
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

TEST(DefaultBundleAdjuster, FreeGaugeMatchesConstantPose) {
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 200;
  SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction);

  Reconstruction noisy_reconstruction = gt_reconstruction;
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.point3D_stddev = 0.1;
  synthetic_noise_options.rig_from_world_rotation_stddev = 0.5;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &noisy_reconstruction);

  const auto solve = [](bool fix_first_pose, Reconstruction& reconstruction) {
    BundleAdjustmentConfig config;
    for (const image_t image_id : reconstruction.RegImageIds()) {
      config.AddImage(image_id);
    }
    if (fix_first_pose) {
      config.SetConstantRigFromWorldPose(1);
    }
    const auto summary = CreateDefaultCeresBundleAdjuster(
                             BundleAdjustmentOptions(), config, reconstruction)
                             ->Solve();
    EXPECT_EQ(summary->termination_type,
              BundleAdjustmentTerminationType::CONVERGENCE);
    return GetCeresSummary(summary.get()).final_cost;
  };

  Reconstruction fixed_reconstruction = noisy_reconstruction;
  const double fixed_cost =
      solve(/*fix_first_pose=*/true, fixed_reconstruction);
  Reconstruction free_reconstruction = noisy_reconstruction;
  const double free_cost = solve(/*fix_first_pose=*/false, free_reconstruction);

  // The gauge does not change the optimum, only its representative.
  EXPECT_NEAR(free_cost, fixed_cost, 1e-6 * fixed_cost);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(free_reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
}

TEST(DefaultBundleAdjuster, FreeGaugeDampingFloorKeepsStepsValid) {
  SetPRNGSeed(42);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction);

  // Place cameras along a narrow baseline at distance 5 from the point cloud
  // (b / Z ~ 4e-4) and reproject exact 2D observations.
  constexpr double kBaseline = 2e-3;
  for (const frame_t frame_id : gt_reconstruction.RegFrameIds()) {
    const double x = kBaseline * (static_cast<int>(frame_id) - 2);
    const Eigen::Vector3d proj_center(x, 0.0, -5.0);
    gt_reconstruction.Frame(frame_id).SetRigFromWorld(
        Rigid3d(Eigen::Quaterniond::Identity(), -proj_center));
  }
  for (const image_t image_id : gt_reconstruction.RegImageIds()) {
    Image& image = gt_reconstruction.Image(image_id);
    const Camera& camera = *image.CameraPtr();
    for (Point2D& point2D : image.Points2D()) {
      if (point2D.HasPoint3D()) {
        point2D.xy = *camera.ImgFromCam(
            image.CamFromWorld() *
            gt_reconstruction.Point3D(point2D.point3D_id).xyz);
      }
    }
  }

  // Perturb camera orientations with a slight inward vergence bias and keep
  // rotations fixed to exercise an ill-conditioned free-gauge translation and
  // point update.
  Reconstruction noisy_reconstruction = gt_reconstruction;
  for (const frame_t frame_id : noisy_reconstruction.RegFrameIds()) {
    const double x = kBaseline * (static_cast<int>(frame_id) - 2);
    const double yaw = -0.195 * x;
    Rigid3d& rig_from_world =
        noisy_reconstruction.Frame(frame_id).RigFromWorld();
    rig_from_world.rotation() =
        Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitY())) *
        rig_from_world.rotation();
  }

  BundleAdjustmentConfig config;
  for (const image_t image_id : noisy_reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }

  auto count_invalid_steps = [&](const double max_trust_region_radius) {
    Reconstruction reconstruction = noisy_reconstruction;
    BundleAdjustmentOptions options;
    options.refine_focal_length = false;
    options.refine_extra_params = false;
    options.constant_rig_from_world_rotation = true;
    options.ceres->loss_function_type = CeresLossFunctionType::HUBER;
    options.ceres->solver_options.max_trust_region_radius =
        max_trust_region_radius;
    options.ceres->solver_options.max_num_iterations = 50;
    const auto summary =
        CreateDefaultCeresBundleAdjuster(options, config, reconstruction)
            ->Solve();
    EXPECT_TRUE(summary->IsSolutionUsable());
    if (!summary->IsSolutionUsable()) {
      return -1;
    }
    const auto& ceres_summary = GetCeresSummary(summary.get());

    int num_invalid_steps = 0;
    for (const auto& iteration : ceres_summary.iterations) {
      if (!iteration.step_is_valid) {
        ++num_invalid_steps;
      }
    }
    return num_invalid_steps;
  };

  // Without capping max_trust_region_radius (Ceres's 1e16 default), the LM
  // damping 1 / tr_radius vanishes below machine precision, causing repeated
  // Cholesky factorization failures on the rank-deficient free-gauge system.
  EXPECT_GT(count_invalid_steps(1e16), 0);

  // With the default 1e8 cap (damping floor 1e-8), all steps remain valid.
  EXPECT_EQ(
      count_invalid_steps(BundleAdjustmentOptions()
                              .ceres->solver_options.max_trust_region_radius),
      0);
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

TEST(PosePriorBundleAdjuster, InsufficientPriorsIgnored) {
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 5;
  synthetic_options.num_points3D = 50;
  synthetic_options.prior_position = true;
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  Reconstruction noisy_reconstruction = gt_reconstruction;
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.point3D_stddev = 0.1;
  synthetic_noise_options.rig_from_world_rotation_stddev = 0.5;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &noisy_reconstruction);

  BundleAdjustmentConfig config;
  for (const image_t image_id : noisy_reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }

  Reconstruction plain_reconstruction = noisy_reconstruction;
  auto plain_adjuster = CreateDefaultCeresBundleAdjuster(
      BundleAdjustmentOptions(), config, plain_reconstruction);
  const int expected_num_residuals =
      plain_adjuster->Problem()->NumResidualBlocks();
  ASSERT_TRUE(plain_adjuster->Solve()->IsSolutionUsable());

  // Case 1: fewer than 3 priors (< 3 required for Sim3 alignment).
  std::vector<PosePrior> two_priors = database->ReadAllPosePriors();
  two_priors.resize(2);

  // Case 2: >= 3 priors whose Sim3 alignment fails (coincident prior
  // positions).
  std::vector<PosePrior> unalignable_priors = database->ReadAllPosePriors();
  for (PosePrior& prior : unalignable_priors) {
    prior.position = Eigen::Vector3d::Zero();
  }

  for (const auto& priors : {two_priors, unalignable_priors}) {
    Reconstruction reconstruction = noisy_reconstruction;
    auto adjuster =
        CreatePosePriorCeresBundleAdjuster(BundleAdjustmentOptions(),
                                           PosePriorBundleAdjustmentOptions(),
                                           config,
                                           priors,
                                           reconstruction);

    EXPECT_EQ(adjuster->Problem()->NumResidualBlocks(), expected_num_residuals);
    ASSERT_TRUE(adjuster->Solve()->IsSolutionUsable());
    EXPECT_THAT(plain_reconstruction,
                ReconstructionNear(reconstruction,
                                   /*max_rotation_error_deg=*/1e-5,
                                   /*max_proj_center_error=*/1e-5,
                                   /*max_scale_error=*/1e-5,
                                   /*num_obs_tolerance=*/0.0));
    EXPECT_THAT(gt_reconstruction,
                ReconstructionNear(reconstruction,
                                   /*max_rotation_error_deg=*/0.1,
                                   /*max_proj_center_error=*/0.1,
                                   /*max_scale_error=*/std::nullopt,
                                   /*num_obs_tolerance=*/0.0));
  }
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
      CeresLossFunctionType::CAUCHY;

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

TEST(PosePriorBundleAdjuster, ConstantSensorFromRigWithMissingPositionCov) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 2;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 50;
  synthetic_options.prior_position = true;
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
  for (const PosePrior& pose_prior : pose_priors) {
    EXPECT_FALSE(pose_prior.HasPositionCov());
  }

  BundleAdjustmentOptions ba_options;
  ba_options.refine_sensor_from_rig = false;
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    ba_config.AddImage(image_id);
  }

  auto adjuster =
      CreatePosePriorBundleAdjuster(ba_options,
                                    PosePriorBundleAdjustmentOptions(),
                                    ba_config,
                                    std::move(pose_priors),
                                    reconstruction);
  ceres::Problem& problem = GetCeresProblem(*adjuster);

  for (auto& [_, rig] : reconstruction.Rigs()) {
    for (auto& [_, sensor_from_rig] : rig.NonRefSensors()) {
      ASSERT_TRUE(sensor_from_rig.has_value());
      ASSERT_TRUE(problem.HasParameterBlock(sensor_from_rig->params.data()));
      EXPECT_TRUE(
          problem.IsParameterBlockConstant(sensor_from_rig->params.data()));
    }
  }
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
      CeresLossFunctionType::CAUCHY;

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

TEST(CeresPosePriorBundleAdjustmentOptions, CheckGemanMcClure) {
  CeresPosePriorBundleAdjustmentOptions options;
  options.prior_position_loss_function_type =
      CeresLossFunctionType::GEMAN_MCCLURE;
  options.prior_position_loss_scale = 1.0;
  EXPECT_TRUE(options.Check());
  for (const double bad_scale :
       {0.0, 1e-200, 1e200, std::numeric_limits<double>::infinity()}) {
    options.prior_position_loss_scale = bad_scale;
    EXPECT_FALSE(options.Check());
  }
}

TEST(PosePriorBundleAdjuster, UnobservedImagePriorsExcludedFromAlignment) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 4;
  synthetic_options.num_points3D = 50;
  synthetic_options.prior_position = true;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  // Strip all 3D point observations from two of the four images so only two
  // parameterized images remain in the BA problem.
  const std::vector<image_t> image_ids = reconstruction.RegImageIds();
  for (int i = 0; i < 2; ++i) {
    Image& image = reconstruction.Image(image_ids[i]);
    for (point2D_t j = 0; j < image.NumPoints2D(); ++j) {
      if (image.Point2D(j).HasPoint3D()) {
        reconstruction.DeleteObservation(image_ids[i], j);
      }
    }
  }

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }

  Reconstruction plain_reconstruction = reconstruction;
  auto plain_adjuster = CreateDefaultCeresBundleAdjuster(
      BundleAdjustmentOptions(), config, plain_reconstruction);
  const int expected_num_residuals =
      plain_adjuster->Problem()->NumResidualBlocks();

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
  const Sim3d shifted_from_orig(
      3.0, Eigen::Quaterniond::Identity(), Eigen::Vector3d(10.0, 20.0, 30.0));
  for (PosePrior& prior : pose_priors) {
    prior.position = shifted_from_orig * prior.position;
  }

  PosePriorBundleAdjustmentOptions prior_options;
  prior_options.alignment_ransac_options.random_seed = 0;
  const Reconstruction orig_reconstruction = reconstruction;
  auto adjuster = CreatePosePriorCeresBundleAdjuster(BundleAdjustmentOptions(),
                                                     prior_options,
                                                     config,
                                                     std::move(pose_priors),
                                                     reconstruction);
  EXPECT_EQ(adjuster->Problem()->NumResidualBlocks(), expected_num_residuals);
  EXPECT_TRUE(adjuster->Solve()->IsSolutionUsable());
  EXPECT_LT((reconstruction.Image(image_ids[2]).ProjectionCenter() -
             orig_reconstruction.Image(image_ids[2]).ProjectionCenter())
                .norm(),
            1.0);
}

TEST(PosePriorBundleAdjuster, ConstantFramePosePreserved) {
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 7;
  synthetic_options.num_points3D = 60;
  synthetic_options.prior_position = true;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  const std::vector<frame_t> reg_frame_ids = gt_reconstruction.RegFrameIds();
  ASSERT_GE(reg_frame_ids.size(), 7);
  std::vector<point3D_t> constant_point_ids;
  for (const auto& [point3D_id, _] : gt_reconstruction.Points3D()) {
    constant_point_ids.push_back(point3D_id);
    if (constant_point_ids.size() == 5) {
      break;
    }
  }
  ASSERT_EQ(constant_point_ids.size(), 5);

  const Sim3d kAlignedShift(
      1.0, Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.02, -0.01, 0.03));
  struct TestCase {
    const char* name;
    Sim3d prior_from_world;
    size_t num_priors;
    std::vector<size_t> constant_frame_indices;
    size_t num_constant_points;
    bool expected_priors_used;
    double max_rotation_error_deg;
    double max_proj_center_error;
    bool refine_points3D = true;
    bool noisy_fallback_priors = false;
    bool two_coincident_priors = false;
    bool constant_rotation = false;
    bool omit_first_frame_from_config = false;
  };
  const std::vector<TestCase> cases = {
      // Aligned within covariance: >= 3 priors, < 3 priors, and 1 constant
      // point + 2 priors all keep priors active without moving anchors.
      {"aligned_5_priors_const_frame",
       kAlignedShift,
       5,
       {0},
       0,
       true,
       0.5,
       0.08},
      {"aligned_2_priors_const_frame", Sim3d(), 2, {0}, 0, true, 0.5, 0.08},
      {"aligned_2_priors_1_const_point", Sim3d(), 2, {}, 1, true, 0.5, 0.08},
      // Fallback covariance (sigma = 1 m -> max_error = 2.80 m) with noisy
      // priors where < 50% (3/7 = 43%) are inliers, matching
      // EstimateSim3dRobust's 3-sample floor rather than requiring >= 50%.
      {"aligned_fallback_cov_sub_median_inliers",
       kAlignedShift,
       7,
       {0, 1},
       0,
       true,
       0.5,
       0.1,
       /*refine_points3D=*/true,
       /*noisy_fallback_priors=*/true},
      // Aligned with refine_points3D = false and 1 prior keeps priors active.
      {"aligned_1_prior_unrefined_points3D",
       kAlignedShift,
       1,
       {},
       0,
       true,
       0.5,
       0.08,
       /*refine_points3D=*/false},
      // Unaligned scale + rotation + translation mismatch skips priors.
      {"unaligned_sim3_const_frames",
       Sim3d(
           2.5,
           Eigen::Quaterniond(Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ())),
           Eigen::Vector3d(100.0, -50.0, 20.0)),
       5,
       {0, 2},
       0,
       false,
       0.1,
       0.05},
      // Unaligned unit-scale rotation + translation mismatch (no scale error).
      {"unaligned_rigid_const_frames",
       Sim3d(
           1.0,
           Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY())),
           Eigen::Vector3d(10.0, -5.0, 2.0)),
       5,
       {0, 2},
       0,
       false,
       0.1,
       0.05},
      // Unaligned with 5 priors where 2 of 5 coincide (< min(3, 5) inliers)
      // must still reject the priors.
      {"unaligned_two_coincident_priors",
       Sim3d(1.0,
             Eigen::Quaterniond::Identity(),
             Eigen::Vector3d(10.0, 0.0, 0.0)),
       5,
       {0, 2},
       0,
       false,
       0.1,
       0.05,
       /*refine_points3D=*/true,
       /*noisy_fallback_priors=*/false,
       /*two_coincident_priors=*/true},
      // Unaligned with < 3 priors (1 constant + 1 variable) and pure shift.
      {"unaligned_2_priors_const_frames",
       Sim3d(1.0,
             Eigen::Quaterniond::Identity(),
             Eigen::Vector3d(10.0, 0.0, 0.0)),
       2,
       {0, 2},
       0,
       false,
       0.1,
       0.05},
      // Unaligned anchored by constant 3D points skips priors without moving
      // constant points.
      {"unaligned_const_points",
       Sim3d(1.0,
             Eigen::Quaterniond::Identity(),
             Eigen::Vector3d(10.0, 0.0, 0.0)),
       5,
       {},
       5,
       false,
       0.1,
       0.05},
      // Unaligned with refine_points3D = false preserves reference 3D points
      // without Sim3d-moving them.
      {"unaligned_unrefined_points3D",
       Sim3d(
           2.5,
           Eigen::Quaterniond(Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ())),
           Eigen::Vector3d(100.0, -50.0, 20.0)),
       5,
       {},
       0,
       false,
       0.1,
       0.05,
       /*refine_points3D=*/false},
      // Constant frame rotations anchor world orientation: aligned priors
      // refine translations while unaligned rotation priors are skipped.
      {"aligned_const_rotation",
       kAlignedShift,
       5,
       {},
       0,
       true,
       0.01,
       0.08,
       /*refine_points3D=*/true,
       /*noisy_fallback_priors=*/false,
       /*two_coincident_priors=*/false,
       /*constant_rotation=*/true},
      {"unaligned_const_rotation",
       Sim3d(
           1.0,
           Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ())),
           Eigen::Vector3d(10.0, -5.0, 2.0)),
       5,
       {},
       0,
       false,
       0.01,
       0.05,
       /*refine_points3D=*/true,
       /*noisy_fallback_priors=*/false,
       /*two_coincident_priors=*/false,
       /*constant_rotation=*/true},
      // Subset BA (an omitted registered frame outside config) anchors the
      // world frame: aligned priors are used while unaligned priors are skipped
      // without Sim3d-transforming omitted frames or shared 3D points.
      {"aligned_subset_ba_omitted_frame",
       kAlignedShift,
       5,
       {},
       0,
       true,
       0.5,
       0.08,
       /*refine_points3D=*/true,
       /*noisy_fallback_priors=*/false,
       /*two_coincident_priors=*/false,
       /*constant_rotation=*/false,
       /*omit_first_frame_from_config=*/true},
      {"unaligned_subset_ba_omitted_frame",
       Sim3d(
           2.5,
           Eigen::Quaterniond(Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ())),
           Eigen::Vector3d(100.0, -50.0, 20.0)),
       5,
       {},
       0,
       false,
       0.1,
       0.05,
       /*refine_points3D=*/true,
       /*noisy_fallback_priors=*/false,
       /*two_coincident_priors=*/false,
       /*constant_rotation=*/false,
       /*omit_first_frame_from_config=*/true},
  };

  for (const TestCase& tc : cases) {
    SCOPED_TRACE(tc.name);
    Reconstruction reconstruction = gt_reconstruction;
    SyntheticNoiseOptions noise_options;
    noise_options.point3D_stddev = tc.refine_points3D ? 0.05 : 0.0;
    noise_options.rig_from_world_rotation_stddev =
        tc.constant_rotation ? 0.0 : 0.5;
    noise_options.rig_from_world_translation_stddev = 0.05;
    SynthesizeNoise(noise_options, &reconstruction);

    BundleAdjustmentConfig config;
    for (const image_t image_id : reconstruction.RegImageIds()) {
      config.AddImage(image_id);
    }
    if (tc.omit_first_frame_from_config) {
      reconstruction.Frame(reg_frame_ids[0])
          .SetRigFromWorld(
              gt_reconstruction.Frame(reg_frame_ids[0]).RigFromWorld());
      for (const data_t& data_id :
           reconstruction.Frame(reg_frame_ids[0]).ImageIds()) {
        config.RemoveImage(data_id.id);
        for (const Point2D& point2D :
             reconstruction.Image(data_id.id).Points2D()) {
          if (point2D.HasPoint3D()) {
            reconstruction.Point3D(point2D.point3D_id).xyz =
                gt_reconstruction.Point3D(point2D.point3D_id).xyz;
          }
        }
      }
    }
    for (const size_t idx : tc.constant_frame_indices) {
      reconstruction.Frame(reg_frame_ids[idx])
          .SetRigFromWorld(
              gt_reconstruction.Frame(reg_frame_ids[idx]).RigFromWorld());
      config.SetConstantRigFromWorldPose(reg_frame_ids[idx]);
    }
    for (size_t i = 0; i < tc.num_constant_points; ++i) {
      const point3D_t point3D_id = constant_point_ids[i];
      reconstruction.Point3D(point3D_id).xyz =
          gt_reconstruction.Point3D(point3D_id).xyz;
      config.AddConstantPoint(point3D_id);
    }
    const Reconstruction orig_reconstruction = reconstruction;

    std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
    pose_priors.resize(tc.num_priors);
    for (size_t i = 0; i < pose_priors.size(); ++i) {
      PosePrior& prior = pose_priors[i];
      prior.position = tc.prior_from_world * prior.position;
      if (tc.two_coincident_priors && (i == 1 || i == 3)) {
        prior.position =
            reconstruction.Image(prior.corr_data_id.id).ProjectionCenter();
      }
      if (tc.noisy_fallback_priors) {
        prior.position_covariance =
            Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());
        // Perturb 4 of 7 priors beyond sqrt(chi2_95_3dof) * 1.0 = 2.80 m
        // (simulating ~2 m per-axis GPS noise) so 3/7 (43%) are inliers.
        if (i >= 3) {
          const double sign = (i % 2 == 0) ? 1.0 : -1.0;
          prior.position += sign * Eigen::Vector3d(2.0, -2.0, 1.5);
        }
      } else {
        prior.position_covariance = 1e-2 * Eigen::Matrix3d::Identity();
      }
    }

    BundleAdjustmentOptions ba_options;
    ba_options.refine_points3D = tc.refine_points3D;
    ba_options.constant_rig_from_world_rotation = tc.constant_rotation;
    Reconstruction plain_reconstruction = reconstruction;
    const int plain_num_residuals =
        CreateDefaultCeresBundleAdjuster(
            ba_options, config, plain_reconstruction)
            ->Problem()
            ->NumResidualBlocks();

    PosePriorBundleAdjustmentOptions prior_options;
    if (tc.noisy_fallback_priors) {
      prior_options.ceres->prior_position_loss_function_type =
          CeresLossFunctionType::CAUCHY;
    }
    auto adjuster = CreatePosePriorCeresBundleAdjuster(ba_options,
                                                       prior_options,
                                                       config,
                                                       std::move(pose_priors),
                                                       reconstruction);
    if (tc.expected_priors_used) {
      EXPECT_GT(adjuster->Problem()->NumResidualBlocks(), plain_num_residuals);
    } else {
      EXPECT_EQ(adjuster->Problem()->NumResidualBlocks(), plain_num_residuals);
    }
    ASSERT_TRUE(adjuster->Solve()->IsSolutionUsable());

    if (tc.constant_rotation) {
      for (const frame_t frame_id : reg_frame_ids) {
        EXPECT_LT(reconstruction.Frame(frame_id)
                      .RigFromWorld()
                      .rotation()
                      .angularDistance(orig_reconstruction.Frame(frame_id)
                                           .RigFromWorld()
                                           .rotation()),
                  kConstantPoseVarEps);
      }
    }
    if (tc.omit_first_frame_from_config) {
      const image_t omitted_image_id =
          reconstruction.Frame(reg_frame_ids[0]).ImageIds().begin()->id;
      CheckConstantCamFromWorld(reconstruction.Image(omitted_image_id),
                                orig_reconstruction.Image(omitted_image_id));
      if (!tc.expected_priors_used) {
        EXPECT_EQ(
            reconstruction.Frame(reg_frame_ids[0]).RigFromWorld().params,
            orig_reconstruction.Frame(reg_frame_ids[0]).RigFromWorld().params);
      }
    }
    for (const size_t idx : tc.constant_frame_indices) {
      const image_t image_id =
          reconstruction.Frame(reg_frame_ids[idx]).ImageIds().begin()->id;
      CheckConstantCamFromWorld(reconstruction.Image(image_id),
                                orig_reconstruction.Image(image_id));
    }
    auto check_constant_point = [&](const point3D_t point3D_id) {
      if (tc.expected_priors_used) {
        EXPECT_LT((reconstruction.Point3D(point3D_id).xyz -
                   orig_reconstruction.Point3D(point3D_id).xyz)
                      .norm(),
                  kConstantPoseVarEps);
      } else {
        EXPECT_EQ(reconstruction.Point3D(point3D_id).xyz,
                  orig_reconstruction.Point3D(point3D_id).xyz);
      }
    };
    if (!tc.refine_points3D) {
      for (const auto& [point3D_id, _] : orig_reconstruction.Points3D()) {
        check_constant_point(point3D_id);
      }
    } else {
      for (size_t i = 0; i < tc.num_constant_points; ++i) {
        check_constant_point(constant_point_ids[i]);
      }
    }
    EXPECT_THAT(gt_reconstruction,
                ReconstructionNear(reconstruction,
                                   tc.max_rotation_error_deg,
                                   tc.max_proj_center_error,
                                   /*max_scale_error=*/std::nullopt,
                                   /*num_obs_tolerance=*/0.02,
                                   /*align=*/false));
  }

  // Duplicate camera priors must be rejected even on the anchored path that
  // bypasses AlignReconstructionToPosePriors.
  BundleAdjustmentConfig anchored_config;
  for (const image_t image_id : gt_reconstruction.RegImageIds()) {
    anchored_config.AddImage(image_id);
  }
  anchored_config.SetConstantRigFromWorldPose(reg_frame_ids[0]);
  std::vector<PosePrior> dup_priors = database->ReadAllPosePriors();
  ASSERT_GE(dup_priors.size(), 2);
  dup_priors.push_back(dup_priors.back());
  EXPECT_THAT(
      [&]() {
        Reconstruction rec = gt_reconstruction;
        CreatePosePriorCeresBundleAdjuster(BundleAdjustmentOptions(),
                                           PosePriorBundleAdjustmentOptions(),
                                           anchored_config,
                                           dup_priors,
                                           rec);
      },
      testing::ThrowsMessage<std::invalid_argument>(
          testing::HasSubstr("Duplicate pose prior")));
}

}  // namespace
}  // namespace colmap

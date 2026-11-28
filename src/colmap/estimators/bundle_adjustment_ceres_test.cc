// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/bundle_adjustment_ceres.h"

#include "colmap/geometry/gps.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/geometry/sim3_matchers.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/models.h"
#include "colmap/util/cuda.h"
#include "colmap/util/testing.h"

#include <limits>

#include <gmock/gmock.h>
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

struct PosePriorScene {
  Reconstruction gt_reconstruction;
  BundleAdjustmentPriors priors;
  BundleAdjustmentConfig config;
};

PosePriorScene SynthesizePosePriorScene(int num_cameras_per_rig,
                                        int num_frames_per_rig,
                                        int num_points3D) {
  PosePriorScene scene;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = num_cameras_per_rig;
  options.num_frames_per_rig = num_frames_per_rig;
  options.num_points3D = num_points3D;
  options.prior_position = true;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  SynthesizeDataset(options, &scene.gt_reconstruction, database.get());
  scene.priors.pose_priors = database->ReadAllPosePriors();
  for (const image_t image_id : scene.gt_reconstruction.RegImageIds()) {
    scene.config.AddImage(image_id);
  }
  return scene;
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

TEST(DefaultBundleAdjuster, PosePriorAlignmentRobustToOutliers) {
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

  BundleAdjustmentPriors priors;
  priors.pose_priors = database->ReadAllPosePriors();
  // Add 2 outlier priors with very large covariance
  priors.pose_priors.at(0).position += Eigen::Vector3d::Constant(10);
  priors.pose_priors.at(0).position_covariance =
      Eigen::Matrix3d::Identity() * 1e6;
  priors.pose_priors.at(1).position += Eigen::Vector3d::Constant(1);
  priors.pose_priors.at(1).position_covariance =
      Eigen::Matrix3d::Identity() * 1e2;

  BundleAdjustmentOptions ba_options;
  ba_options.pose_prior.use_position = true;
  ba_options.pose_prior.position_alignment_ransac.random_seed = 0;
  ba_options.pose_prior.position_alignment_ransac.max_error = 0.0;

  BundleAdjustmentConfig ba_config;
  for (const frame_t frame_id : reconstruction.RegFrameIds()) {
    const Frame& frame = reconstruction.Frame(frame_id);
    for (const data_t& data_id : frame.ImageIds()) {
      ba_config.AddImage(data_id.id);
    }
  }

  auto adjuster = CreateDefaultBundleAdjuster(
      ba_options, ba_config, reconstruction, priors);
  auto summary = adjuster->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());
  EXPECT_TRUE(reconstruction.HasMetricScale());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(DefaultBundleAdjuster, PosePriorInsufficientPriorsIgnored) {
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
  BundleAdjustmentPriors two_priors;
  two_priors.pose_priors = database->ReadAllPosePriors();
  two_priors.pose_priors.resize(2);

  // Case 2: >= 3 priors whose Sim3 alignment fails (coincident prior
  // positions).
  BundleAdjustmentPriors unalignable_priors;
  unalignable_priors.pose_priors = database->ReadAllPosePriors();
  for (PosePrior& prior : unalignable_priors.pose_priors) {
    prior.position = Eigen::Vector3d::Zero();
  }

  BundleAdjustmentOptions ba_options;
  ba_options.pose_prior.use_position = true;
  for (const auto& priors : {two_priors, unalignable_priors}) {
    Reconstruction reconstruction = noisy_reconstruction;
    auto adjuster = CreateDefaultCeresBundleAdjuster(
        ba_options, config, reconstruction, priors);

    EXPECT_FALSE(adjuster->NormalizedFromMetric().has_value());
    EXPECT_FALSE(reconstruction.HasMetricScale());
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

TEST(DefaultBundleAdjuster, PosePriorMissingPositionCov) {
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

  BundleAdjustmentPriors priors;
  priors.pose_priors = database->ReadAllPosePriors();
  for (PosePrior& pose_prior : priors.pose_priors) {
    EXPECT_FALSE(pose_prior.HasPositionCov());
  }

  BundleAdjustmentOptions ba_options;
  ba_options.pose_prior.use_position = true;
  ba_options.pose_prior.position_alignment_ransac.random_seed = 0;
  ba_options.ceres->prior_position_loss_function_type =
      CeresLossFunctionType::CAUCHY;

  BundleAdjustmentConfig ba_config;
  for (const frame_t frame_id : reconstruction.RegFrameIds()) {
    const Frame& frame = reconstruction.Frame(frame_id);
    for (const data_t& data_id : frame.ImageIds()) {
      ba_config.AddImage(data_id.id);
    }
  }

  auto adjuster = CreateDefaultBundleAdjuster(
      ba_options, ba_config, reconstruction, priors);
  auto summary = adjuster->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(DefaultBundleAdjuster, PosePriorConstantSensorFromRigWithMissingCov) {
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

  BundleAdjustmentPriors priors;
  priors.pose_priors = database->ReadAllPosePriors();
  for (const PosePrior& pose_prior : priors.pose_priors) {
    EXPECT_FALSE(pose_prior.HasPositionCov());
  }

  BundleAdjustmentOptions ba_options;
  ba_options.pose_prior.use_position = true;
  ba_options.refine_sensor_from_rig = false;
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    ba_config.AddImage(image_id);
  }

  auto adjuster = CreateDefaultCeresBundleAdjuster(
      ba_options, ba_config, reconstruction, priors);
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

TEST(DefaultBundleAdjuster, PosePriorOptimizationRobustToOutliers) {
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

  BundleAdjustmentPriors priors;
  priors.pose_priors = database->ReadAllPosePriors();
  // Add 2 confident but wrong priors.
  priors.pose_priors[0].position_covariance =
      Eigen::Matrix3d::Identity() * 0.01;
  priors.pose_priors[0].position += Eigen::Vector3d::Constant(10);
  priors.pose_priors[1].position_covariance =
      Eigen::Matrix3d::Identity() * 1.01;
  priors.pose_priors[1].position += Eigen::Vector3d::Constant(10);

  BundleAdjustmentOptions ba_options;
  ba_options.pose_prior.use_position = true;
  ba_options.pose_prior.position_alignment_ransac.random_seed = 0;
  ba_options.ceres->prior_position_loss_function_type =
      CeresLossFunctionType::CAUCHY;

  BundleAdjustmentConfig ba_config;
  for (const frame_t frame_id : reconstruction.RegFrameIds()) {
    const Frame& frame = reconstruction.Frame(frame_id);
    for (const data_t& data_id : frame.ImageIds()) {
      ba_config.AddImage(data_id.id);
    }
  }

  auto adjuster = CreateDefaultBundleAdjuster(
      ba_options, ba_config, reconstruction, priors);
  auto summary = adjuster->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(CeresBundleAdjustmentOptions, CheckPriorPositionGemanMcClure) {
  const auto check = [](auto options) {
    options.prior_position_loss_function_type =
        CeresLossFunctionType::GEMAN_MCCLURE;
    options.prior_position_loss_scale = 1.0;
    EXPECT_TRUE(options.Check());
    for (const double bad_scale :
         {0.0, 1e-200, 1e200, std::numeric_limits<double>::infinity()}) {
      options.prior_position_loss_scale = bad_scale;
      EXPECT_FALSE(options.Check());
    }
  };
  check(CeresBundleAdjustmentOptions());
  check(CeresPosePriorBundleAdjustmentOptions());
}

TEST(DefaultBundleAdjuster, PosePriorOnRigCameraWithoutVisualObservations) {
  PosePriorScene scene = SynthesizePosePriorScene(/*num_cameras_per_rig=*/2,
                                                  /*num_frames_per_rig=*/5,
                                                  /*num_points3D=*/60);

  // Strip all 3D point observations from the non-reference camera while
  // keeping position priors only on that camera, verifying that priors on a
  // camera with zero visual tracks are still retained and SE(3)-parameterized.
  FlatHashSet<image_t> non_ref_image_ids;
  for (const image_t image_id : scene.gt_reconstruction.RegImageIds()) {
    Image& image = scene.gt_reconstruction.Image(image_id);
    if (!image.IsRefInFrame()) {
      non_ref_image_ids.insert(image_id);
      for (point2D_t j = 0; j < image.NumPoints2D(); ++j) {
        if (image.Point2D(j).HasPoint3D()) {
          scene.gt_reconstruction.DeleteObservation(image_id, j);
        }
      }
    }
  }
  ASSERT_EQ(non_ref_image_ids.size(), 5);
  Reconstruction reconstruction = scene.gt_reconstruction;
  SyntheticNoiseOptions noise_options;
  noise_options.point3D_stddev = 0.01;
  noise_options.rig_from_world_rotation_stddev = 0.1;
  noise_options.rig_from_world_translation_stddev = 0.005;
  SynthesizeNoise(noise_options, &reconstruction);

  BundleAdjustmentPriors priors;
  for (PosePrior& prior : scene.priors.pose_priors) {
    if (non_ref_image_ids.contains(prior.corr_data_id.id)) {
      prior.position_covariance = 1e-4 * Eigen::Matrix3d::Identity();
      priors.pose_priors.push_back(prior);
    }
  }
  ASSERT_EQ(priors.pose_priors.size(), 5);

  reconstruction.Transform(Sim3d(
      2.5,
      Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ())),
      Eigen::Vector3d(10.0, -5.0, 2.0)));

  BundleAdjustmentOptions ba_options;
  ba_options.refine_sensor_from_rig = false;
  Reconstruction plain_reconstruction = reconstruction;
  const int plain_num_residuals =
      CreateDefaultCeresBundleAdjuster(
          ba_options, scene.config, plain_reconstruction)
          ->Problem()
          ->NumResidualBlocks();

  ba_options.pose_prior.use_position = true;
  ba_options.pose_prior.position_alignment_ransac.random_seed = 0;
  auto adjuster = CreateDefaultCeresBundleAdjuster(
      ba_options, scene.config, reconstruction, priors);
  EXPECT_EQ(adjuster->Problem()->NumResidualBlocks(), plain_num_residuals + 5);
  ASSERT_TRUE(adjuster->Solve()->IsSolutionUsable());
  EXPECT_THAT(scene.gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.05,
                                 /*max_proj_center_error=*/0.05,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02,
                                 /*align=*/false));
}

TEST(DefaultBundleAdjuster, PosePriorMixedConstantPoses) {
  PosePriorScene scene = SynthesizePosePriorScene(/*num_cameras_per_rig=*/2,
                                                  /*num_frames_per_rig=*/6,
                                                  /*num_points3D=*/60);

  const std::vector<frame_t> reg_frame_ids =
      scene.gt_reconstruction.RegFrameIds();
  ASSERT_EQ(reg_frame_ids.size(), 6);

  for (PosePrior& prior : scene.priors.pose_priors) {
    prior.position_covariance = 1e-2 * Eigen::Matrix3d::Identity();
  }

  // With a mix of constant and variable frames (e.g. fix_existing_frames),
  // constant frames participate in Sim3d alignment while keeping their relative
  // pose rigid, and prior residuals are added only on the variable frames.
  Reconstruction reconstruction = scene.gt_reconstruction;
  SyntheticNoiseOptions noise_options;
  noise_options.point3D_stddev = 0.05;
  noise_options.rig_from_world_rotation_stddev = 0.3;
  noise_options.rig_from_world_translation_stddev = 0.02;
  SynthesizeNoise(noise_options, &reconstruction);
  for (const size_t idx : {0, 1}) {
    reconstruction.Frame(reg_frame_ids[idx])
        .SetRigFromWorld(
            scene.gt_reconstruction.Frame(reg_frame_ids[idx]).RigFromWorld());
  }
  reconstruction.Transform(Sim3d(
      2.5,
      Eigen::Quaterniond(Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ())),
      Eigen::Vector3d(10.0, -5.0, 2.0)));

  BundleAdjustmentConfig mixed_config = scene.config;
  mixed_config.SetConstantRigFromWorldPose(reg_frame_ids[0]);
  mixed_config.SetConstantRigFromWorldPose(reg_frame_ids[1]);

  BundleAdjustmentOptions ba_options;
  ba_options.refine_sensor_from_rig = false;
  Reconstruction plain_reconstruction = reconstruction;
  const int plain_num_residuals =
      CreateDefaultCeresBundleAdjuster(
          ba_options, mixed_config, plain_reconstruction)
          ->Problem()
          ->NumResidualBlocks();

  ba_options.pose_prior.use_position = true;
  ba_options.pose_prior.position_alignment_ransac.random_seed = 0;
  auto adjuster = CreateDefaultCeresBundleAdjuster(
      ba_options, mixed_config, reconstruction, scene.priors);
  EXPECT_TRUE(adjuster->NormalizedFromMetric().has_value());
  EXPECT_EQ(adjuster->Problem()->NumResidualBlocks(), plain_num_residuals + 8);
  ASSERT_TRUE(adjuster->Solve()->IsSolutionUsable());
  EXPECT_THAT(scene.gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.5,
                                 /*max_proj_center_error=*/0.08,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02,
                                 /*align=*/false));

  // Duplicate camera priors in the problem are rejected loudly.
  BundleAdjustmentPriors dup_priors = scene.priors;
  dup_priors.pose_priors.push_back(dup_priors.pose_priors.back());
  BundleAdjustmentOptions dup_ba_options;
  dup_ba_options.pose_prior.use_position = true;
  EXPECT_THAT(
      [&]() {
        Reconstruction rec = scene.gt_reconstruction;
        CreateDefaultCeresBundleAdjuster(
            dup_ba_options, scene.config, rec, dup_priors);
      },
      testing::ThrowsMessage<std::invalid_argument>(
          testing::HasSubstr("Duplicate pose prior")));
}

TEST(DefaultBundleAdjuster, PosePriorOverwriteCovarianceAndAnisotropicStddev) {
  PosePriorScene scene = SynthesizePosePriorScene(/*num_cameras_per_rig=*/1,
                                                  /*num_frames_per_rig=*/6,
                                                  /*num_points3D=*/80);

  // Shift ground-truth along Z while providing noisy Z priors and accurate XY
  // priors. Overwriting the bogus database covariances with anisotropic stddev
  // (tight XY, loose Z) should align XY tightly.
  const Eigen::Matrix3d bogus_cov = 1e8 * Eigen::Matrix3d::Identity();
  for (size_t i = 0; i < scene.priors.pose_priors.size(); ++i) {
    scene.priors.pose_priors[i].position.z() += (i % 2 == 0 ? 5.0 : -5.0);
    scene.priors.pose_priors[i].position_covariance = bogus_cov;
  }

  Reconstruction reconstruction = scene.gt_reconstruction;
  SyntheticNoiseOptions noise_options;
  noise_options.point3D_stddev = 0.01;
  noise_options.rig_from_world_rotation_stddev = 0.1;
  noise_options.rig_from_world_translation_stddev = 0.005;
  SynthesizeNoise(noise_options, &reconstruction);

  BundleAdjustmentOptions ba_options;
  ba_options.pose_prior.use_position = true;
  ba_options.pose_prior.overwrite_position_covariance = true;
  ba_options.pose_prior.position_stddev = Eigen::Vector3d(0.02, 0.02, 100.0);
  ba_options.pose_prior.position_alignment_ransac.random_seed = 0;

  auto adjuster = CreateDefaultBundleAdjuster(
      ba_options, scene.config, reconstruction, scene.priors);
  ASSERT_TRUE(adjuster->Solve()->IsSolutionUsable());

  for (const image_t image_id : scene.gt_reconstruction.RegImageIds()) {
    const Eigen::Vector3d gt_center =
        scene.gt_reconstruction.Image(image_id).ProjectionCenter();
    const Eigen::Vector3d est_center =
        reconstruction.Image(image_id).ProjectionCenter();
    EXPECT_LT((gt_center.head<2>() - est_center.head<2>()).norm(), 0.05);
  }
}

TEST(DefaultBundleAdjuster, PosePriorWGS84ConvertedToENU) {
  PosePriorScene scene = SynthesizePosePriorScene(/*num_cameras_per_rig=*/1,
                                                  /*num_frames_per_rig=*/6,
                                                  /*num_points3D=*/80);

  ASSERT_FALSE(scene.priors.pose_priors.empty());
  const Eigen::Vector3d origin_shift = -scene.priors.pose_priors[0].position;
  scene.gt_reconstruction.Transform(
      Sim3d(1.0, Eigen::Quaterniond::Identity(), origin_shift));

  std::vector<Eigen::Vector3d> enu_positions;
  enu_positions.reserve(scene.priors.pose_priors.size());
  for (const PosePrior& prior : scene.priors.pose_priors) {
    enu_positions.push_back(prior.position + origin_shift);
  }

  constexpr double kRefLat = 47.3769;
  constexpr double kRefLon = 8.5417;
  constexpr double kRefAlt = 408.0;
  const GPSTransform gps_transform(GPSTransform::Ellipsoid::WGS84);
  const std::vector<Eigen::Vector3d> wgs84_positions =
      gps_transform.ENUToEllipsoid(enu_positions, kRefLat, kRefLon, kRefAlt);

  for (size_t i = 0; i < scene.priors.pose_priors.size(); ++i) {
    scene.priors.pose_priors[i].position = wgs84_positions[i];
    scene.priors.pose_priors[i].coordinate_system =
        PosePrior::CoordinateSystem::WGS84;
    scene.priors.pose_priors[i].position_covariance =
        1e-2 * Eigen::Matrix3d::Identity();
  }

  Reconstruction reconstruction = scene.gt_reconstruction;
  // Perturb input reconstruction into an arbitrary unaligned frame.
  reconstruction.Transform(Sim3d(
      0.25,
      Eigen::Quaterniond(Eigen::AngleAxisd(0.6, Eigen::Vector3d::UnitZ())),
      Eigen::Vector3d(-10.0, 20.0, 5.0)));

  BundleAdjustmentOptions ba_options;
  ba_options.pose_prior.use_position = true;
  ba_options.pose_prior.position_alignment_ransac.random_seed = 0;
  auto adjuster = CreateDefaultBundleAdjuster(
      ba_options, scene.config, reconstruction, scene.priors);
  ASSERT_TRUE(adjuster->Solve()->IsSolutionUsable());

  EXPECT_THAT(scene.gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.05,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02,
                                 /*align=*/false));
}

TEST(DefaultBundleAdjuster, DeprecatedPosePriorBundleAdjusterShim) {
  PosePriorScene scene = SynthesizePosePriorScene(/*num_cameras_per_rig=*/1,
                                                  /*num_frames_per_rig=*/5,
                                                  /*num_points3D=*/60);

  Reconstruction reconstruction = scene.gt_reconstruction;
  SyntheticNoiseOptions noise_options;
  noise_options.point3D_stddev = 0.01;
  noise_options.rig_from_world_rotation_stddev = 0.1;
  noise_options.rig_from_world_translation_stddev = 0.005;
  SynthesizeNoise(noise_options, &reconstruction);

  PosePriorBundleAdjustmentOptions prior_options;
  prior_options.prior_position_fallback_stddev = 2.5;
  prior_options.alignment_ransac_options.max_error = 3.0;
  prior_options.alignment_ransac_options.random_seed = 0;
  prior_options.ceres->prior_position_loss_function_type =
      CeresLossFunctionType::CAUCHY;
  prior_options.ceres->prior_position_loss_scale = 1.5;

  auto adjuster =
      CreatePosePriorBundleAdjuster(BundleAdjustmentOptions(),
                                    prior_options,
                                    scene.config,
                                    std::move(scene.priors.pose_priors),
                                    reconstruction);
  const BundleAdjustmentOptions& forwarded = adjuster->Options();
  EXPECT_TRUE(forwarded.pose_prior.use_position);
  EXPECT_EQ(forwarded.pose_prior.position_stddev,
            Eigen::Vector3d::Constant(2.5));
  EXPECT_DOUBLE_EQ(forwarded.pose_prior.position_alignment_ransac.max_error,
                   3.0);
  ASSERT_TRUE(forwarded.ceres);
  EXPECT_EQ(forwarded.ceres->prior_position_loss_function_type,
            CeresLossFunctionType::CAUCHY);
  EXPECT_DOUBLE_EQ(forwarded.ceres->prior_position_loss_scale, 1.5);
  ASSERT_TRUE(adjuster->Solve()->IsSolutionUsable());
}

TEST(DefaultBundleAdjuster,
     MetricNormalizationPreservesMetricFrameOutsideSolve) {
  PosePriorScene scene = SynthesizePosePriorScene(/*num_cameras_per_rig=*/1,
                                                  /*num_frames_per_rig=*/5,
                                                  /*num_points3D=*/60);

  // Offset and scale ground-truth metric scene away from the origin so leaving
  // the reconstruction centered would fail non-aligned metric comparisons.
  const Sim3d metric_from_synth(50.0,
                                Eigen::Quaterniond::Identity(),
                                Eigen::Vector3d(100.0, -200.0, 30.0));
  scene.gt_reconstruction.Transform(metric_from_synth);

  for (PosePrior& prior : scene.priors.pose_priors) {
    prior.position = metric_from_synth * prior.position;
    prior.position_covariance =
        (metric_from_synth.scale() * metric_from_synth.scale() * 1e-4) *
        Eigen::Matrix3d::Identity();
  }

  BundleAdjustmentOptions ba_options;
  ba_options.pose_prior.use_position = true;
  ba_options.pose_prior.position_stddev =
      Eigen::Vector3d::Constant(metric_from_synth.scale() * 1e-2);
  ba_options.refine_focal_length = false;
  ba_options.refine_extra_params = false;
  ba_options.pose_prior.position_alignment_ransac.random_seed = 0;

  Reconstruction reconstruction = scene.gt_reconstruction;
  SyntheticNoiseOptions noise_options;
  noise_options.point3D_stddev = 0.005 * metric_from_synth.scale();
  noise_options.rig_from_world_rotation_stddev = 0.3;
  noise_options.rig_from_world_translation_stddev =
      0.005 * metric_from_synth.scale();
  SynthesizeNoise(noise_options, &reconstruction);
  reconstruction.Transform(Sim3d(
      0.2,
      Eigen::Quaterniond(Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ())),
      Eigen::Vector3d(-10.0, 20.0, 5.0)));

  auto adjuster = CreateDefaultCeresBundleAdjuster(
      ba_options, scene.config, reconstruction, scene.priors);
  // Immediately after construction (before Solve()), `reconstruction` is
  // already in the aligned metric frame rather than the internal normalized
  // frame, and NormalizedFromMetric() exposes the centering transform.
  EXPECT_THAT(scene.gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/10.0,
                                 /*max_proj_center_error=*/50.0,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0,
                                 /*align=*/false));
  const Eigen::Vector3d centroid = reconstruction.ComputeCentroid(
      /*min_percentile=*/0.1, /*max_percentile=*/0.9, /*use_images=*/true);
  ASSERT_GT(centroid.norm(), 50.0);
  ASSERT_TRUE(adjuster->NormalizedFromMetric().has_value());
  EXPECT_THAT(*adjuster->NormalizedFromMetric(),
              Sim3dNear(Sim3d(1.0, Eigen::Quaterniond::Identity(), -centroid),
                        1e-6,
                        1e-6,
                        1e-6));

  ASSERT_TRUE(adjuster->Solve()->IsSolutionUsable());
  const Reconstruction after_first_solve = reconstruction;

  // Calling Solve() multiple times re-normalizes and restores cleanly without
  // drifting out of the metric frame.
  ASSERT_TRUE(adjuster->Solve()->IsSolutionUsable());
  EXPECT_THAT(after_first_solve,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/1e-3,
                                 /*max_proj_center_error=*/1e-3,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0,
                                 /*align=*/false));
  EXPECT_THAT(scene.gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-2,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0,
                                 /*align=*/false));

  // Non-SPD covariances fail loudly while RestoreMetricGuard restores the
  // metric frame during stack unwinding.
  BundleAdjustmentPriors bad_priors = scene.priors;
  bad_priors.pose_priors.front().position_covariance = Eigen::Matrix3d::Zero();
  EXPECT_THAT(
      [&]() {
        CreateDefaultBundleAdjuster(
            ba_options, scene.config, reconstruction, bad_priors);
      },
      testing::ThrowsMessage<std::invalid_argument>(testing::HasSubstr(
          "Covariance matrix is not symmetric positive definite")));
  EXPECT_THAT(scene.gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-2,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0,
                                 /*align=*/false));
}

}  // namespace
}  // namespace colmap

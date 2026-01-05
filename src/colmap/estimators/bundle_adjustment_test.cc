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
#ifdef CASPAR_ENABLED
#include "colmap/estimators/caspar_bundle_adjustment.h"
#endif
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

  config.IgnorePoint(3);
  EXPECT_EQ(config.NumResiduals(reconstruction), 792);
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

#ifdef CASPAR_ENABLED

TEST(CasparBundleAdjuster, ThreePointsGauge) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_model_id = CameraModelId::kSimpleRadial;

  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

  BundleAdjustmentOptions options;
  caspar::SolverParams params;
  auto summary =
      CreateCasparBundleAdjuster(options, config, reconstruction, params)
          ->Solve();
  ASSERT_NE(summary.termination_type, ceres::FAILURE);

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckVariableCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));

  size_t num_variable_points = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D != orig_reconstruction.Point3D(point3D_id)) {
      ++num_variable_points;
    }
  }
  EXPECT_EQ(num_variable_points, 97);
}

TEST(CasparBundleAdjuster, GaugeFixedPointsStayFixed) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_model_id = CameraModelId::kSimpleRadial;

  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

  BundleAdjustmentOptions options;
  caspar::SolverParams params;
  auto summary =
      CreateCasparBundleAdjuster(options, config, reconstruction, params)
          ->Solve();
  ASSERT_NE(summary.termination_type, ceres::FAILURE);

  // Find top 3 most-observed points (gauge-fixed)
  std::vector<std::pair<size_t, point3D_t>> points_by_obs;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    points_by_obs.push_back({point3D.track.Length(), point3D_id});
  }
  std::sort(points_by_obs.rbegin(), points_by_obs.rend());

  // Gauge-fixed points must not move
  for (size_t i = 0; i < 3; ++i) {
    const point3D_t point_id = points_by_obs[i].second;
    const double movement = (reconstruction.Point3D(point_id).xyz -
                             orig_reconstruction.Point3D(point_id).xyz)
                                .norm();
    EXPECT_LT(movement, 1e-9)
        << "Gauge-fixed point " << point_id << " moved by " << movement;
  }
}

TEST(CasparBundleAdjuster, CompareThreePointsGaugeWithDefault) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_model_id = CameraModelId::kSimpleRadial;

  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  Reconstruction reconstruction_ceres = reconstruction;
  Reconstruction reconstruction_caspar = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

  BundleAdjustmentOptions options;

  auto ceres_summary =
      CreateDefaultBundleAdjuster(options, config, reconstruction_ceres)
          ->Solve();
  ASSERT_NE(ceres_summary.termination_type, ceres::FAILURE);

  caspar::SolverParams params;
  auto caspar_summary =
      CreateCasparBundleAdjuster(options, config, reconstruction_caspar, params)
          ->Solve();
  ASSERT_NE(caspar_summary.termination_type, ceres::FAILURE);

  std::vector<double> point_errors;
  for (const auto& point3D_id : reconstruction_ceres.Point3DIds()) {
    const Point3D& point_ceres = reconstruction_ceres.Point3D(point3D_id);
    const Point3D& point_caspar = reconstruction_caspar.Point3D(point3D_id);
    point_errors.push_back((point_ceres.xyz - point_caspar.xyz).norm());
  }

  const double mean_error =
      std::accumulate(point_errors.begin(), point_errors.end(), 0.0) /
      point_errors.size();
  const double max_error =
      *std::max_element(point_errors.begin(), point_errors.end());

  EXPECT_LT(mean_error, 0.05);
  EXPECT_LT(max_error, 0.05);
}

TEST(CasparBundleAdjuster, TwoCamsGaugeCameraMovement) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_model_id = CameraModelId::kSimpleRadial;

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
  caspar::SolverParams params;
  auto summary =
      CreateCasparBundleAdjuster(options, config, reconstruction, params)
          ->Solve();
  ASSERT_NE(summary.termination_type, ceres::FAILURE);

  // Camera intrinsics should change
  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckVariableCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));

  // Camera 1 should be fixed
  EXPECT_THAT(reconstruction.Image(1).CamFromWorld(),
              Rigid3dNear(orig_reconstruction.Image(1).CamFromWorld(),
                          0.001,
                          0.001));  // Allow small numerical error

  // Camera 2 translation norm should be preserved
  const double orig_norm =
      orig_reconstruction.Image(2).CamFromWorld().translation.norm();
  const double new_norm =
      reconstruction.Image(2).CamFromWorld().translation.norm();
  EXPECT_NEAR(orig_norm, new_norm, 1e-6);

  // Points should move (optimization happened)
  size_t num_variable_points = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D != orig_reconstruction.Point3D(point3D_id)) {
      ++num_variable_points;
    }
  }
  EXPECT_GT(num_variable_points, 90);  // Most points should change
}

TEST(CasparBundleAdjuster, TwoCamsGaugeFrameOneNotCreatingNodes) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_model_id = CameraModelId::kSimpleRadial;

  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  // Save original pose BEFORE creating adjuster
  const Rigid3d frame1_pose_before = reconstruction.Image(1).CamFromWorld();

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  caspar::SolverParams params;

  // Create adjuster (constructor runs here)
  auto adjuster =
      CreateCasparBundleAdjuster(options, config, reconstruction, params);

  // Check if Frame 1 was modified during construction
  const Rigid3d frame1_pose_after_construction =
      reconstruction.Image(1).CamFromWorld();

  EXPECT_THAT(frame1_pose_after_construction,
              Rigid3dNear(frame1_pose_before, 1e-10, 1e-10))
      << "Frame 1 was modified during CasparBundleAdjuster construction!";
}

TEST(CasparBundleAdjuster, TwoCamsGaugeFrameOneDoesNotMoveAfterSolve) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_model_id = CameraModelId::kSimpleRadial;

  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  const Rigid3d frame1_before = reconstruction.Image(1).CamFromWorld();

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  caspar::SolverParams params;

  auto adjuster =
      CreateCasparBundleAdjuster(options, config, reconstruction, params);
  adjuster->Solve();  // This is where Frame 1 gets corrupted

  const Rigid3d frame1_after = reconstruction.Image(1).CamFromWorld();

  EXPECT_THAT(frame1_after, Rigid3dNear(frame1_before, 1e-9, 1e-9))
      << "Frame 1 moved during Solve()!";
}
TEST(CasparBundleAdjuster, GaugeFixingSingularityIsolate) {
  Reconstruction reconstruction;

  // 1. Setup a Camera
  Camera camera;
  camera.camera_id = 1;
  camera.model_id = CameraModelId::kSimpleRadial;
  camera.params = {1000, 500, 500, 0};
  reconstruction.AddCamera(camera);

  auto RunTest = [&](const Eigen::Vector3d& t2, const std::string& label) {
    // Clear and reset for each run
    reconstruction = Reconstruction();
    reconstruction.AddCamera(camera);

    // Setup Frame 1 and Image 1
    const frame_t frame_id1 = 1;
    const image_t image_id1 = 1;
    reconstruction.AddFrame(Frame());  // Create empty frame
    reconstruction.Frame(frame_id1).RigFromWorld() = Rigid3d();  // Identity

    Image image1;
    image1.SetImageId(image_id1);
    image1.SetCameraId(1);
    image1.SetFrameId(frame_id1);  // Link Image to Frame
    reconstruction.AddImage(image1);

    // Setup Frame 2 and Image 2
    const frame_t frame_id2 = 2;
    const image_t image_id2 = 2;
    reconstruction.AddFrame(Frame());
    Rigid3d pose2;
    pose2.translation = t2;
    reconstruction.Frame(frame_id2).RigFromWorld() = pose2;

    Image image2;
    image2.SetImageId(image_id2);
    image2.SetCameraId(1);
    image2.SetFrameId(frame_id2);
    reconstruction.AddImage(image2);

    // Register Frames to make them active in the reconstruction
    reconstruction.RegisterFrame(frame_id1);
    reconstruction.RegisterFrame(frame_id2);

    BundleAdjustmentConfig config;
    config.AddImage(image_id1);
    config.AddImage(image_id2);
    config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

    BundleAdjustmentOptions options;
    caspar::SolverParams params;

    auto adjuster =
        CreateCasparBundleAdjuster(options, config, reconstruction, params);
    adjuster->Solve();

    // Check if Frame 1 moved from Identity
    const bool frame1_moved =
        !reconstruction.Frame(frame_id1).RigFromWorld().ToMatrix().isApprox(
            Rigid3d().ToMatrix(), 1e-12);

    LOG(INFO) << "[" << label
              << "] Frame 1 moved: " << (frame1_moved ? "YES" : "NO");
    return frame1_moved;
  };

  // Execution
  bool failed_on_x = RunTest(Eigen::Vector3d(1.0, 0.0, 0.0), "X-AXIS");
  bool failed_on_y = RunTest(Eigen::Vector3d(0.0, 1.0, 0.0), "Y-AXIS");

  EXPECT_FALSE(failed_on_y)
      << "Gauge fixing failed even on non-singular Y-axis.";
  EXPECT_FALSE(failed_on_x)
      << "Gauge fixing failed on X-axis (Singularity triggered fallback).";
}
#endif  // CASPAR_ENABLED

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
      CreateDefaultBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary.termination_type, ceres::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            bundle_adjuster->Problem()->NumResiduals());

  // 100 points, 3 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 600);
  // 100 x 3 point parameters
  // + 2 translation parameters (second image, one coord fixed for gauge)
  // + 3 translation parameters (third image)
  // + 3 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 311);

  // Check rotations are constant for all images
  for (const image_t image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);
    const auto& orig_image = orig_reconstruction.Image(image_id);
    // Rotation should be nearly unchanged (use angular distance)
    EXPECT_LE(image.CamFromWorld().rotation.angularDistance(
                  orig_image.CamFromWorld().rotation),
              kConstantPoseVarEps);
  }

  // Check translations are variable (except for gauge-fixed parts)
  // At least one image should have changed translation
  bool has_variable_translation = false;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);
    const auto& orig_image = orig_reconstruction.Image(image_id);
    if ((image.CamFromWorld().translation -
         orig_image.CamFromWorld().translation)
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
    const auto summary1 = CreateDefaultBundleAdjuster(
                              BundleAdjustmentOptions(), config, reconstruction)
                              ->Solve();
    ASSERT_NE(summary1.termination_type, ceres::FAILURE);
    EXPECT_EQ(summary1.num_effective_parameters_reduced,
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
        CreateDefaultBundleAdjuster(options, config, reconstruction)->Solve();
    ASSERT_NE(summary1.termination_type, ceres::FAILURE);
    EXPECT_EQ(summary1.num_effective_parameters_reduced,
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
        CreateDefaultBundleAdjuster(options, config, reconstruction)->Solve();
    ASSERT_NE(summary1.termination_type, ceres::FAILURE);
    EXPECT_EQ(summary1.num_effective_parameters_reduced,
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
        CreateDefaultBundleAdjuster(options, config, reconstruction)->Solve();
    THROW_CHECK_NE(summary1.termination_type, ceres::FAILURE);
    THROW_CHECK_EQ(summary1.num_effective_parameters_reduced,
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
      CreateDefaultBundleAdjuster(options, config, reconstruction)->Solve();
  ASSERT_NE(summary.termination_type, ceres::FAILURE);
  EXPECT_EQ(summary.num_effective_parameters, 316);
  EXPECT_EQ(summary.num_effective_parameters_reduced, 307);
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
      CreateDefaultBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary.termination_type, ceres::FAILURE);

  EXPECT_EQ(config.NumResiduals(reconstruction),
            bundle_adjuster->Problem()->NumResiduals());

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 396);
  // 99 x 3 point parameters
  // + 5 rig_from_world parameters (pose of second image)
  // + 2 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 306);
}

TEST(PosePriorBundleAdjuster, AlignmentRobustToOutliers) {
  SetPRNGSeed(0);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 7;
  synthetic_options.num_points3D = 50;
  synthetic_options.prior_position = true;
  std::string database_path = CreateTestDir() + "/database.db";
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
  ASSERT_TRUE(summary.IsSolutionUsable());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(PosePriorBundleAdjuster, MissingPositionCov) {
  SetPRNGSeed(0);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 7;
  synthetic_options.num_points3D = 100;
  synthetic_options.prior_position = true;
  std::string database_path = CreateTestDir() + "/database.db";
  auto database = Database::Open(database_path);
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction = gt_reconstruction;

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
  for (PosePrior& pose_prior : pose_priors) {
    EXPECT_FALSE(pose_prior.HasPositionCov());
  }

  PosePriorBundleAdjustmentOptions prior_ba_options;
  prior_ba_options.alignment_ransac_options.random_seed = 0;
  prior_ba_options.prior_position_loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::CAUCHY;

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
  ASSERT_TRUE(summary.IsSolutionUsable());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(PosePriorBundleAdjuster, OptimizationRobustToOutliers) {
  SetPRNGSeed(0);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 7;
  synthetic_options.num_points3D = 100;
  synthetic_options.prior_position = true;
  std::string database_path = CreateTestDir() + "/database.db";
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
  prior_ba_options.prior_position_loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::CAUCHY;

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
  ASSERT_TRUE(summary.IsSolutionUsable());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

}  // namespace
}  // namespace colmap

// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/math/random.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/projection.h"
#include "colmap/sensor/models.h"

#include <gtest/gtest.h>

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

#define CheckVariableImage(image, orig_image)                 \
  {                                                           \
    EXPECT_NE((image).CamFromWorld().rotation.coeffs(),       \
              (orig_image).CamFromWorld().rotation.coeffs()); \
    EXPECT_NE((image).CamFromWorld().translation,             \
              (orig_image).CamFromWorld().translation);       \
  }

#define CheckConstantImage(image, orig_image)                 \
  {                                                           \
    EXPECT_EQ((image).CamFromWorld().rotation.coeffs(),       \
              (orig_image).CamFromWorld().rotation.coeffs()); \
    EXPECT_EQ((image).CamFromWorld().translation,             \
              (orig_image).CamFromWorld().translation);       \
  }

#define CheckConstantXImage(image, orig_image)              \
  {                                                         \
    CheckVariableImage(image, orig_image);                  \
    EXPECT_EQ((image).CamFromWorld().translation.x(),       \
              (orig_image).CamFromWorld().translation.x()); \
  }

#define CheckConstantCameraRig(camera_rig, orig_camera_rig, camera_id)    \
  {                                                                       \
    EXPECT_EQ((camera_rig).CamFromRig(camera_id).rotation.coeffs(),       \
              (orig_camera_rig).CamFromRig(camera_id).rotation.coeffs()); \
    EXPECT_EQ((camera_rig).CamFromRig(camera_id).translation,             \
              (orig_camera_rig).CamFromRig(camera_id).translation);       \
  }

#define CheckVariableCameraRig(camera_rig, orig_camera_rig, camera_id)      \
  {                                                                         \
    if ((camera_rig).RefCameraId() == (camera_id)) {                        \
      CheckConstantCameraRig(camera_rig, orig_camera_rig, camera_id);       \
    } else {                                                                \
      EXPECT_NE((camera_rig).CamFromRig(camera_id).rotation.coeffs(),       \
                (orig_camera_rig).CamFromRig(camera_id).rotation.coeffs()); \
      EXPECT_NE((camera_rig).CamFromRig(camera_id).translation,             \
                (orig_camera_rig).CamFromRig(camera_id).translation);       \
    }                                                                       \
  }

#define CheckVariablePoint(point, orig_point) \
  { EXPECT_NE((point).xyz, (orig_point).xyz); }

#define CheckConstantPoint(point, orig_point) \
  { EXPECT_EQ((point).xyz, (orig_point).xyz); }

namespace colmap {
namespace {

void GeneratePointCloud(const size_t num_points,
                        const Eigen::Vector3d& min,
                        const Eigen::Vector3d& max,
                        Reconstruction* reconstruction) {
  for (size_t i = 0; i < num_points; ++i) {
    Eigen::Vector3d xyz;
    xyz.x() = RandomUniformReal(min.x(), max.x());
    xyz.y() = RandomUniformReal(min.y(), max.y());
    xyz.z() = RandomUniformReal(min.z(), max.z());
    reconstruction->AddPoint3D(xyz, Track());
  }
}

void GenerateReconstruction(const size_t num_images,
                            const size_t num_points,
                            Reconstruction* reconstruction) {
  SetPRNGSeed(0);

  GeneratePointCloud(num_points,
                     Eigen::Vector3d(-1, -1, -1),
                     Eigen::Vector3d(1, 1, 1),
                     reconstruction);

  const double kFocalLengthFactor = 1.2;
  const size_t kImageSize = 1000;

  for (size_t i = 0; i < num_images; ++i) {
    const camera_t camera_id = static_cast<camera_t>(i);
    const image_t image_id = static_cast<image_t>(i);

    const Camera camera =
        Camera::CreateFromModelId(camera_id,
                                  SimpleRadialCameraModel::model_id,
                                  kFocalLengthFactor * kImageSize,
                                  kImageSize,
                                  kImageSize);
    reconstruction->AddCamera(camera);

    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(camera_id);
    image.SetName(std::to_string(i));
    image.CamFromWorld() = Rigid3d(
        Eigen::Quaterniond::Identity(),
        Eigen::Vector3d(
            RandomUniformReal(-1.0, 1.0), RandomUniformReal(-1.0, 1.0), 10));
    image.SetRegistered(true);
    reconstruction->AddImage(image);

    const Eigen::Matrix3x4d cam_from_world_matrix =
        image.CamFromWorld().ToMatrix();

    std::vector<Eigen::Vector2d> points2D;
    for (const auto& point3D : reconstruction->Points3D()) {
      EXPECT_TRUE(
          HasPointPositiveDepth(cam_from_world_matrix, point3D.second.xyz));
      // Get exact projection of 3D point.
      Eigen::Vector2d point2D = camera.ImgFromCam(
          (image.CamFromWorld() * point3D.second.xyz).hnormalized());
      // Add some uniform noise.
      point2D += Eigen::Vector2d(RandomUniformReal(-2.0, 2.0),
                                 RandomUniformReal(-2.0, 2.0));
      points2D.push_back(point2D);
    }

    reconstruction->Image(image_id).SetPoints2D(points2D);
  }

  for (size_t i = 0; i < num_images; ++i) {
    const image_t image_id = static_cast<image_t>(i);
    TrackElement track_el;
    track_el.image_id = image_id;
    track_el.point2D_idx = 0;
    for (const auto& point3D : reconstruction->Points3D()) {
      reconstruction->AddObservation(point3D.first, track_el);
      track_el.point2D_idx += 1;
    }
  }
}

TEST(BundleAdjustment, ConfigNumObservations) {
  Reconstruction reconstruction;
  GenerateReconstruction(4, 100, &reconstruction);

  BundleAdjustmentConfig config;

  config.AddImage(0);
  config.AddImage(1);
  EXPECT_EQ(config.NumResiduals(reconstruction), 400);

  config.AddVariablePoint(1);
  EXPECT_EQ(config.NumResiduals(reconstruction), 404);

  config.AddConstantPoint(2);
  EXPECT_EQ(config.NumResiduals(reconstruction), 408);

  config.AddImage(2);
  EXPECT_EQ(config.NumResiduals(reconstruction), 604);

  config.AddImage(3);
  EXPECT_EQ(config.NumResiduals(reconstruction), 800);
}

TEST(BundleAdjustment, TwoView) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, 100, &reconstruction);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantCamPose(0);
  config.SetConstantCamPositions(1, {0});

  BundleAdjustmentOptions options;
  BundleAdjuster bundle_adjuster(options, config);
  ASSERT_TRUE(bundle_adjuster.Solve(&reconstruction));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 5 image parameters (pose of second image)
  // + 2 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 309);

  CheckVariableCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckConstantImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckConstantXImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}

TEST(BundleAdjustment, TwoViewConstantCamera) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, 100, &reconstruction);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantCamPose(0);
  config.SetConstantCamPose(1);
  config.SetConstantCamIntrinsics(0);

  BundleAdjustmentOptions options;
  BundleAdjuster bundle_adjuster(options, config);
  ASSERT_TRUE(bundle_adjuster.Solve(&reconstruction));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 302);

  CheckConstantCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckConstantImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckConstantImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}

TEST(BundleAdjustment, PartiallyContainedTracks) {
  Reconstruction reconstruction;
  GenerateReconstruction(3, 100, &reconstruction);
  const auto variable_point3D_id =
      reconstruction.Image(2).Point2D(0).point3D_id;
  reconstruction.DeleteObservation(2, 0);

  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantCamPose(0);
  config.SetConstantCamPose(1);

  BundleAdjustmentOptions options;
  BundleAdjuster bundle_adjuster(options, config);
  ASSERT_TRUE(bundle_adjuster.Solve(&reconstruction));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 400);
  // 1 x 3 point parameters
  // 2 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 7);

  CheckVariableCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckConstantImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckConstantImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  CheckConstantCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
  CheckConstantImage(reconstruction.Image(2), orig_reconstruction.Image(2));

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

TEST(BundleAdjustment, PartiallyContainedTracksForceToOptimizePoint) {
  Reconstruction reconstruction;
  GenerateReconstruction(3, 100, &reconstruction);
  const point3D_t variable_point3D_id =
      reconstruction.Image(2).Point2D(0).point3D_id;
  const point3D_t add_variable_point3D_id =
      reconstruction.Image(2).Point2D(1).point3D_id;
  const point3D_t add_constant_point3D_id =
      reconstruction.Image(2).Point2D(2).point3D_id;
  reconstruction.DeleteObservation(2, 0);

  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantCamPose(0);
  config.SetConstantCamPose(1);
  config.AddVariablePoint(add_variable_point3D_id);
  config.AddConstantPoint(add_constant_point3D_id);

  BundleAdjustmentOptions options;
  BundleAdjuster bundle_adjuster(options, config);
  ASSERT_TRUE(bundle_adjuster.Solve(&reconstruction));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 2 images, 2 residuals per point per image
  // + 2 residuals in 3rd image for added variable 3D point
  // (added constant point does not add residuals since the image/camera
  // is also constant).
  EXPECT_EQ(summary.num_residuals_reduced, 402);
  // 2 x 3 point parameters
  // 2 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 10);

  CheckVariableCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckConstantImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckConstantImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  CheckConstantCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
  CheckConstantImage(reconstruction.Image(2), orig_reconstruction.Image(2));

  for (const auto& point3D : reconstruction.Points3D()) {
    if (point3D.first == variable_point3D_id ||
        point3D.first == add_variable_point3D_id) {
      CheckVariablePoint(point3D.second,
                         orig_reconstruction.Point3D(point3D.first));
    } else {
      CheckConstantPoint(point3D.second,
                         orig_reconstruction.Point3D(point3D.first));
    }
  }
}

TEST(BundleAdjustment, ConstantPoints) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, 100, &reconstruction);
  const auto orig_reconstruction = reconstruction;

  const point3D_t constant_point3D_id1 = 1;
  const point3D_t constant_point3D_id2 = 2;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantCamPose(0);
  config.SetConstantCamPose(1);
  config.AddConstantPoint(constant_point3D_id1);
  config.AddConstantPoint(constant_point3D_id2);

  BundleAdjustmentOptions options;
  BundleAdjuster bundle_adjuster(options, config);
  ASSERT_TRUE(bundle_adjuster.Solve(&reconstruction));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 400);
  // 98 x 3 point parameters
  // + 2 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 298);

  CheckVariableCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckConstantImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckConstantImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  for (const auto& point3D : reconstruction.Points3D()) {
    if (point3D.first == constant_point3D_id1 ||
        point3D.first == constant_point3D_id2) {
      CheckConstantPoint(point3D.second,
                         orig_reconstruction.Point3D(point3D.first));
    } else {
      CheckVariablePoint(point3D.second,
                         orig_reconstruction.Point3D(point3D.first));
    }
  }
}

TEST(BundleAdjustment, VariableImage) {
  Reconstruction reconstruction;
  GenerateReconstruction(3, 100, &reconstruction);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.AddImage(2);
  config.SetConstantCamPose(0);
  config.SetConstantCamPositions(1, {0});

  BundleAdjustmentOptions options;
  BundleAdjuster bundle_adjuster(options, config);
  ASSERT_TRUE(bundle_adjuster.Solve(&reconstruction));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 3 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 600);
  // 100 x 3 point parameters
  // + 5 image parameters (pose of second image)
  // + 6 image parameters (pose of third image)
  // + 3 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 317);

  CheckVariableCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckConstantImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckConstantXImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  CheckVariableCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
  CheckVariableImage(reconstruction.Image(2), orig_reconstruction.Image(2));

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}

TEST(BundleAdjustment, ConstantFocalLength) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, 100, &reconstruction);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantCamPose(0);
  config.SetConstantCamPositions(1, {0});

  BundleAdjustmentOptions options;
  options.refine_focal_length = false;
  BundleAdjuster bundle_adjuster(options, config);
  ASSERT_TRUE(bundle_adjuster.Solve(&reconstruction));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 3 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 5 image parameters (pose of second image)
  // + 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 307);

  CheckConstantImage(reconstruction.Image(0), orig_reconstruction.Image(0));
  CheckConstantXImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  const size_t focal_length_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t extra_param_idx = SimpleRadialCameraModel::extra_params_idxs[0];

  const auto& camera0 = reconstruction.Camera(0);
  const auto& orig_camera0 = orig_reconstruction.Camera(0);
  EXPECT_TRUE(camera0.params[focal_length_idx] ==
              orig_camera0.params[focal_length_idx]);
  EXPECT_TRUE(camera0.params[extra_param_idx] !=
              orig_camera0.params[extra_param_idx]);

  const auto& camera1 = reconstruction.Camera(1);
  const auto& orig_camera1 = orig_reconstruction.Camera(1);
  EXPECT_TRUE(camera1.params[focal_length_idx] ==
              orig_camera1.params[focal_length_idx]);
  EXPECT_TRUE(camera1.params[extra_param_idx] !=
              orig_camera1.params[extra_param_idx]);

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}

TEST(BundleAdjustment, VariablePrincipalPoint) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, 100, &reconstruction);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantCamPose(0);
  config.SetConstantCamPositions(1, {0});

  BundleAdjustmentOptions options;
  options.refine_principal_point = true;
  BundleAdjuster bundle_adjuster(options, config);
  ASSERT_TRUE(bundle_adjuster.Solve(&reconstruction));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 3 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 5 image parameters (pose of second image)
  // + 8 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 313);

  CheckConstantImage(reconstruction.Image(0), orig_reconstruction.Image(0));
  CheckConstantXImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  const size_t focal_length_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t principal_point_idx_x =
      SimpleRadialCameraModel::principal_point_idxs[0];
  const size_t principal_point_idx_y =
      SimpleRadialCameraModel::principal_point_idxs[0];
  const size_t extra_param_idx = SimpleRadialCameraModel::extra_params_idxs[0];

  const auto& camera0 = reconstruction.Camera(0);
  const auto& orig_camera0 = orig_reconstruction.Camera(0);
  EXPECT_TRUE(camera0.params[focal_length_idx] !=
              orig_camera0.params[focal_length_idx]);
  EXPECT_TRUE(camera0.params[principal_point_idx_x] !=
              orig_camera0.params[principal_point_idx_x]);
  EXPECT_TRUE(camera0.params[principal_point_idx_y] !=
              orig_camera0.params[principal_point_idx_y]);
  EXPECT_TRUE(camera0.params[extra_param_idx] !=
              orig_camera0.params[extra_param_idx]);

  const auto& camera1 = reconstruction.Camera(1);
  const auto& orig_camera1 = orig_reconstruction.Camera(1);
  EXPECT_TRUE(camera1.params[focal_length_idx] !=
              orig_camera1.params[focal_length_idx]);
  EXPECT_TRUE(camera1.params[principal_point_idx_x] !=
              orig_camera1.params[principal_point_idx_x]);
  EXPECT_TRUE(camera1.params[principal_point_idx_y] !=
              orig_camera1.params[principal_point_idx_y]);
  EXPECT_TRUE(camera1.params[extra_param_idx] !=
              orig_camera1.params[extra_param_idx]);

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}

TEST(BundleAdjustment, ConstantExtraParam) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, 100, &reconstruction);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantCamPose(0);
  config.SetConstantCamPositions(1, {0});

  BundleAdjustmentOptions options;
  options.refine_extra_params = false;
  BundleAdjuster bundle_adjuster(options, config);
  ASSERT_TRUE(bundle_adjuster.Solve(&reconstruction));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 3 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 5 image parameters (pose of second image)
  // + 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 307);

  CheckConstantImage(reconstruction.Image(0), orig_reconstruction.Image(0));
  CheckConstantXImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  const size_t focal_length_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t extra_param_idx = SimpleRadialCameraModel::extra_params_idxs[0];

  const auto& camera0 = reconstruction.Camera(0);
  const auto& orig_camera0 = orig_reconstruction.Camera(0);
  EXPECT_TRUE(camera0.params[focal_length_idx] !=
              orig_camera0.params[focal_length_idx]);
  EXPECT_TRUE(camera0.params[extra_param_idx] ==
              orig_camera0.params[extra_param_idx]);

  const auto& camera1 = reconstruction.Camera(1);
  const auto& orig_camera1 = orig_reconstruction.Camera(1);
  EXPECT_TRUE(camera1.params[focal_length_idx] !=
              orig_camera1.params[focal_length_idx]);
  EXPECT_TRUE(camera1.params[extra_param_idx] ==
              orig_camera1.params[extra_param_idx]);

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}

TEST(BundleAdjustment, RigTwoView) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, 100, &reconstruction);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);

  std::vector<CameraRig> camera_rigs;
  camera_rigs.emplace_back();
  camera_rigs[0].AddCamera(0, Rigid3d());
  camera_rigs[0].AddCamera(1, Rigid3d());
  camera_rigs[0].AddSnapshot({0, 1});
  camera_rigs[0].SetRefCameraId(0);
  const auto orig_camera_rigs = camera_rigs;

  BundleAdjustmentOptions options;
  RigBundleAdjuster::Options rig_options;
  RigBundleAdjuster bundle_adjuster(options, rig_options, config);
  ASSERT_TRUE(bundle_adjuster.Solve(&reconstruction, &camera_rigs));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 6 pose parameters for camera rig
  // + 1 x 6 relative pose parameters for camera rig
  // + 2 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 316);

  CheckVariableCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckVariableImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckVariableImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  CheckVariableCameraRig(camera_rigs[0], orig_camera_rigs[0], 0);
  CheckVariableCameraRig(camera_rigs[0], orig_camera_rigs[0], 1);

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}

TEST(BundleAdjustment, RigFourView) {
  Reconstruction reconstruction;
  GenerateReconstruction(4, 100, &reconstruction);
  reconstruction.Image(2).SetCameraId(0);
  reconstruction.Image(3).SetCameraId(1);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.AddImage(2);
  config.AddImage(3);

  std::vector<CameraRig> camera_rigs;
  camera_rigs.emplace_back();
  camera_rigs[0].AddCamera(0, Rigid3d());
  camera_rigs[0].AddCamera(1, Rigid3d());
  camera_rigs[0].AddSnapshot({0, 1});
  camera_rigs[0].AddSnapshot({2, 3});
  camera_rigs[0].SetRefCameraId(0);
  const auto orig_camera_rigs = camera_rigs;

  BundleAdjustmentOptions options;
  RigBundleAdjuster::Options rig_options;
  RigBundleAdjuster bundle_adjuster(options, rig_options, config);
  ASSERT_TRUE(bundle_adjuster.Solve(&reconstruction, &camera_rigs));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 800);
  // 100 x 3 point parameters
  // + 2 x 6 pose parameters for camera rig
  // + 1 x 6 relative pose parameters for camera rig
  // + 2 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 322);

  CheckVariableCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckVariableImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckVariableImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  CheckVariableCameraRig(camera_rigs[0], orig_camera_rigs[0], 0);
  CheckVariableCameraRig(camera_rigs[0], orig_camera_rigs[0], 1);

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}

TEST(BundleAdjustment, ConstantRigFourView) {
  Reconstruction reconstruction;
  GenerateReconstruction(4, 100, &reconstruction);
  reconstruction.Image(2).SetCameraId(0);
  reconstruction.Image(3).SetCameraId(1);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.AddImage(2);
  config.AddImage(3);

  std::vector<CameraRig> camera_rigs;
  camera_rigs.emplace_back();
  camera_rigs[0].AddCamera(0, Rigid3d());
  camera_rigs[0].AddCamera(1, Rigid3d());
  camera_rigs[0].AddSnapshot({0, 1});
  camera_rigs[0].AddSnapshot({2, 3});
  camera_rigs[0].SetRefCameraId(0);
  const auto orig_camera_rigs = camera_rigs;

  BundleAdjustmentOptions options;
  RigBundleAdjuster::Options rig_options;
  rig_options.refine_relative_poses = false;
  RigBundleAdjuster bundle_adjuster(options, rig_options, config);
  ASSERT_TRUE(bundle_adjuster.Solve(&reconstruction, &camera_rigs));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 800);
  // 100 x 3 point parameters
  // + 2 x 6 pose parameters for camera rig
  // + 2 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 316);

  CheckVariableCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckVariableImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckVariableImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  CheckConstantCameraRig(camera_rigs[0], orig_camera_rigs[0], 0);
  CheckConstantCameraRig(camera_rigs[0], orig_camera_rigs[0], 1);

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}

TEST(BundleAdjustment, RigFourViewPartial) {
  Reconstruction reconstruction;
  GenerateReconstruction(4, 100, &reconstruction);
  reconstruction.Image(2).SetCameraId(0);
  reconstruction.Image(3).SetCameraId(1);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.AddImage(2);
  config.AddImage(3);

  std::vector<CameraRig> camera_rigs;
  camera_rigs.emplace_back();
  camera_rigs[0].AddCamera(0, Rigid3d());
  camera_rigs[0].AddCamera(1, Rigid3d());
  camera_rigs[0].AddSnapshot({0, 1});
  camera_rigs[0].AddSnapshot({2});
  camera_rigs[0].SetRefCameraId(0);
  const auto orig_camera_rigs = camera_rigs;

  BundleAdjustmentOptions options;
  RigBundleAdjuster::Options rig_options;
  RigBundleAdjuster bundle_adjuster(options, rig_options, config);
  ASSERT_TRUE(bundle_adjuster.Solve(&reconstruction, &camera_rigs));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 2 images, 2 residuals per point per image
  EXPECT_EQ(summary.num_residuals_reduced, 800);
  // 100 x 3 point parameters
  // + 2 x 6 pose parameters for camera rig
  // + 1 x 6 relative pose parameters for camera rig
  // + 1 x 6 pose parameters for individual image
  // + 2 x 2 camera parameters
  EXPECT_EQ(summary.num_effective_parameters_reduced, 328);

  CheckVariableCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckVariableImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckVariableImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  CheckVariableCameraRig(camera_rigs[0], orig_camera_rigs[0], 0);
  CheckVariableCameraRig(camera_rigs[0], orig_camera_rigs[0], 1);

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}

}  // namespace
}  // namespace colmap

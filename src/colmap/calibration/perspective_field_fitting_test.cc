// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/calibration/perspective_field_fitting.h"

#include "colmap/util/eigen_matchers.h"

#include <cmath>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(PerspectiveFieldFittingTest, SubsampleMaxPoolVsUniform) {
  Camera camera =
      Camera::CreateFromModelName(1, "SIMPLE_PINHOLE", 50.0, 16, 16);
  const Eigen::Vector3d gt_gravity =
      Eigen::Vector3d(0.1, 0.9, -0.2).normalized();
  PerspectiveField field =
      ComputePerspectiveFieldFromCameraAndGravity(camera, gt_gravity, 16, 16);
  ASSERT_TRUE(field.Check());
  ASSERT_EQ(field.NumPoints(), 256);

  // Set all confidences low except one pixel per 4x4 window.
  field.up_confidence.setConstant(0.1);
  field.latitude_confidence.setConstant(0.1);
  for (int r0 = 0; r0 < 16; r0 += 4) {
    for (int c0 = 0; c0 < 16; c0 += 4) {
      const int idx = (r0 + 2) * 16 + (c0 + 3);
      field.up_confidence(idx) = 0.95;
      field.latitude_confidence(idx) = 0.95;
    }
  }

  const PerspectiveField pooled = SubsamplePerspectiveField(
      field, /*stride=*/4, /*max_pool_confidence=*/true);
  ASSERT_EQ(pooled.NumPoints(), 16);
  for (size_t i = 0; i < pooled.NumPoints(); ++i) {
    EXPECT_NEAR(pooled.up_confidence(i), 0.95, 1e-9);
    EXPECT_NEAR(pooled.latitude_confidence(i), 0.95, 1e-9);
  }

  const PerspectiveField uniform = SubsamplePerspectiveField(
      field, /*stride=*/4, /*max_pool_confidence=*/false);
  ASSERT_EQ(uniform.NumPoints(), 16);
  for (size_t i = 0; i < uniform.NumPoints(); ++i) {
    EXPECT_NEAR(uniform.up_confidence(i), 0.1, 1e-9);
  }
}

TEST(PerspectiveFieldFittingTest, FixedCameraRecoversGravity) {
  Camera camera = Camera::CreateFromModelName(1, "SIMPLE_RADIAL", 80.0, 64, 48);
  camera.params[3] = -0.15;
  camera.has_prior_focal_length = true;

  const Eigen::Vector3d gt_gravity =
      Eigen::Vector3d(0.25, 0.85, -0.45).normalized();
  const PerspectiveField field =
      ComputePerspectiveFieldFromCameraAndGravity(camera, gt_gravity, 32, 24);

  PerspectiveFieldFittingOptions options;
  options.stride = 2;
  const FittedPerspectiveFields fitted =
      FitPerspectiveField(options, field, &camera, /*refine_camera=*/false);
  ASSERT_TRUE(fitted.success);
  EXPECT_LT(fitted.final_cost, 1e-10);
  EXPECT_THAT(fitted.gravity_in_rig, EigenMatrixNear(gt_gravity, 1e-5));
}

struct CameraModelCase {
  std::string model_name;
  std::vector<double> gt_params;
};

class RefineCameraPerspectiveFieldTest
    : public ::testing::TestWithParam<CameraModelCase> {};

TEST_P(RefineCameraPerspectiveFieldTest, RecoversGravityAndIntrinsics) {
  const CameraModelCase& test_case = GetParam();
  Camera gt_camera;
  gt_camera.camera_id = 1;
  gt_camera.model_id = CameraModelNameToId(test_case.model_name);
  gt_camera.width = 64;
  gt_camera.height = 48;
  gt_camera.params = test_case.gt_params;
  ASSERT_TRUE(gt_camera.VerifyParams());

  // Non-zero roll and pitch so vertical vanishing point and radial distortion
  // are well-constrained.
  const Eigen::Vector3d gt_gravity =
      Eigen::Vector3d(0.2, 0.88, -0.42).normalized();
  const PerspectiveField field = ComputePerspectiveFieldFromCameraAndGravity(
      gt_camera, gt_gravity, 32, 24);

  // Start with a wrong focal length and zero distortion.
  Camera est_camera = Camera::CreateFromModelId(
      1, gt_camera.model_id, /*focal_length=*/45.0, 64, 48);
  est_camera.has_prior_focal_length = false;

  PerspectiveFieldFittingOptions options;
  options.stride = 2;
  options.refine_focal_length = true;
  options.refine_principal_point = false;
  options.refine_extra_params = true;

  const FittedPerspectiveFields fitted =
      FitPerspectiveField(options, field, &est_camera, /*refine_camera=*/true);
  ASSERT_TRUE(fitted.success);
  EXPECT_LT(fitted.final_cost, 1e-8);
  EXPECT_THAT(fitted.gravity_in_rig, EigenMatrixNear(gt_gravity, 1e-3));
  for (size_t i = 0; i < test_case.gt_params.size(); ++i) {
    EXPECT_NEAR(est_camera.params[i], test_case.gt_params[i], 1e-2)
        << "param " << i << " of " << test_case.model_name;
  }
}

INSTANTIATE_TEST_SUITE_P(
    CameraModels,
    RefineCameraPerspectiveFieldTest,
    ::testing::Values(
        CameraModelCase{"SIMPLE_PINHOLE", {72.0, 32.0, 24.0}},
        CameraModelCase{"PINHOLE", {72.0, 72.0, 32.0, 24.0}},
        CameraModelCase{"SIMPLE_RADIAL", {72.0, 32.0, 24.0, -0.12}},
        CameraModelCase{"SIMPLE_DIVISION", {72.0, 32.0, 24.0, -0.15}},
        CameraModelCase{"SIMPLE_FISHEYE", {55.0, 32.0, 24.0}},
        CameraModelCase{"SIMPLE_RADIAL_FISHEYE", {55.0, 32.0, 24.0, -0.08}}));

TEST(PerspectiveFieldFittingTest, EquirectangularFixedCameraRecoversGravity) {
  Camera camera =
      Camera::CreateFromModelName(1, "EQUIRECTANGULAR", 1.0, 64, 32);
  const Eigen::Vector3d gt_gravity =
      Eigen::Vector3d(-0.3, 0.85, 0.43).normalized();
  const PerspectiveField field =
      ComputePerspectiveFieldFromCameraAndGravity(camera, gt_gravity, 32, 16);

  PerspectiveFieldFittingOptions options;
  options.stride = 2;
  const FittedPerspectiveFields fitted =
      FitPerspectiveField(options, field, &camera, /*refine_camera=*/false);
  ASSERT_TRUE(fitted.success);
  EXPECT_LT(fitted.final_cost, 1e-10);
  EXPECT_THAT(fitted.gravity_in_rig, EigenMatrixNear(gt_gravity, 1e-5));
}

TEST(PerspectiveFieldFittingTest, MultiCameraRigJointOptimization) {
  Camera cam1 = Camera::CreateFromModelName(1, "SIMPLE_RADIAL", 75.0, 64, 48);
  cam1.params[3] = -0.1;
  Camera cam2 = Camera::CreateFromModelName(2, "SIMPLE_PINHOLE", 60.0, 64, 48);

  const Eigen::Quaterniond cam1_from_rig = Eigen::Quaterniond::Identity();
  const Eigen::Quaterniond cam2_from_rig(
      Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(-0.15, Eigen::Vector3d::UnitX()));

  const Eigen::Vector3d gt_gravity_in_rig =
      Eigen::Vector3d(-0.15, 0.92, 0.36).normalized();
  const Eigen::Vector3d gt_gravity_cam1 = cam1_from_rig * gt_gravity_in_rig;
  const Eigen::Vector3d gt_gravity_cam2 = cam2_from_rig * gt_gravity_in_rig;

  const PerspectiveField field1 = ComputePerspectiveFieldFromCameraAndGravity(
      cam1, gt_gravity_cam1, 32, 24);
  const PerspectiveField field2 = ComputePerspectiveFieldFromCameraAndGravity(
      cam2, gt_gravity_cam2, 32, 24);

  // Perturb cam1 intrinsics and keep cam2 fixed.
  Camera est_cam1 =
      Camera::CreateFromModelName(1, "SIMPLE_RADIAL", 50.0, 64, 48);
  est_cam1.has_prior_focal_length = false;
  Camera est_cam2 = cam2;
  est_cam2.has_prior_focal_length = true;

  std::vector<PerspectiveFieldCameraInput> inputs(2);
  inputs[0].field = &field1;
  inputs[0].camera = &est_cam1;
  inputs[0].cam_from_rig = cam1_from_rig;
  inputs[0].refine_camera = true;

  inputs[1].field = &field2;
  inputs[1].camera = &est_cam2;
  inputs[1].cam_from_rig = cam2_from_rig;
  inputs[1].refine_camera = false;

  PerspectiveFieldFittingOptions options;
  options.stride = 2;
  const FittedPerspectiveFields fitted = FitPerspectiveFields(options, inputs);
  ASSERT_TRUE(fitted.success);
  EXPECT_THAT(fitted.gravity_in_rig, EigenMatrixNear(gt_gravity_in_rig, 1e-4));
  EXPECT_NEAR(est_cam1.MeanFocalLength(), 75.0, 1e-2);
  EXPECT_NEAR(est_cam1.params[3], -0.1, 1e-3);
}

}  // namespace
}  // namespace colmap

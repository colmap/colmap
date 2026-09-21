// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/pose.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/math/random_eigen.h"
#include "colmap/scene/synthetic.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

struct AbsolutePoseProblem {
  Reconstruction reconstruction;
  Image image;
  Camera camera;
  std::vector<Eigen::Vector2d> points2D;
  std::vector<Eigen::Vector3d> points3D;
};

AbsolutePoseProblem CreateAbsolutePoseTestData(
    CameraModelId camera_model_id = CameraModelId::kSimpleRadial,
    const std::vector<double>& camera_params = {1280, 512, 384, 0.05}) {
  AbsolutePoseProblem problem;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_model_id = camera_model_id;
  synthetic_dataset_options.camera_params = camera_params;
  SynthesizeDataset(synthetic_dataset_options, &problem.reconstruction);

  problem.image = problem.reconstruction.Image(1);
  problem.camera = *problem.image.CameraPtr();
  CHECK_EQ(problem.camera.model_id, camera_model_id);
  for (const auto& point2D : problem.image.Points2D()) {
    if (point2D.HasPoint3D()) {
      problem.points2D.push_back(point2D.xy);
      problem.points3D.push_back(
          problem.reconstruction.Point3D(point2D.point3D_id).xyz);
    }
  }

  return problem;
}

TEST(EstimateAbsolutePose, Nominal) {
  const AbsolutePoseProblem problem = CreateAbsolutePoseTestData();

  AbsolutePoseEstimationOptions options;
  Rigid3d cam_from_world;
  size_t num_inliers = 0;
  std::vector<char> inlier_mask;
  Camera camera = problem.camera;
  EXPECT_TRUE(EstimateAbsolutePose(options,
                                   problem.points2D,
                                   problem.points3D,
                                   &cam_from_world,
                                   &camera,
                                   &num_inliers,
                                   &inlier_mask));
  EXPECT_THAT(
      cam_from_world,
      Rigid3dNear(problem.image.CamFromWorld(), /*rtol=*/1e-6, /*ttol=*/1e-6));
  EXPECT_EQ(camera, problem.camera);
  EXPECT_EQ(num_inliers, problem.points2D.size());
  EXPECT_THAT(inlier_mask, testing::Each(testing::Eq(true)));
}

TEST(EstimateAbsolutePose, EstimateFocalLength) {
  const AbsolutePoseProblem problem = CreateAbsolutePoseTestData();

  AbsolutePoseEstimationOptions options;
  options.estimate_focal_length = true;
  Rigid3d cam_from_world;
  size_t num_inliers = 0;
  std::vector<char> inlier_mask;
  Camera camera = problem.camera;
  EXPECT_TRUE(EstimateAbsolutePose(options,
                                   problem.points2D,
                                   problem.points3D,
                                   &cam_from_world,
                                   &camera,
                                   &num_inliers,
                                   &inlier_mask));
  EXPECT_THAT(
      cam_from_world,
      Rigid3dNear(problem.image.CamFromWorld(), /*rtol=*/1e-3, /*ttol=*/1e-2));
  EXPECT_NEAR(camera.FocalLength(), problem.camera.FocalLength(), 5);
  camera.SetFocalLength(problem.camera.FocalLength());
  EXPECT_EQ(camera, problem.camera);
  EXPECT_EQ(num_inliers, problem.points2D.size());
  EXPECT_THAT(inlier_mask, testing::Each(testing::Eq(true)));
}

TEST(EstimateAbsolutePose, EstimateSeparateFocalLengths) {
  // PINHOLE camera with distinct fx and fy (fx, fy, cx, cy).
  const AbsolutePoseProblem problem = CreateAbsolutePoseTestData(
      CameraModelId::kPinhole, {1280, 1000, 512, 384});

  AbsolutePoseEstimationOptions options;
  options.estimate_focal_length = true;
  Rigid3d cam_from_world;
  size_t num_inliers = 0;
  std::vector<char> inlier_mask;
  Camera camera = problem.camera;
  EXPECT_TRUE(EstimateAbsolutePose(options,
                                   problem.points2D,
                                   problem.points3D,
                                   &cam_from_world,
                                   &camera,
                                   &num_inliers,
                                   &inlier_mask));
  EXPECT_THAT(
      cam_from_world,
      Rigid3dNear(problem.image.CamFromWorld(), /*rtol=*/1e-3, /*ttol=*/1e-2));
  EXPECT_NEAR(camera.FocalLengthX(), problem.camera.FocalLengthX(), 5);
  EXPECT_NEAR(camera.FocalLengthY(), problem.camera.FocalLengthY(), 5);
  EXPECT_EQ(num_inliers, problem.points2D.size());
  EXPECT_THAT(inlier_mask, testing::Each(testing::Eq(true)));
}

struct PinholeFocalProblem {
  std::vector<Eigen::Vector2d> points2D;  // centered on principal point
  std::vector<Eigen::Vector3d> points3D;
  Rigid3d gt_cam_from_world;
  double gt_focal;
};

// Synthetic pinhole scene with pixel noise and outliers. Deterministic for a
// fixed seed.
PinholeFocalProblem CreatePinholeFocalProblem(int seed,
                                              double noise_px,
                                              double outlier_ratio) {
  PinholeFocalProblem problem;
  problem.gt_focal = 1280.0;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uniform_px(-1200.0, 1200.0);
  std::uniform_real_distribution<double> uniform_depth(4.0, 12.0);
  std::uniform_real_distribution<double> uniform_trans(-6.0, 6.0);
  std::uniform_real_distribution<double> uniform_angle(0.0, 2 * EIGEN_PI);
  std::uniform_real_distribution<double> uniform_outlier(-1500.0, 1500.0);
  std::uniform_real_distribution<double> uniform_01(0.0, 1.0);
  std::normal_distribution<double> noise(0.0, noise_px);

  Eigen::Vector3d translation;
  do {
    translation = Eigen::Vector3d(
        uniform_trans(rng), uniform_trans(rng), uniform_trans(rng));
  } while (translation.norm() < 1.0);
  // Deterministic pseudo-random rotation (avoid Eigen::UnitRandom, which draws
  // from global RNG state).
  const Eigen::Quaterniond rotation =
      Eigen::AngleAxisd(uniform_angle(rng), Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(uniform_angle(rng), Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(uniform_angle(rng), Eigen::Vector3d::UnitZ());
  problem.gt_cam_from_world = Rigid3d(rotation, translation);

  const Rigid3d world_from_cam = Inverse(problem.gt_cam_from_world);
  for (int i = 0; i < 100; ++i) {
    const double depth = uniform_depth(rng);
    const Eigen::Vector2d projection(uniform_px(rng), uniform_px(rng));
    problem.points3D.push_back(
        world_from_cam *
        Eigen::Vector3d(projection.x() / problem.gt_focal * depth,
                        projection.y() / problem.gt_focal * depth,
                        depth));
    if (uniform_01(rng) < outlier_ratio) {
      problem.points2D.push_back(
          Eigen::Vector2d(uniform_outlier(rng), uniform_outlier(rng)));
    } else {
      problem.points2D.push_back(projection +
                                 Eigen::Vector2d(noise(rng), noise(rng)));
    }
  }
  return problem;
}

double RotationErrorDeg(const Rigid3d& a, const Rigid3d& b) {
  const Eigen::Matrix3d R_err = a.rotation().toRotationMatrix().transpose() *
                                b.rotation().toRotationMatrix();
  return std::acos(std::clamp((R_err.trace() - 1.0) / 2.0, -1.0, 1.0)) * 180.0 /
         EIGEN_PI;
}

TEST(EstimateAbsolutePose, EstimateFocalLengthShippedDefaults) {
  // Locks in that the shipped default RANSAC options achieve accurate focal
  // length estimation on noisy data with outliers. The multi-scene aggregate
  // with wide margins is robust to platform numerics, unlike any single
  // hand-picked seed. Passes with either trial floor value.
  const Camera init_camera = Camera::CreateFromModelId(
      kInvalidCameraId, CameraModelId::kSimplePinhole, 1280, 1024, 768);
  const Eigen::Vector2d principal_point(init_camera.PrincipalPointX(),
                                        init_camera.PrincipalPointY());
  int num_recalled = 0;
  for (int s = 0; s < 20; ++s) {
    const PinholeFocalProblem problem =
        CreatePinholeFocalProblem(s, /*noise_px=*/4.0, /*outlier_ratio=*/0.3);
    std::vector<Eigen::Vector2d> points2D = problem.points2D;
    for (auto& point : points2D) point += principal_point;

    AbsolutePoseEstimationOptions options;
    options.estimate_focal_length = true;
    options.ransac_options.random_seed = 1000 + s;
    Rigid3d cam_from_world;
    size_t num_inliers = 0;
    std::vector<char> inlier_mask;
    Camera camera = init_camera;
    if (!EstimateAbsolutePose(options,
                              points2D,
                              problem.points3D,
                              &cam_from_world,
                              &camera,
                              &num_inliers,
                              &inlier_mask)) {
      continue;
    }
    const double rot_err =
        RotationErrorDeg(cam_from_world, problem.gt_cam_from_world);
    const double focal_err =
        std::abs(camera.FocalLength() - problem.gt_focal) / problem.gt_focal;
    if (rot_err < 1.0 && focal_err < 0.02) ++num_recalled;
  }
  EXPECT_GE(num_recalled, 18);
}

TEST(EstimateRelativePose, Nominal) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 512.0, 1024, 1024);
  const Rigid3d cam1_from_world;
  const Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                                Eigen::Vector3d(1, 0.1, 0.2).normalized());

  // EstimateRelativePose scores with the pixel-unit tangent Sampson error, so
  // it takes rays with their unprojection Jacobians (CamRayWithJac). Points are
  // placed in front of both cameras and projected to pixels.
  std::vector<CamRayWithJac> cam_rays1_with_jac(100);
  std::vector<CamRayWithJac> cam_rays2_with_jac(100);
  for (size_t i = 0; i < cam_rays1_with_jac.size(); ++i) {
    const Eigen::Vector3d point3D =
        RandomEigenVectord<3>() + Eigen::Vector3d(0, 0, 3);
    const Eigen::Vector2d point2D1 =
        camera.ImgFromCam(cam1_from_world * point3D).value();
    const Eigen::Vector2d point2D2 =
        camera.ImgFromCam(cam2_from_world * point3D).value();
    cam_rays1_with_jac[i] = camera.CamRayFromImgWithJac(point2D1).value();
    cam_rays2_with_jac[i] = camera.CamRayFromImgWithJac(point2D2).value();
  }

  RANSACOptions options;
  options.max_error = 1.0;  // pixels
  Rigid3d cam2_from_cam1;
  size_t num_inliers = 0;
  std::vector<char> inlier_mask;
  EXPECT_TRUE(EstimateRelativePose(options,
                                   cam_rays1_with_jac,
                                   cam_rays2_with_jac,
                                   &cam2_from_cam1,
                                   &num_inliers,
                                   &inlier_mask));

  EXPECT_THAT(cam2_from_cam1,
              Rigid3dNear(cam2_from_world * Inverse(cam1_from_world),
                          /*rtol=*/1e-3,
                          /*ttol=*/1e-3));
  EXPECT_EQ(num_inliers, cam_rays1_with_jac.size());
  EXPECT_THAT(inlier_mask, testing::Each(testing::Eq(true)));
}

TEST(EstimateRelativePose, ZeroSentinelRaysExcluded) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 512.0, 1024, 1024);
  const Rigid3d cam1_from_world;
  const Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                                Eigen::Vector3d(1, 0.1, 0.2).normalized());

  std::vector<CamRayWithJac> cam_rays1_with_jac(100);
  std::vector<CamRayWithJac> cam_rays2_with_jac(100);
  for (size_t i = 0; i < cam_rays1_with_jac.size(); ++i) {
    const Eigen::Vector3d point3D =
        RandomEigenVectord<3>() + Eigen::Vector3d(0, 0, 3);
    cam_rays1_with_jac[i] =
        camera
            .CamRayFromImgWithJac(
                camera.ImgFromCam(cam1_from_world * point3D).value())
            .value();
    cam_rays2_with_jac[i] =
        camera
            .CamRayFromImgWithJac(
                camera.ImgFromCam(cam2_from_world * point3D).value())
            .value();
  }

  // Emulate unprojectable correspondences: CamRayFromImgWithJac returns nullopt
  // and callers substitute the CamRayWithJac::Zero() sentinel. These must be
  // rejected (infinite tangent Sampson residual) and never counted as inliers,
  // while the pose is still recovered from the remaining correspondences.
  std::vector<bool> is_zeroed(cam_rays1_with_jac.size(), false);
  for (const size_t i : {3, 17, 42, 88}) {
    cam_rays1_with_jac[i] = CamRayWithJac::Zero();
    cam_rays2_with_jac[i] = CamRayWithJac::Zero();
    is_zeroed[i] = true;
  }

  RANSACOptions options;
  options.max_error = 1.0;  // pixels
  Rigid3d cam2_from_cam1;
  size_t num_inliers = 0;
  std::vector<char> inlier_mask;
  EXPECT_TRUE(EstimateRelativePose(options,
                                   cam_rays1_with_jac,
                                   cam_rays2_with_jac,
                                   &cam2_from_cam1,
                                   &num_inliers,
                                   &inlier_mask));

  EXPECT_THAT(cam2_from_cam1,
              Rigid3dNear(cam2_from_world * Inverse(cam1_from_world),
                          /*rtol=*/1e-3,
                          /*ttol=*/1e-3));
  EXPECT_EQ(num_inliers, cam_rays1_with_jac.size() - 4);
  ASSERT_EQ(inlier_mask.size(), is_zeroed.size());
  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    EXPECT_EQ(static_cast<bool>(inlier_mask[i]), !is_zeroed[i]);
  }
}

TEST(RefineAbsolutePose, Nominal) {
  const AbsolutePoseProblem problem = CreateAbsolutePoseTestData();
  std::vector<char> inlier_mask(problem.points2D.size(), true);

  AbsolutePoseRefinementOptions options;
  Rigid3d cam_from_world = problem.image.CamFromWorld();
  cam_from_world =
      cam_from_world * Rigid3d(Eigen::Quaterniond(Eigen::AngleAxisd(
                                   0.1, RandomEigenVectord<3>())),
                               0.1 * RandomEigenVectord<3>());
  Camera camera = problem.camera;
  Eigen::Matrix6d cam_from_world_cov = Eigen::Matrix6d::Zero();
  EXPECT_TRUE(RefineAbsolutePose(options,
                                 inlier_mask,
                                 problem.points2D,
                                 problem.points3D,
                                 &cam_from_world,
                                 &camera,
                                 &cam_from_world_cov));
  EXPECT_THAT(
      cam_from_world,
      Rigid3dNear(problem.image.CamFromWorld(), /*rtol=*/1e-6, /*ttol=*/1e-6));
  EXPECT_NEAR(cam_from_world.rotation().norm(), 1.0, 1e-6);
  EXPECT_EQ(camera, problem.camera);
  EXPECT_NE(cam_from_world_cov, Eigen::Matrix6d::Zero());
}

TEST(RefineAbsolutePose, RefineFocalLength) {
  const AbsolutePoseProblem problem = CreateAbsolutePoseTestData();
  std::vector<char> inlier_mask(problem.points2D.size(), true);

  AbsolutePoseRefinementOptions options;
  options.refine_focal_length = true;
  Rigid3d cam_from_world = problem.image.CamFromWorld();
  Camera camera = problem.camera;
  camera.SetFocalLength(0.9 * camera.FocalLength());
  Eigen::Matrix6d cam_from_world_cov = Eigen::Matrix6d::Zero();
  EXPECT_TRUE(RefineAbsolutePose(options,
                                 inlier_mask,
                                 problem.points2D,
                                 problem.points3D,
                                 &cam_from_world,
                                 &camera,
                                 &cam_from_world_cov));
  EXPECT_THAT(
      cam_from_world,
      Rigid3dNear(problem.image.CamFromWorld(), /*rtol=*/1e-3, /*ttol=*/1e-3));
  EXPECT_NEAR(cam_from_world.rotation().norm(), 1.0, 1e-6);
  EXPECT_NEAR(camera.FocalLength(), problem.camera.FocalLength(), 5);
  camera.SetFocalLength(problem.camera.FocalLength());
  EXPECT_EQ(camera, problem.camera);
  EXPECT_NE(cam_from_world_cov, Eigen::Matrix6d::Zero());
}

TEST(RefineAbsolutePose, RefineExtraParams) {
  const AbsolutePoseProblem problem = CreateAbsolutePoseTestData();
  std::vector<char> inlier_mask(problem.points2D.size(), true);

  AbsolutePoseRefinementOptions options;
  options.refine_extra_params = true;
  Rigid3d cam_from_world = problem.image.CamFromWorld();
  Camera camera = problem.camera;
  camera.params.at(3) += 0.1;
  Eigen::Matrix6d cam_from_world_cov = Eigen::Matrix6d::Zero();
  EXPECT_TRUE(RefineAbsolutePose(options,
                                 inlier_mask,
                                 problem.points2D,
                                 problem.points3D,
                                 &cam_from_world,
                                 &camera,
                                 &cam_from_world_cov));
  EXPECT_THAT(
      cam_from_world,
      Rigid3dNear(problem.image.CamFromWorld(), /*rtol=*/1e-3, /*ttol=*/1e-3));
  EXPECT_NEAR(cam_from_world.rotation().norm(), 1.0, 1e-6);
  EXPECT_NEAR(camera.params.at(3), problem.camera.params.at(3), 1e-3);
  camera.params.at(3) = problem.camera.params.at(3);
  EXPECT_EQ(camera, problem.camera);
  EXPECT_NE(cam_from_world_cov, Eigen::Matrix6d::Zero());
}

TEST(RefineAbsolutePose, PositionPrior) {
  const AbsolutePoseProblem problem = CreateAbsolutePoseTestData();
  // Isolate the position-prior-only refinement path without reprojection terms.
  std::vector<char> inlier_mask(problem.points2D.size(), false);

  AbsolutePoseRefinementOptions options;
  options.use_position_prior = true;
  options.position_prior_in_world = Eigen::Vector3d(1.0, 2.0, 3.0);
  options.position_prior_covariance = Eigen::Matrix3d::Identity();
  Rigid3d cam_from_world(
      Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY())),
      Eigen::Vector3d(0.3, -0.5, 0.7));
  auto compute_position_error = [&](const Rigid3d& cam_from_world_to_check) {
    return (Inverse(cam_from_world_to_check).translation() -
            options.position_prior_in_world)
        .norm();
  };
  const double initial_error = compute_position_error(cam_from_world);
  Camera camera = problem.camera;
  EXPECT_TRUE(RefineAbsolutePose(options,
                                 inlier_mask,
                                 problem.points2D,
                                 problem.points3D,
                                 &cam_from_world,
                                 &camera));
  EXPECT_LT(compute_position_error(cam_from_world), initial_error);
  EXPECT_NEAR(cam_from_world.rotation().norm(), 1.0, 1e-6);
}

TEST(RefineAbsolutePose, PositionPriorCovariance) {
  const AbsolutePoseProblem problem = CreateAbsolutePoseTestData();
  std::vector<char> inlier_mask(problem.points2D.size(), true);

  AbsolutePoseRefinementOptions weak_prior_options;
  weak_prior_options.use_position_prior = true;
  weak_prior_options.position_prior_in_world =
      Inverse(problem.image.CamFromWorld()).translation() +
      Eigen::Vector3d(1.0, -0.7, 0.5);
  // Large covariance = weak prior (high uncertainty).
  weak_prior_options.position_prior_covariance = Eigen::Matrix3d::Identity();

  AbsolutePoseRefinementOptions strong_prior_options = weak_prior_options;
  // Small covariance = strong prior (low uncertainty).
  strong_prior_options.position_prior_covariance =
      0.0001 * Eigen::Matrix3d::Identity();

  const Rigid3d initial_cam_from_world(
      Eigen::Quaterniond(Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX())),
      problem.image.CamFromWorld().translation() +
          Eigen::Vector3d(0.2, 0.1, -0.1));
  Camera weak_prior_camera = problem.camera;
  Camera strong_prior_camera = problem.camera;
  Rigid3d weak_prior_cam_from_world = initial_cam_from_world;
  Rigid3d strong_prior_cam_from_world = initial_cam_from_world;

  EXPECT_TRUE(RefineAbsolutePose(weak_prior_options,
                                 inlier_mask,
                                 problem.points2D,
                                 problem.points3D,
                                 &weak_prior_cam_from_world,
                                 &weak_prior_camera));
  EXPECT_NEAR(weak_prior_cam_from_world.rotation().norm(), 1.0, 1e-6);
  EXPECT_TRUE(RefineAbsolutePose(strong_prior_options,
                                 inlier_mask,
                                 problem.points2D,
                                 problem.points3D,
                                 &strong_prior_cam_from_world,
                                 &strong_prior_camera));
  EXPECT_NEAR(strong_prior_cam_from_world.rotation().norm(), 1.0, 1e-6);

  auto compute_position_error = [&](const Rigid3d& cam_from_world_to_check) {
    return (Inverse(cam_from_world_to_check).translation() -
            weak_prior_options.position_prior_in_world)
        .norm();
  };
  const double weak_prior_error =
      compute_position_error(weak_prior_cam_from_world);
  const double strong_prior_error =
      compute_position_error(strong_prior_cam_from_world);
  EXPECT_LT(strong_prior_error, weak_prior_error);
}

TEST(RefineEssentialMatrix, Nominal) {
  const Rigid3d cam1_from_world;
  const Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                                Eigen::Vector3d(1, 0, 0).normalized());
  const Eigen::Matrix3d E =
      EssentialMatrixFromPose(cam2_from_world * Inverse(cam1_from_world));

  std::vector<Eigen::Vector3d> points3D(150);
  for (size_t i = 0; i < points3D.size() / 3; ++i) {
    points3D[3 * i + 0] = Eigen::Vector3d(i * 0.01, 0, 1);
    points3D[3 * i + 1] = Eigen::Vector3d(0, i * 0.01, 1);
    points3D[3 * i + 2] = Eigen::Vector3d(i * 0.01, i * 0.01, 1);
  }

  // Score with the pixel-unit tangent Sampson error, so each ray carries its
  // unprojection Jacobian. A spherical camera keeps every direction valid.
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, /*focal_length=*/0.0, 1000, 500);
  std::vector<CamRayWithJac> cam_rays1_with_jac(points3D.size());
  std::vector<CamRayWithJac> cam_rays2_with_jac(points3D.size());
  for (size_t i = 0; i < points3D.size(); ++i) {
    const Eigen::Vector3d ray1 = (cam1_from_world * points3D[i]).normalized();
    const Eigen::Vector3d ray2 = (cam2_from_world * points3D[i]).normalized();
    cam_rays1_with_jac[i] =
        camera.CamRayFromImgWithJac(camera.ImgFromCam(ray1).value()).value();
    cam_rays2_with_jac[i] =
        camera.CamRayFromImgWithJac(camera.ImgFromCam(ray2).value()).value();
  }

  const Rigid3d cam2_from_world_perturbed(
      Eigen::Quaterniond::Identity(),
      Eigen::Vector3d(1.02, 0.02, 0.01).normalized());
  const Eigen::Matrix3d E_perturbed = EssentialMatrixFromPose(
      cam2_from_world_perturbed * Inverse(cam1_from_world));

  Eigen::Matrix3d E_refined = E_perturbed;
  ceres::Solver::Options options;
  RefineEssentialMatrix(options,
                        cam_rays1_with_jac,
                        cam_rays2_with_jac,
                        std::vector<char>(cam_rays1_with_jac.size(), true),
                        &E_refined);

  EXPECT_LE((E - E_refined).norm(), (E - E_perturbed).norm());
}

}  // namespace
}  // namespace colmap

// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/triangulation.h"

#include <random>

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace colmap {
namespace {

Camera TestCamera() {
  return Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 512.0, 1024, 1024);
}

// Three cameras with identity rotation at different centers, all observing
// points around (0, 0, 5).
std::vector<Rigid3d> TestCamsFromWorld() {
  return {
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(0, 0, 0)),
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(-2, 0, 0)),
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(0, -2, 0)),
  };
}

std::vector<Eigen::Vector2d> ProjectPoint(
    const Eigen::Vector3d& xyz,
    const std::vector<Rigid3d>& cams_from_world,
    const Camera& camera) {
  std::vector<Eigen::Vector2d> points2D;
  for (const Rigid3d& cam_from_world : cams_from_world) {
    points2D.push_back(camera.ImgFromCam(cam_from_world * xyz).value());
  }
  return points2D;
}

TEST(PropagatePoseCovarianceToImage, MatchesNumericJacobian) {
  const Camera camera = TestCamera();
  const Rigid3d cam_from_world(
      Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d(0, 1, 0))),
      Eigen::Vector3d(0.5, -0.2, 0.1));
  const Eigen::Vector3d xyz(0.3, -0.4, 5.0);
  const Eigen::Vector3d point3D_in_cam = cam_from_world * xyz;

  Eigen::Matrix2x3d J_proj;
  ASSERT_TRUE(camera.ImgFromCamWithJac(point3D_in_cam, &J_proj).has_value());

  // Numeric Jacobian of the projection w.r.t. the Ceres
  // EigenQuaternionManifold tangent: the delta quaternion left-multiplies the
  // current rotation and its vector is half the physical rotation angle.
  const auto project_perturbed = [&](const Eigen::Matrix<double, 6, 1>& delta) {
    const double delta_norm = delta.head<3>().norm();
    Eigen::Quaterniond delta_q = Eigen::Quaterniond::Identity();
    if (delta_norm > 0.0) {
      const Eigen::Vector3d delta_imag =
          std::sin(delta_norm) * delta.head<3>() / delta_norm;
      delta_q = Eigen::Quaterniond(
          std::cos(delta_norm), delta_imag.x(), delta_imag.y(), delta_imag.z());
    }
    const Eigen::Quaterniond q_pert = delta_q * cam_from_world.rotation();
    const Eigen::Vector3d t_pert =
        cam_from_world.translation() + delta.tail<3>();
    return camera.ImgFromCam(q_pert * xyz + t_pert).value();
  };
  Eigen::Matrix<double, 2, 6> J_numeric;
  const double h = 1e-7;
  for (int j = 0; j < 6; ++j) {
    Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
    delta[j] = h;
    J_numeric.col(j) =
        (project_perturbed(delta) - project_perturbed(-delta)) / (2 * h);
  }

  // Compare propagated covariances for a random SPD pose covariance.
  std::mt19937 rng(42);
  std::normal_distribution<double> normal;
  Eigen::Matrix6d A;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      A(i, j) = normal(rng);
    }
  }
  const Eigen::Matrix6d pose_cov =
      1e-4 * (A * A.transpose() + Eigen::Matrix6d::Identity());
  const Eigen::Matrix2d propagated = PropagatePoseCovarianceToImage(
      cam_from_world.rotation().toRotationMatrix(), xyz, J_proj, pose_cov);
  const Eigen::Matrix2d expected = J_numeric * pose_cov * J_numeric.transpose();
  EXPECT_TRUE(propagated.allFinite());
  EXPECT_LT((propagated - expected).norm(), 1e-6 * expected.norm());
}

TEST(PropagatePoseCovarianceToImage, ZeroCovariancePropagatesToZero) {
  EXPECT_EQ(PropagatePoseCovarianceToImage(Eigen::Matrix3d::Identity(),
                                           Eigen::Vector3d(0, 0, 5),
                                           Eigen::Matrix2x3d::Identity(),
                                           Eigen::Matrix6d::Zero()),
            Eigen::Matrix2d::Zero());
}

TEST(EstimateCovariantTriangulation, NominalNoiseless) {
  const Camera camera = TestCamera();
  const std::vector<Rigid3d> cams_from_world = TestCamsFromWorld();
  const std::vector<const Camera*> cameras(3, &camera);
  const Eigen::Vector3d xyz_true(0.3, -0.4, 5.0);
  const std::vector<Eigen::Vector2d> points2D =
      ProjectPoint(xyz_true, cams_from_world, camera);

  CovariantTriangulationOptions options;
  options.ransac_options.random_seed = 0;
  std::vector<char> inlier_mask;
  Eigen::Vector3d xyz;
  Eigen::Matrix3d xyz_cov;
  EXPECT_TRUE(EstimateCovariantTriangulation(options,
                                             points2D,
                                             /*points2D_cov=*/{},
                                             cams_from_world,
                                             /*pose_covs=*/{},
                                             cameras,
                                             &inlier_mask,
                                             &xyz,
                                             &xyz_cov));
  EXPECT_EQ(inlier_mask, std::vector<char>({1, 1, 1}));
  EXPECT_LT((xyz - xyz_true).norm(), 1e-8);
  EXPECT_GT(xyz_cov.determinant(), 0.0);

  // Agrees with the plain estimator on easy data.
  EstimateTriangulationOptions plain_options;
  plain_options.ransac_options.random_seed = 0;
  std::vector<char> plain_inlier_mask;
  Eigen::Vector3d plain_xyz;
  EXPECT_TRUE(EstimateTriangulation(plain_options,
                                    points2D,
                                    cams_from_world,
                                    cameras,
                                    &plain_inlier_mask,
                                    &plain_xyz));
  EXPECT_LT((xyz - plain_xyz).norm(), 1e-8);
}

TEST(EstimateCovariantTriangulation, KnownNoiseCalibration) {
  const Camera camera = TestCamera();
  const std::vector<Rigid3d> cams_from_world = TestCamsFromWorld();
  const std::vector<const Camera*> cameras(3, &camera);
  const Eigen::Vector3d xyz_true(0.3, -0.4, 5.0);
  const std::vector<Eigen::Vector2d> points2D_exact =
      ProjectPoint(xyz_true, cams_from_world, camera);
  // Unit pixel noise.
  const std::vector<Eigen::Matrix2d> points2D_cov(3,
                                                  Eigen::Matrix2d::Identity());

  CovariantTriangulationOptions options;
  options.ransac_options.random_seed = 0;

  // With a 99% gate over 3 observations, ~97% of trials should keep all
  // inliers; the NEES should average ~3 (3 DoF).
  std::mt19937 rng(7);
  std::normal_distribution<double> normal;
  const int kNumTrials = 200;
  int num_all_inliers = 0;
  int num_success = 0;
  double nees_sum = 0.0;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    std::vector<Eigen::Vector2d> points2D = points2D_exact;
    for (Eigen::Vector2d& p : points2D) {
      p.x() += normal(rng);
      p.y() += normal(rng);
    }
    std::vector<char> inlier_mask;
    Eigen::Vector3d xyz;
    Eigen::Matrix3d xyz_cov;
    if (!EstimateCovariantTriangulation(options,
                                        points2D,
                                        points2D_cov,
                                        cams_from_world,
                                        /*pose_covs=*/{},
                                        cameras,
                                        &inlier_mask,
                                        &xyz,
                                        &xyz_cov)) {
      continue;
    }
    ++num_success;
    if (inlier_mask == std::vector<char>({1, 1, 1})) {
      ++num_all_inliers;
    }
    const Eigen::Vector3d error = xyz - xyz_true;
    nees_sum += error.transpose() * xyz_cov.inverse() * error;
  }
  EXPECT_GT(num_success, 0.95 * kNumTrials);
  const double all_inlier_rate =
      static_cast<double>(num_all_inliers) / kNumTrials;
  EXPECT_GT(all_inlier_rate, 0.94);
  EXPECT_LT(all_inlier_rate, 1.0);
  EXPECT_GT(nees_sum / num_success, 2.0);
  EXPECT_LT(nees_sum / num_success, 4.5);
}

TEST(EstimateCovariantTriangulation, WithPoseCovariance) {
  const Camera camera = TestCamera();
  const std::vector<Rigid3d> cams_from_world = TestCamsFromWorld();
  const std::vector<const Camera*> cameras(3, &camera);
  const Eigen::Vector3d xyz_true(0.3, -0.4, 5.0);
  const std::vector<Eigen::Vector2d> points2D =
      ProjectPoint(xyz_true, cams_from_world, camera);
  const std::vector<Eigen::Matrix2d> points2D_cov(3,
                                                  Eigen::Matrix2d::Identity());

  CovariantTriangulationOptions options;
  options.ransac_options.random_seed = 0;
  std::vector<char> inlier_mask_exact, inlier_mask_uncertain;
  Eigen::Vector3d xyz_exact, xyz_uncertain;
  Eigen::Matrix3d cov_exact, cov_uncertain;
  EXPECT_TRUE(EstimateCovariantTriangulation(options,
                                             points2D,
                                             points2D_cov,
                                             cams_from_world,
                                             /*pose_covs=*/{},
                                             cameras,
                                             &inlier_mask_exact,
                                             &xyz_exact,
                                             &cov_exact));

  Eigen::Matrix6d pose_cov = Eigen::Matrix6d::Identity();
  pose_cov.topLeftCorner<3, 3>() *= 1e-6;
  pose_cov.bottomRightCorner<3, 3>() *= 1e-4;
  EXPECT_TRUE(
      EstimateCovariantTriangulation(options,
                                     points2D,
                                     points2D_cov,
                                     cams_from_world,
                                     std::vector<Eigen::Matrix6d>(3, pose_cov),
                                     cameras,
                                     &inlier_mask_uncertain,
                                     &xyz_uncertain,
                                     &cov_uncertain));

  EXPECT_LT((xyz_uncertain - xyz_true).norm(), 1e-6);
  // Pose uncertainty inflates the point covariance.
  EXPECT_GT(cov_uncertain.trace(), cov_exact.trace());
}

TEST(EstimateCovariantTriangulation, TwoViewIsRefined) {
  const Camera camera = TestCamera();
  const std::vector<Rigid3d> cams_from_world = {TestCamsFromWorld()[0],
                                                TestCamsFromWorld()[1]};
  const std::vector<const Camera*> cameras(2, &camera);
  std::vector<Eigen::Vector2d> points2D =
      ProjectPoint(Eigen::Vector3d(0.3, -0.4, 5.0), cams_from_world, camera);
  points2D[0] += Eigen::Vector2d(0.8, -0.5);
  points2D[1] += Eigen::Vector2d(-0.6, 0.7);
  // Anisotropic covariances, under which the covariance-agnostic DLT seed is
  // not the maximum likelihood estimate.
  const std::vector<Eigen::Matrix2d> points2D_cov = {
      Eigen::Vector2d(4.0, 0.25).asDiagonal(),
      Eigen::Vector2d(0.25, 4.0).asDiagonal()};

  CovariantTriangulationOptions options;
  options.ransac_options.random_seed = 0;
  std::vector<char> inlier_mask;
  Eigen::Vector3d xyz;
  Eigen::Matrix3d xyz_cov;
  ASSERT_TRUE(EstimateCovariantTriangulation(options,
                                             points2D,
                                             points2D_cov,
                                             cams_from_world,
                                             /*pose_covs=*/{},
                                             cameras,
                                             &inlier_mask,
                                             &xyz,
                                             &xyz_cov));
  EXPECT_EQ(inlier_mask, std::vector<char>({1, 1}));

  std::vector<CovariantTriangulationEstimator::PointData> point_data(2);
  std::vector<CovariantTriangulationEstimator::PoseData> pose_data(2);
  for (size_t i = 0; i < 2; ++i) {
    point_data[i].img_point = points2D[i];
    point_data[i].cam_ray = camera.CamRayFromImg(points2D[i]).value();
    point_data[i].img_cov = points2D_cov[i];
    pose_data[i].cam_from_world = cams_from_world[i].ToMatrix();
    pose_data[i].camera = &camera;
  }

  // The result is a stationary point of the whitened reprojection cost and
  // thus differs from the DLT seed.
  Eigen::Vector3d refined_xyz = xyz;
  ASSERT_TRUE(CovariantTriangulationEstimator::Refine(
      point_data, pose_data, &refined_xyz));
  EXPECT_LT((refined_xyz - xyz).norm(), 1e-6);
  std::vector<Eigen::Vector3d> seed_xyz;
  CovariantTriangulationEstimator::Estimate(point_data, pose_data, &seed_xyz);
  ASSERT_EQ(seed_xyz.size(), 1);
  EXPECT_GT((seed_xyz[0] - xyz).norm(), 1e-4);
}

TEST(MaxRelativeDepthUncertainty, Nominal) {
  const std::vector<Rigid3d> cams_from_world = TestCamsFromWorld();
  const Eigen::Vector3d xyz(0, 0, 5);
  const Eigen::Matrix3d xyz_cov =
      Eigen::Vector3d(0, 0, 0.01).asDiagonal().toDenseMatrix();
  // The camera at the origin looks along the uncertain axis.
  EXPECT_NEAR(MaxRelativeDepthUncertainty(xyz, xyz_cov, cams_from_world),
              0.1 / 5,
              1e-12);
  EXPECT_NEAR(MaxRelativeDepthUncertainty(xyz, xyz_cov, {cams_from_world[1]}),
              0.1 * 5 / 29,
              1e-12);
  EXPECT_EQ(MaxRelativeDepthUncertainty(xyz, xyz_cov, {}), 0);
  EXPECT_EQ(MaxRelativeDepthUncertainty(
                Eigen::Vector3d::Zero(), xyz_cov, cams_from_world),
            std::numeric_limits<double>::infinity());
}

TEST(EstimateCovariantTriangulation, RejectsSmallAngle) {
  const Camera camera = TestCamera();
  // Near-zero baseline: exact observations triangulate fine, but the depth
  // uncertainty gate must reject the triangulation.
  const std::vector<Rigid3d> cams_from_world = {
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(0, 0, 0)),
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(-1e-4, 0, 0)),
  };
  const std::vector<const Camera*> cameras(2, &camera);
  const std::vector<Eigen::Vector2d> points2D =
      ProjectPoint(Eigen::Vector3d(0, 0, 5), cams_from_world, camera);

  CovariantTriangulationOptions options;
  options.ransac_options.random_seed = 0;
  std::vector<char> inlier_mask;
  Eigen::Vector3d xyz;
  Eigen::Matrix3d xyz_cov;
  EXPECT_FALSE(EstimateCovariantTriangulation(options,
                                              points2D,
                                              /*points2D_cov=*/{},
                                              cams_from_world,
                                              /*pose_covs=*/{},
                                              cameras,
                                              &inlier_mask,
                                              &xyz,
                                              &xyz_cov));
}

TEST(EstimateCovariantTriangulation, RejectsBehindCamera) {
  const Camera camera = TestCamera();
  // Diverging rays whose DLT intersection lies behind both cameras.
  const std::vector<Rigid3d> cams_from_world = {
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(0, 0, 0)),
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(-10, 0, 0)),
  };
  const std::vector<const Camera*> cameras(2, &camera);
  const std::vector<Eigen::Vector2d> points2D = {Eigen::Vector2d(0, 512),
                                                 Eigen::Vector2d(1024, 512)};

  CovariantTriangulationOptions options;
  options.ransac_options.random_seed = 0;
  std::vector<char> inlier_mask;
  Eigen::Vector3d xyz;
  Eigen::Matrix3d xyz_cov;
  EXPECT_FALSE(EstimateCovariantTriangulation(options,
                                              points2D,
                                              /*points2D_cov=*/{},
                                              cams_from_world,
                                              /*pose_covs=*/{},
                                              cameras,
                                              &inlier_mask,
                                              &xyz,
                                              &xyz_cov));
}

TEST(EstimateCovariantTriangulation, EquirectangularBackHemisphere) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, 0.0, 1024, 512);
  const std::vector<Rigid3d> cams_from_world = {
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero()),
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(-2, 0, 0)),
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(0, -2, 0)),
  };
  const std::vector<const Camera*> cameras(3, &camera);
  const Eigen::Vector3d xyz_true(0.3, -0.4, -5.0);
  const std::vector<Eigen::Vector2d> points2D =
      ProjectPoint(xyz_true, cams_from_world, camera);

  CovariantTriangulationOptions options;
  options.ransac_options.random_seed = 0;
  std::vector<char> inlier_mask;
  Eigen::Vector3d xyz;
  Eigen::Matrix3d xyz_cov;
  EXPECT_TRUE(EstimateCovariantTriangulation(options,
                                             points2D,
                                             /*points2D_cov=*/{},
                                             cams_from_world,
                                             /*pose_covs=*/{},
                                             cameras,
                                             &inlier_mask,
                                             &xyz,
                                             &xyz_cov));
  EXPECT_EQ(inlier_mask, std::vector<char>({1, 1, 1}));
  EXPECT_LT((xyz - xyz_true).norm(), 1e-8);
}

TEST(EstimateCovariantTriangulation, EmptyCovarianceDefaults) {
  const Camera camera = TestCamera();
  const std::vector<Rigid3d> cams_from_world = TestCamsFromWorld();
  const std::vector<const Camera*> cameras(3, &camera);
  const std::vector<Eigen::Vector2d> points2D =
      ProjectPoint(Eigen::Vector3d(0.3, -0.4, 5.0), cams_from_world, camera);

  CovariantTriangulationOptions options;
  options.ransac_options.random_seed = 0;
  std::vector<char> mask_empty, mask_explicit;
  Eigen::Vector3d xyz_empty, xyz_explicit;
  Eigen::Matrix3d cov_empty, cov_explicit;
  EXPECT_TRUE(EstimateCovariantTriangulation(options,
                                             points2D,
                                             /*points2D_cov=*/{},
                                             cams_from_world,
                                             /*pose_covs=*/{},
                                             cameras,
                                             &mask_empty,
                                             &xyz_empty,
                                             &cov_empty));
  EXPECT_TRUE(EstimateCovariantTriangulation(
      options,
      points2D,
      std::vector<Eigen::Matrix2d>(3, Eigen::Matrix2d::Identity()),
      cams_from_world,
      std::vector<Eigen::Matrix6d>(3, Eigen::Matrix6d::Zero()),
      cameras,
      &mask_explicit,
      &xyz_explicit,
      &cov_explicit));
  EXPECT_EQ(mask_empty, mask_explicit);
  EXPECT_EQ(xyz_empty, xyz_explicit);
  EXPECT_EQ(cov_empty, cov_explicit);
}

}  // namespace
}  // namespace colmap

// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/solvers/absolute_pose.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/optim/loransac.h"
#include "colmap/optim/ransac.h"
#include "colmap/scene/camera.h"
#include "colmap/sensor/models.h"
#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(AbsolutePose, P3P) {
  const std::vector<Eigen::Vector3d> points3D = {
      Eigen::Vector3d(1, 1, 1),
      Eigen::Vector3d(0, 1, 1),
      Eigen::Vector3d(3, 1.0, 4),
      Eigen::Vector3d(3, 1.1, 4),
      Eigen::Vector3d(3, 1.2, 4),
      Eigen::Vector3d(3, 1.3, 4),
      Eigen::Vector3d(3, 1.4, 4),
      Eigen::Vector3d(2, 1, 7),
  };

  auto points3D_faulty = points3D;
  for (size_t i = 0; i < points3D.size(); ++i) {
    points3D_faulty[i](0) = 20;
  }

  const Camera camera = Camera::CreateFromModelId(
      kInvalidCameraId, CameraModelId::kPinhole, 12, 34, 56);
  ImgFromCamFunc img_from_cam_func =
      [&camera](const Eigen::Vector3d& cam_point) {
        return camera.ImgFromCam(cam_point);
      };

  // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
  for (double qx = 0; qx < 1; qx += 0.2) {
    // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
    for (double tx = 0; tx < 1; tx += 0.1) {
      const Rigid3d expected_cam_from_world(
          Eigen::Quaterniond(1, qx, 0, 0).normalized(),
          Eigen::Vector3d(tx, 0, 0));

      // Project points to camera coordinate system.
      std::vector<P3PEstimator::X_t> points2D;
      for (size_t i = 0; i < points3D.size(); ++i) {
        auto& point2D = points2D.emplace_back();
        point2D.camera_ray =
            (expected_cam_from_world * points3D[i]).normalized();
        point2D.image_point = img_from_cam_func(point2D.camera_ray).value();
      }

      RANSACOptions options;
      options.max_error = 1e-3;
      RANSAC<P3PEstimator> ransac(options, P3PEstimator(img_from_cam_func));
      const auto report = ransac.Estimate(points2D, points3D);

      EXPECT_TRUE(report.success);
      EXPECT_THAT(report.model,
                  Rigid3dNear(expected_cam_from_world,
                              /*rtol=*/1e-6,
                              /*ttol=*/1e-6));

      // Test residuals of exact points.
      std::vector<double> residuals;
      ransac.estimator.Residuals(points2D, points3D, report.model, &residuals);
      for (size_t i = 0; i < residuals.size(); ++i) {
        EXPECT_LT(residuals[i], 1e-3);
      }

      // Test residuals of faulty points.
      ransac.estimator.Residuals(
          points2D, points3D_faulty, report.model, &residuals);
      for (size_t i = 0; i < residuals.size(); ++i) {
        EXPECT_GT(residuals[i], 0.1);
      }
    }
  }
}

TEST(AbsolutePose, P4PFSharedFocalLength) {
  const std::vector<Eigen::Vector3d> points3D = {
      Eigen::Vector3d(1, 1, 1),
      Eigen::Vector3d(0, 1, 1),
      Eigen::Vector3d(3, 1.0, 4),
      Eigen::Vector3d(3, 1.1, 4),
      Eigen::Vector3d(3, 1.2, 4),
      Eigen::Vector3d(3, 1.3, 4),
      Eigen::Vector3d(3, 1.4, 4),
      Eigen::Vector3d(2, 1, 7),
  };

  auto points3D_faulty = points3D;
  for (size_t i = 0; i < points3D.size(); ++i) {
    points3D_faulty[i](0) = 20;
  }

  // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
  for (double qx = 0; qx < 1; qx += 0.2) {
    // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
    for (double tx = 0; tx < 1; tx += 0.1) {
      // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
      for (double f = 0.5; f < 20; f += 2) {
        const Rigid3d expected_cam_from_world(
            Eigen::Quaterniond(1, qx, 0, 0).normalized(),
            Eigen::Vector3d(tx, 0, 0));

        // Project points to camera coordinate system.
        std::vector<Eigen::Vector2d> points2D;
        points2D.reserve(points3D.size());
        for (size_t i = 0; i < points3D.size(); ++i) {
          points2D.push_back(
              f * (expected_cam_from_world * points3D[i]).hnormalized());
        }

        RANSACOptions options;
        options.max_error = 1e-5;
        LORANSAC<P4PFEstimator, P4PFEstimator> ransac(
            options, P4PFEstimator(), P4PFEstimator());
        const auto report = ransac.Estimate(points2D, points3D);

        EXPECT_TRUE(report.success);
        // In shared mode both focal lengths must be identical.
        EXPECT_EQ(report.model.focal_lengths.x(),
                  report.model.focal_lengths.y());
        EXPECT_NEAR(report.model.focal_lengths.x(), f, 1e-3);
        EXPECT_THAT(report.model.cam_from_world,
                    Rigid3dNear(expected_cam_from_world,
                                /*rtol=*/1e-6,
                                /*ttol=*/1e-6));

        // Test residuals of exact points.
        std::vector<double> residuals;
        P4PFEstimator::Residuals(points2D, points3D, report.model, &residuals);
        for (size_t i = 0; i < residuals.size(); ++i) {
          EXPECT_LT(residuals[i], 1e-3);
        }

        // Test residuals of faulty points.
        P4PFEstimator::Residuals(
            points2D, points3D_faulty, report.model, &residuals);
        for (size_t i = 0; i < residuals.size(); ++i) {
          EXPECT_GT(residuals[i], 0.1);
        }
      }
    }
  }
}

TEST(AbsolutePose, P4PFSeparateFocalLengths) {
  const std::vector<Eigen::Vector3d> points3D = {
      Eigen::Vector3d(1, 1, 1),
      Eigen::Vector3d(0, 1, 1),
      Eigen::Vector3d(3, 1.0, 4),
      Eigen::Vector3d(3, 1.1, 4),
      Eigen::Vector3d(3, 1.2, 4),
      Eigen::Vector3d(3, 1.3, 4),
      Eigen::Vector3d(3, 1.4, 4),
      Eigen::Vector3d(2, 1, 7),
  };

  auto points3D_faulty = points3D;
  for (size_t i = 0; i < points3D.size(); ++i) {
    points3D_faulty[i](0) = 20;
  }

  // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
  for (double qx = 0; qx < 1; qx += 0.2) {
    // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
    for (double tx = 0; tx < 1; tx += 0.1) {
      // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
      for (double f = 0.5; f < 20; f += 2) {
        const Eigen::Vector2d focal_lengths(f, 1.5 * f);
        const Rigid3d expected_cam_from_world(
            Eigen::Quaterniond(1, qx, 0, 0).normalized(),
            Eigen::Vector3d(tx, 0, 0));

        // Project points using separate focal lengths for x and y.
        std::vector<Eigen::Vector2d> points2D;
        points2D.reserve(points3D.size());
        for (size_t i = 0; i < points3D.size(); ++i) {
          points2D.push_back(focal_lengths.cwiseProduct(
              (expected_cam_from_world * points3D[i]).hnormalized()));
        }

        RANSACOptions options;
        options.max_error = 1e-5;
        LORANSAC<P4PFEstimator, P4PFEstimator> ransac(
            options,
            P4PFEstimator(/*share_focal_length=*/false),
            P4PFEstimator(/*share_focal_length=*/false));
        const auto report = ransac.Estimate(points2D, points3D);

        EXPECT_TRUE(report.success);
        EXPECT_NEAR(report.model.focal_lengths.x(), focal_lengths.x(), 1e-3);
        EXPECT_NEAR(report.model.focal_lengths.y(), focal_lengths.y(), 1e-3);
        EXPECT_THAT(report.model.cam_from_world,
                    Rigid3dNear(expected_cam_from_world,
                                /*rtol=*/1e-6,
                                /*ttol=*/1e-6));

        // Test residuals of exact points.
        std::vector<double> residuals;
        P4PFEstimator::Residuals(points2D, points3D, report.model, &residuals);
        for (size_t i = 0; i < residuals.size(); ++i) {
          EXPECT_LT(residuals[i], 1e-3);
        }

        // Test residuals of faulty points.
        P4PFEstimator::Residuals(
            points2D, points3D_faulty, report.model, &residuals);
        for (size_t i = 0; i < residuals.size(); ++i) {
          EXPECT_GT(residuals[i], 0.1);
        }
      }
    }
  }
}

TEST(AbsolutePose, P4PFRefine) {
  const std::vector<Eigen::Vector3d> points3D = {
      Eigen::Vector3d(1, 1, 1),
      Eigen::Vector3d(0, 1, 1),
      Eigen::Vector3d(3, 1.0, 4),
      Eigen::Vector3d(3, 1.1, 4),
      Eigen::Vector3d(3, 1.2, 4),
      Eigen::Vector3d(3, 1.3, 4),
      Eigen::Vector3d(3, 1.4, 4),
      Eigen::Vector3d(2, 1, 7),
  };

  const Rigid3d expected_cam_from_world(
      Eigen::Quaterniond(1, 0.3, 0.1, 0.05).normalized(),
      Eigen::Vector3d(0.5, -0.2, 0.1));

  for (const bool share_focal_length : {true, false}) {
    const Eigen::Vector2d expected_focal_lengths =
        share_focal_length ? Eigen::Vector2d(6, 6) : Eigen::Vector2d(6, 4);

    // Project points to camera coordinate system, centered by the principal
    // point.
    std::vector<P4PFEstimator::X_t> points2D;
    for (size_t i = 0; i < points3D.size(); ++i) {
      points2D.push_back(expected_focal_lengths.cwiseProduct(
          (expected_cam_from_world * points3D[i]).hnormalized()));
    }

    // Refine recovers the true pose and focal length(s) from a perturbed
    // initialization.
    P4PFEstimator estimator(share_focal_length);
    P4PFEstimator::M_t model;
    model.cam_from_world = expected_cam_from_world;
    model.cam_from_world.rotation() =
        Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()) *
        model.cam_from_world.rotation();
    model.cam_from_world.translation() += Eigen::Vector3d(0.1, -0.1, 0.2);
    model.focal_lengths = expected_focal_lengths.cwiseProduct(
        share_focal_length ? Eigen::Vector2d(1.2, 1.2)
                           : Eigen::Vector2d(1.2, 0.8));
    EXPECT_TRUE(estimator.Refine(points2D, points3D, &model));
    EXPECT_THAT(model.cam_from_world,
                Rigid3dNear(expected_cam_from_world,
                            /*rtol=*/1e-6,
                            /*ttol=*/1e-6));
    EXPECT_TRUE(model.focal_lengths.isApprox(expected_focal_lengths, 1e-6));
  }

  // Fewer than four correspondences cannot determine the model; it is left
  // unchanged.
  P4PFEstimator estimator(/*share_focal_length=*/true);
  const std::vector<P4PFEstimator::X_t> points2D(4, Eigen::Vector2d(1, 1));
  const std::vector<P4PFEstimator::X_t> three_points2D(points2D.begin(),
                                                       points2D.begin() + 3);
  const std::vector<Eigen::Vector3d> three_points3D(points3D.begin(),
                                                    points3D.begin() + 3);
  P4PFEstimator::M_t model;
  model.cam_from_world = expected_cam_from_world;
  model.focal_lengths = Eigen::Vector2d(4, 5);
  const P4PFEstimator::M_t model_before = model;
  EXPECT_FALSE(estimator.Refine(three_points2D, three_points3D, &model));
  EXPECT_THAT(model.cam_from_world, Rigid3dEq(model_before.cam_from_world));
  EXPECT_EQ(model.focal_lengths, model_before.focal_lengths);

  // Non-positive focal lengths cannot be optimized in log-space; the model
  // is left unchanged.
  model.focal_lengths.x() = 0;
  const P4PFEstimator::M_t model_before_focal = model;
  const std::vector<Eigen::Vector3d> four_points3D(points3D.begin(),
                                                   points3D.begin() + 4);
  EXPECT_FALSE(estimator.Refine(points2D, four_points3D, &model));
  EXPECT_THAT(model.cam_from_world,
              Rigid3dEq(model_before_focal.cam_from_world));
  EXPECT_EQ(model.focal_lengths, model_before_focal.focal_lengths);
}

TEST(AbsolutePose, P3PRefine) {
  const std::vector<Eigen::Vector3d> points3D = {
      Eigen::Vector3d(1, 1, 1),
      Eigen::Vector3d(0, 1, 1),
      Eigen::Vector3d(3, 1.0, 4),
      Eigen::Vector3d(3, 1.1, 4),
      Eigen::Vector3d(3, 1.2, 4),
      Eigen::Vector3d(3, 1.3, 4),
      Eigen::Vector3d(3, 1.4, 4),
      Eigen::Vector3d(2, 1, 7),
  };

  auto points3D_faulty = points3D;
  for (size_t i = 0; i < points3D.size(); ++i) {
    points3D_faulty[i](0) = 20;
  }

  const Camera camera = Camera::CreateFromModelId(
      kInvalidCameraId, CameraModelId::kPinhole, 12, 34, 56);
  auto img_from_cam_func = [&camera](const Eigen::Vector3d& cam_point) {
    return camera.ImgFromCam(cam_point);
  };

  // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
  for (double qx = 0; qx < 1; qx += 0.2) {
    // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
    for (double tx = 0; tx < 1; tx += 0.1) {
      const Rigid3d expected_cam_from_world(
          Eigen::Quaterniond(1, qx, 0, 0).normalized(),
          Eigen::Vector3d(tx, 0, 0));

      // Project points to camera coordinate system.
      std::vector<P3PEstimator::X_t> points2D;
      for (size_t i = 0; i < points3D.size(); ++i) {
        auto& point2D = points2D.emplace_back();
        point2D.camera_ray =
            (expected_cam_from_world * points3D[i]).normalized();
        point2D.image_point = img_from_cam_func(point2D.camera_ray).value();
      }

      RANSACOptions options;
      options.max_error = 1e-5;
      LORANSAC<P3PEstimator, P3PEstimator> ransac(
          options,
          P3PEstimator(img_from_cam_func),
          P3PEstimator(img_from_cam_func));
      const auto report = ransac.Estimate(points2D, points3D);

      EXPECT_TRUE(report.success);
      EXPECT_THAT(report.model,
                  Rigid3dNear(expected_cam_from_world,
                              /*rtol=*/1e-6,
                              /*ttol=*/1e-6));

      // Test residuals of exact points.
      std::vector<double> residuals;
      ransac.local_estimator.Residuals(
          points2D, points3D, report.model, &residuals);
      for (size_t i = 0; i < residuals.size(); ++i) {
        EXPECT_LT(residuals[i], 1e-3);
      }

      // Test residuals of faulty points.
      ransac.local_estimator.Residuals(
          points2D, points3D_faulty, report.model, &residuals);
      for (size_t i = 0; i < residuals.size(); ++i) {
        EXPECT_GT(residuals[i], 0.1);
      }
    }
  }

  // Refine recovers the true pose from a perturbed initialization.
  const Rigid3d refine_expected_cam_from_world(
      Eigen::Quaterniond(1, 0.3, 0.1, 0.05).normalized(),
      Eigen::Vector3d(0.5, -0.2, 0.1));

  // Project points to camera coordinate system.
  std::vector<P3PEstimator::X_t> refine_points2D;
  for (size_t i = 0; i < points3D.size(); ++i) {
    auto& point2D = refine_points2D.emplace_back();
    point2D.camera_ray =
        (refine_expected_cam_from_world * points3D[i]).normalized();
    point2D.image_point = img_from_cam_func(point2D.camera_ray).value();
  }

  P3PEstimator estimator(img_from_cam_func);
  Rigid3d model = refine_expected_cam_from_world;
  model.rotation() =
      Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()) * model.rotation();
  model.translation() += Eigen::Vector3d(0.1, -0.1, 0.2);
  EXPECT_TRUE(estimator.Refine(refine_points2D, points3D, &model));
  EXPECT_THAT(model,
              Rigid3dNear(refine_expected_cam_from_world,
                          /*rtol=*/1e-6,
                          /*ttol=*/1e-6));

  // Fewer than three correspondences cannot determine the pose; the model is
  // left unchanged.
  const std::vector<P3PEstimator::X_t> two_points2D(
      refine_points2D.begin(), refine_points2D.begin() + 2);
  const std::vector<Eigen::Vector3d> two_points3D(points3D.begin(),
                                                  points3D.begin() + 2);
  const Rigid3d model_before = model;
  EXPECT_FALSE(estimator.Refine(two_points2D, two_points3D, &model));
  EXPECT_THAT(model, Rigid3dEq(model_before));
}

TEST(ComputeSquaredReprojectionError, Nominal) {
  const Camera camera = Camera::CreateFromModelId(
      kInvalidCameraId, CameraModelId::kSimplePinhole, 12, 34, 56);
  auto img_from_cam_func = [&camera](const Eigen::Vector3d& cam_point) {
    return camera.ImgFromCam(cam_point);
  };

  std::vector<Eigen::Vector3d> points3D;
  points3D.emplace_back(-1, 0, 1);
  points3D.emplace_back(-1, 1, 1);
  points3D.emplace_back(0, 0, -1);
  points3D.emplace_back(0, 0, 0);

  std::vector<Point2DWithRay> points2D;
  points2D.push_back(Point2DWithRay{
      Eigen::Vector2d(camera.PrincipalPointX(), camera.PrincipalPointY()),
      Eigen::Vector3d::Zero()});
  points2D.push_back(Point2DWithRay{
      Eigen::Vector2d(camera.PrincipalPointX(), camera.PrincipalPointY()),
      Eigen::Vector3d::Zero()});
  points2D.push_back(
      Point2DWithRay{Eigen::Vector2d::Zero(), Eigen::Vector3d::Zero()});
  points2D.push_back(
      Point2DWithRay{Eigen::Vector2d::Zero(), Eigen::Vector3d::Zero()});

  const Rigid3d cam_from_world(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d(1, 0, 0));

  std::vector<double> residuals;
  ComputeSquaredReprojectionError(points2D,
                                  points3D,
                                  cam_from_world.ToMatrix(),
                                  img_from_cam_func,
                                  &residuals);

  EXPECT_THAT(residuals,
              testing::ElementsAre(0,
                                   camera.FocalLength() * camera.FocalLength(),
                                   std::numeric_limits<double>::max(),
                                   std::numeric_limits<double>::max()));
}

TEST(PropagatePointCovarianceToImage, MatchesNumericJacobian) {
  const Rigid3d cam_from_world(
      Eigen::Quaterniond(1, 0.2, -0.1, 0.05).normalized(),
      Eigen::Vector3d(0.5, -0.3, 0.2));
  const Eigen::Vector3d point3D(1.0, 2.0, 8.0);
  const Eigen::Matrix3d point3D_cov =
      (Eigen::Matrix3d() << 4, 1, 0.5, 1, 2, -0.3, 0.5, -0.3, 9).finished();

  const Eigen::Vector3d point3D_in_cam = cam_from_world * point3D;
  ASSERT_GT(point3D_in_cam.z(), 0);

  const Eigen::Matrix2d actual = PropagatePointCovarianceToImage(
      cam_from_world.rotation().toRotationMatrix(),
      point3D_in_cam,
      point3D_cov);

  // Numeric Jacobian of the normalized projection w.r.t. the world point.
  // Note the explicit return type: hnormalized() on the temporary projected
  // point must be evaluated inside the lambda.
  const auto project =
      [&cam_from_world](const Eigen::Vector3d& point) -> Eigen::Vector2d {
    return (cam_from_world * point).hnormalized();
  };
  const double kEps = 1e-8;
  Eigen::Matrix<double, 2, 3> J_numeric;
  for (int c = 0; c < 3; ++c) {
    Eigen::Vector3d delta = Eigen::Vector3d::Zero();
    delta(c) = kEps;
    J_numeric.col(c) =
        (project(point3D + delta) - project(point3D - delta)) / (2 * kEps);
  }
  const Eigen::Matrix2d expected =
      J_numeric * point3D_cov * J_numeric.transpose();

  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 2; ++c) {
      EXPECT_NEAR(actual(r, c), expected(r, c), 1e-6);
    }
  }
}

TEST(PropagatePixelCovarianceToNormalized, MatchesNumericJacobian) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimpleRadial, 1280.0, 1024, 768);
  const Eigen::Vector2d img_point(700, 300);
  const Eigen::Matrix2d img_cov = (Eigen::Matrix2d() << 4, 1, 1, 2).finished();

  const std::optional<Eigen::Matrix2d> actual =
      PropagatePixelCovarianceToNormalized(camera, img_point, img_cov);
  ASSERT_TRUE(actual.has_value());

  // Numeric Jacobian of the composed unproject-then-normalize map.
  // Note the explicit return type: hnormalized() on the temporary ray must
  // be evaluated inside the lambda.
  const auto normalize =
      [&camera](const Eigen::Vector2d& pixel) -> Eigen::Vector2d {
    const std::optional<Eigen::Vector3d> ray = camera.CamRayFromImg(pixel);
    EXPECT_TRUE(ray.has_value());
    return ray->hnormalized();
  };
  ASSERT_TRUE(camera.CamRayFromImg(img_point).has_value());
  // Note the large epsilon: iterative undistortion stops at a step norm of
  // 1e-5 in normalized coordinates, so smaller pixel steps drown in solver
  // noise. The undistorted map is smooth, keeping truncation error small.
  const double kEps = 1.0;
  Eigen::Matrix2d J_numeric;
  for (int c = 0; c < 2; ++c) {
    Eigen::Vector2d delta = Eigen::Vector2d::Zero();
    delta(c) = kEps;
    J_numeric.col(c) =
        (normalize(img_point + delta) - normalize(img_point - delta)) /
        (2 * kEps);
  }
  const Eigen::Matrix2d expected = J_numeric * img_cov * J_numeric.transpose();

  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 2; ++c) {
      EXPECT_NEAR((*actual)(r, c), expected(r, c), 1e-6);
    }
  }
}

TEST(PropagatePixelCovarianceToNormalized, PinholeCenter) {
  const double focal = 512.0;
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, focal, 1024, 1024);
  const Eigen::Vector2d center(512, 512);
  const std::optional<Eigen::Matrix2d> normalized =
      PropagatePixelCovarianceToNormalized(
          camera, center, Eigen::Matrix2d::Identity());
  ASSERT_TRUE(normalized.has_value());
  // At the principal point of a pinhole camera, J = (1/f) * I.
  EXPECT_NEAR((*normalized)(0, 0), 1 / (focal * focal), 1e-12);
  EXPECT_NEAR((*normalized)(1, 1), 1 / (focal * focal), 1e-12);
  EXPECT_NEAR((*normalized)(0, 1), 0, 1e-12);
  EXPECT_NEAR((*normalized)(1, 0), 0, 1e-12);
}

TEST(PropagatePixelCovarianceToNormalized, RejectsNonFinite) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 512.0, 1024, 1024);
  Eigen::Matrix2d nan_cov = Eigen::Matrix2d::Identity();
  nan_cov(0, 0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(PropagatePixelCovarianceToNormalized(
                   camera, Eigen::Vector2d(512, 512), nan_cov)
                   .has_value());
}

TEST(CovariantP3PEstimator, DegenerateCovariance) {
  const Rigid3d cam_from_world(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d::Zero());

  CovariantP3PEstimator::X_t point2D;
  point2D.first.image_point = Eigen::Vector2d(0.5, -0.25);
  point2D.first.camera_ray = Eigen::Vector3d(0.5, -0.25, 1).normalized();
  CovariantP3PEstimator::Y_t point3D;
  point3D.first = Eigen::Vector3d(1, -0.5, 2);

  // Zero covariances lead to a singular joint covariance.
  point2D.second = Eigen::Matrix2d::Zero();
  point3D.second = Eigen::Matrix3d::Zero();
  std::vector<double> residuals;
  CovariantP3PEstimator::Residuals(
      {point2D}, {point3D}, cam_from_world, &residuals);
  ASSERT_EQ(residuals.size(), 1);
  EXPECT_EQ(residuals[0], std::numeric_limits<double>::max());

  // Non-finite covariances are rejected as well.
  point2D.second = Eigen::Matrix2d::Identity();
  point3D.second =
      Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());
  CovariantP3PEstimator::Residuals(
      {point2D}, {point3D}, cam_from_world, &residuals);
  ASSERT_EQ(residuals.size(), 1);
  EXPECT_EQ(residuals[0], std::numeric_limits<double>::max());

  // Sanity check: exact correspondence with valid covariances.
  point3D.second = Eigen::Matrix3d::Identity();
  CovariantP3PEstimator::Residuals(
      {point2D}, {point3D}, cam_from_world, &residuals);
  ASSERT_EQ(residuals.size(), 1);
  EXPECT_LT(residuals[0], 1e-12);
}

TEST(CovariantP3PEstimator, BackwardRayRejected) {
  const Rigid3d cam_from_world(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d::Zero());

  // The backward ray (0.5, -0.25, -1) has the normalized coordinates
  // (-0.5, 0.25) and would thus match the mirrored point in front of the
  // camera, if not rejected.
  CovariantP3PEstimator::X_t point2D;
  point2D.first.image_point = Eigen::Vector2d(0.5, -0.25);
  point2D.first.camera_ray = Eigen::Vector3d(0.5, -0.25, -1).normalized();
  point2D.second = Eigen::Matrix2d::Identity();
  CovariantP3PEstimator::Y_t point3D;
  point3D.first = Eigen::Vector3d(-1, 0.5, 2);
  point3D.second = Eigen::Matrix3d::Identity();

  std::vector<double> residuals;
  CovariantP3PEstimator::Residuals(
      {point2D}, {point3D}, cam_from_world, &residuals);
  ASSERT_EQ(residuals.size(), 1);
  EXPECT_EQ(residuals[0], std::numeric_limits<double>::max());

  // Sanity check: the forward ray is an exact correspondence.
  point2D.first.camera_ray = Eigen::Vector3d(-0.5, 0.25, 1).normalized();
  CovariantP3PEstimator::Residuals(
      {point2D}, {point3D}, cam_from_world, &residuals);
  ASSERT_EQ(residuals.size(), 1);
  EXPECT_LT(residuals[0], 1e-12);
}

TEST(CovariantP3PEstimator, ResidualsAreUncappedMahalanobisDistances) {
  const Rigid3d cam_from_world(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d::Zero());

  // The point projects to (0.5, -0.25), i.e., the observation is off by 5
  // standard deviations in x. Gating is left to RANSAC, so the residual is
  // not capped at the RANSAC threshold.
  constexpr double kSigma = 0.01;
  CovariantP3PEstimator::X_t point2D;
  point2D.first.image_point = Eigen::Vector2d(0.5 + 5 * kSigma, -0.25);
  point2D.first.camera_ray =
      point2D.first.image_point.homogeneous().normalized();
  point2D.second = kSigma * kSigma * Eigen::Matrix2d::Identity();
  CovariantP3PEstimator::Y_t point3D;
  point3D.first = Eigen::Vector3d(1, -0.5, 2);
  point3D.second = Eigen::Matrix3d::Zero();

  std::vector<double> residuals;
  CovariantP3PEstimator::Residuals(
      {point2D}, {point3D}, cam_from_world, &residuals);
  ASSERT_EQ(residuals.size(), 1);
  EXPECT_NEAR(residuals[0], 25, 1e-6);
}

}  // namespace
}  // namespace colmap

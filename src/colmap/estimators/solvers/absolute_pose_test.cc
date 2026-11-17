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

}  // namespace
}  // namespace colmap

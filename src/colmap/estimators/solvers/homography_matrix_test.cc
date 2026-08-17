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

#include "colmap/estimators/solvers/homography_matrix.h"

#include "colmap/math/math.h"
#include "colmap/math/random_eigen.h"
#include "colmap/util/eigen_alignment.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

class HomographyMatrixTests : public ::testing::TestWithParam<size_t> {};

TEST(HomographyMatrixEstimator, CollinearMinimalSampleTriplets) {
  const std::vector<Eigen::Vector2d> dst = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

  for (size_t outlier_idx = 0; outlier_idx < 4; ++outlier_idx) {
    std::vector<Eigen::Vector2d> src = {{0, 0}, {1, 0}, {2, 0}, {3, 0}};
    src[outlier_idx] = {0, 1};

    HomographyMatrixEstimator estimator;
    std::vector<Eigen::Matrix3d> models;
    estimator.Estimate(src, dst, &models);

    EXPECT_TRUE(models.empty());
  }
}

TEST_P(HomographyMatrixTests, Nominal) {
  const size_t kNumPoints = GetParam();
  for (int x = 0; x < 10; ++x) {
    Eigen::Matrix3d expected_H;
    expected_H << x, 0.2, 0.3, 30, 0.2, 0.1, 0.3, 20, 1;

    std::vector<Eigen::Vector2d> src;
    std::vector<Eigen::Vector2d> dst;
    for (size_t i = 0; i < kNumPoints; ++i) {
      src.push_back(RandomEigenVectord<2>());
      dst.push_back((expected_H * src[i].homogeneous()).hnormalized());
    }

    HomographyMatrixEstimator estimator;
    std::vector<Eigen::Matrix3d> models;
    estimator.Estimate(src, dst, &models);

    ASSERT_EQ(models.size(), 1);

    std::vector<double> residuals;
    estimator.Residuals(src, dst, models[0], &residuals);

    for (size_t i = 0; i < kNumPoints; ++i) {
      EXPECT_LT(residuals[i], 1e-6);
    }
  }
}

// Test numerical stability with large coordinates. This is to ensure that the
// homography matrix estimator is numerically stable despite not using
// coordinate normalization. We can do this because of double precision.
TEST_P(HomographyMatrixTests, NumericalStability) {
  const size_t kNumPoints = GetParam();
  constexpr double kCoordinateScale = 1e6;
  for (int x = 1; x < 10; ++x) {
    Eigen::Matrix3d expected_H = Eigen::Matrix3d::Identity();
    expected_H(0, 0) = x;

    std::vector<Eigen::Vector2d> src;
    std::vector<Eigen::Vector2d> dst;
    for (size_t i = 0; i < kNumPoints; ++i) {
      src.push_back(RandomEigenVectord<2>() * kCoordinateScale);
      dst.push_back((expected_H * src[i].homogeneous()).hnormalized());
    }

    HomographyMatrixEstimator estimator;
    std::vector<Eigen::Matrix3d> models;
    estimator.Estimate(src, dst, &models);
    ASSERT_EQ(models.size(), 1);

    std::vector<double> residuals;
    estimator.Residuals(src, dst, models[0], &residuals);

    for (size_t i = 0; i < kNumPoints; ++i) {
      EXPECT_LT(residuals[i], 1e-6);
    }
  }
}

TEST_P(HomographyMatrixTests, NoiseStability) {
  const size_t kNumPoints = GetParam();
  constexpr double kNoise = 1e-3;
  for (int x = 1; x < 10; ++x) {
    Eigen::Matrix3d expected_H = Eigen::Matrix3d::Identity();
    expected_H(0, 0) = x;

    std::vector<Eigen::Vector2d> src;
    std::vector<Eigen::Vector2d> dst;
    for (size_t i = 0; i < kNumPoints; ++i) {
      src.push_back(RandomEigenVectord<2>());
      dst.push_back((expected_H * src[i].homogeneous()).hnormalized() +
                    RandomEigenVectord<2>() * kNoise);
    }

    HomographyMatrixEstimator estimator;
    std::vector<Eigen::Matrix3d> models;
    estimator.Estimate(src, dst, &models);
    ASSERT_EQ(models.size(), 1);

    std::vector<double> residuals;
    estimator.Residuals(src, dst, models[0], &residuals);

    for (size_t i = 0; i < kNumPoints; ++i) {
      EXPECT_LT(residuals[i], 1e-5);
    }
  }
}

TEST_P(HomographyMatrixTests, Degenerate) {
  const size_t kNumPoints = GetParam();
  constexpr double kNoise = 1e-3;

  for (int x = 0; x < 10; ++x) {
    Eigen::Matrix3d expected_H;
    expected_H << x, 0.2, 0.3, 30, 0.2, 0.1, 0.3, 20, 1;

    std::vector<Eigen::Vector2d> src;
    src.emplace_back(2, 1);
    src.emplace_back(3, 1);
    src.emplace_back(10, 30);
    ASSERT_GE(kNumPoints, 4);
    const size_t num_redundant_points = kNumPoints - src.size();
    for (size_t i = 0; i < num_redundant_points; ++i) {
      src.emplace_back(src.front());
    }

    std::vector<Eigen::Vector2d> dst;
    for (size_t i = 0; i < src.size(); ++i) {
      const Eigen::Vector3d dsth = expected_H * src[i].homogeneous();
      dst.push_back(dsth.hnormalized() + RandomEigenVectord<2>() * kNoise);
    }

    HomographyMatrixEstimator estimator;
    std::vector<Eigen::Matrix3d> models;
    estimator.Estimate(src, dst, &models);

    ASSERT_EQ(models.size(), 0);
  }
}

INSTANTIATE_TEST_SUITE_P(HomographyMatrix,
                         HomographyMatrixTests,
                         ::testing::Values(4, 8, 64, 1024));

class HomographyMatrixRayTests : public ::testing::TestWithParam<size_t> {};

// Correspondences for a plane whose pixel-space homography is `H_pix`, with
// `dst` displaced by `offset` pixels so that residuals are non-trivial.
void SyntheticRayCorrespondences(const Camera& camera1,
                                 const Camera& camera2,
                                 const Eigen::Matrix3d& H_pix,
                                 size_t num_points,
                                 double offset,
                                 std::vector<Eigen::Vector2d>* src,
                                 std::vector<Eigen::Vector2d>* dst,
                                 std::vector<Eigen::Vector3d>* cam_rays1,
                                 std::vector<CamRayWithImgPoint>* cam_rays2) {
  for (size_t i = 0; i < num_points; ++i) {
    const Eigen::Vector2d image_size(static_cast<double>(camera1.width),
                                     static_cast<double>(camera1.height));
    const Eigen::Vector2d point1 = image_size.cwiseProduct(
        0.5 * (RandomEigenVectord<2>() + Eigen::Vector2d::Ones()));
    const Eigen::Vector2d point2 =
        (H_pix * point1.homogeneous()).hnormalized() +
        offset * RandomEigenVectord<2>();
    src->push_back(point1);
    dst->push_back(point2);
    cam_rays1->push_back(camera1.CamRayFromImg(point1).value());
    cam_rays2->push_back({camera2.CamRayFromImg(point2).value(), point2});
  }
}

TEST_P(HomographyMatrixRayTests, Nominal) {
  const size_t kNumPoints = GetParam();
  const Camera camera1 = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 1000, 1920, 1080);
  const Camera camera2 = Camera::CreateFromModelId(
      2, CameraModelId::kSimplePinhole, 1200, 1920, 1080);

  for (int x = 1; x < 10; ++x) {
    Eigen::Matrix3d H_pix;
    H_pix << 1 + 0.1 * x, 0.02, 30, 0.03, 1.1, 20, 1e-5, 2e-5, 1;

    std::vector<Eigen::Vector2d> src;
    std::vector<Eigen::Vector2d> dst;
    std::vector<Eigen::Vector3d> cam_rays1;
    std::vector<CamRayWithImgPoint> cam_rays2;
    SyntheticRayCorrespondences(camera1,
                                camera2,
                                H_pix,
                                kNumPoints,
                                /*offset=*/0,
                                &src,
                                &dst,
                                &cam_rays1,
                                &cam_rays2);

    const HomographyMatrixRayEstimator estimator(&camera2);
    std::vector<Eigen::Matrix3d> models;
    estimator.Estimate(cam_rays1, cam_rays2, &models);

    ASSERT_EQ(models.size(), 1);

    // Also guards the global sign: a negated H transfers every ray behind the
    // camera and scores every residual at the maximum.
    std::vector<double> residuals;
    estimator.Residuals(cam_rays1, cam_rays2, models[0], &residuals);
    for (size_t i = 0; i < kNumPoints; ++i) {
      EXPECT_LT(residuals[i], 1e-6);
    }

    // The estimate maps rays to rays, so it must agree with the pixel-space
    // homography only after conjugation by the calibration matrices.
    const Eigen::Matrix3d H_from_rays = camera2.CalibrationMatrix() *
                                        models[0] *
                                        camera1.CalibrationMatrix().inverse();
    EXPECT_TRUE(H_from_rays.normalized().isApprox(H_pix.normalized(), 1e-6) ||
                H_from_rays.normalized().isApprox(-H_pix.normalized(), 1e-6));
  }
}

// For undistorted pinhole cameras the two residuals are algebraically
// identical, since projecting K2^-1 H K1 x1 back through K2 dehomogenizes to
// exactly H p1. Large offsets are included because a first-order relationship
// would also pass near zero.
TEST_P(HomographyMatrixRayTests, PixelEquivalence) {
  const size_t kNumPoints = GetParam();
  const Camera camera1 = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 1000, 1920, 1080);
  const Camera camera2 =
      Camera::CreateFromModelId(2, CameraModelId::kPinhole, 1200, 1920, 1080);

  for (const double offset : {0.0, 1.0, 20.0, 200.0}) {
    Eigen::Matrix3d H_pix;
    H_pix << 1.2, 0.02, 30, 0.03, 1.1, 20, 1e-5, 2e-5, 1;

    std::vector<Eigen::Vector2d> src;
    std::vector<Eigen::Vector2d> dst;
    std::vector<Eigen::Vector3d> cam_rays1;
    std::vector<CamRayWithImgPoint> cam_rays2;
    SyntheticRayCorrespondences(camera1,
                                camera2,
                                H_pix,
                                kNumPoints,
                                offset,
                                &src,
                                &dst,
                                &cam_rays1,
                                &cam_rays2);

    const Eigen::Matrix3d H_ray = camera2.CalibrationMatrix().inverse() *
                                  H_pix * camera1.CalibrationMatrix();

    std::vector<double> pixel_residuals;
    HomographyMatrixEstimator().Residuals(src, dst, H_pix, &pixel_residuals);

    std::vector<double> ray_residuals;
    HomographyMatrixRayEstimator(&camera2).Residuals(
        cam_rays1, cam_rays2, H_ray, &ray_residuals);

    for (size_t i = 0; i < kNumPoints; ++i) {
      EXPECT_NEAR(ray_residuals[i],
                  pixel_residuals[i],
                  1e-9 * std::max(1.0, pixel_residuals[i]));
    }
  }
}

// A hypothesis that transfers a ray out of a perspective camera's field has no
// image point to score against. Cannot happen for a true inlier, whose measured
// ray faces forward, so this only guards wrong hypotheses.
TEST(HomographyMatrixRay, BehindCamera) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 1000, 1920, 1080);
  const Eigen::Matrix3d H =
      Eigen::AngleAxisd(DegToRad(100.0), Eigen::Vector3d::UnitY())
          .toRotationMatrix();
  // The second ray stays in front after the rotation, so the rejection must be
  // selective rather than blanket.
  const Eigen::Vector3d forward =
      Eigen::AngleAxisd(DegToRad(-70.0), Eigen::Vector3d::UnitY()) *
      Eigen::Vector3d::UnitZ();
  const std::vector<Eigen::Vector3d> cam_rays1 = {Eigen::Vector3d::UnitZ(),
                                                  forward};
  const std::vector<CamRayWithImgPoint> cam_rays2 = {
      {Eigen::Vector3d::UnitZ(), Eigen::Vector2d(960, 540)},
      {(H * forward).normalized(),
       camera.ImgFromCam((H * forward).normalized()).value()}};

  std::vector<double> residuals;
  HomographyMatrixRayEstimator(&camera).Residuals(
      cam_rays1, cam_rays2, H, &residuals);

  EXPECT_EQ(residuals[0], std::numeric_limits<double>::max());
  EXPECT_LT(residuals[1], 1e-6);
}

// A spherical camera images every direction, so the estimator must work over
// the whole sphere and reject nothing on cheirality grounds.
TEST_P(HomographyMatrixRayTests, Spherical) {
  const size_t kNumPoints = GetParam();
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, /*focal_length=*/0, 2048, 1024);
  ASSERT_TRUE(camera.IsSpherical());

  for (int x = 1; x < 10; ++x) {
    const Eigen::Matrix3d expected_H =
        Eigen::AngleAxisd(0.2 * x, Eigen::Vector3d(0.2, 1, 0.3).normalized())
            .toRotationMatrix();

    std::vector<Eigen::Vector3d> cam_rays1;
    std::vector<CamRayWithImgPoint> cam_rays2;
    for (size_t i = 0; i < kNumPoints; ++i) {
      const Eigen::Vector3d ray1 = RandomEigenVectord<3>().normalized();
      const Eigen::Vector3d ray2 = (expected_H * ray1).normalized();
      cam_rays1.push_back(ray1);
      cam_rays2.push_back({ray2, camera.ImgFromCam(ray2).value()});
    }

    const HomographyMatrixRayEstimator estimator(&camera);
    std::vector<Eigen::Matrix3d> models;
    estimator.Estimate(cam_rays1, cam_rays2, &models);

    ASSERT_EQ(models.size(), 1);
    EXPECT_TRUE(models[0].normalized().isApprox(expected_H.normalized(), 1e-6));

    std::vector<double> residuals;
    estimator.Residuals(cam_rays1, cam_rays2, models[0], &residuals);
    for (size_t i = 0; i < kNumPoints; ++i) {
      EXPECT_LT(residuals[i], 1e-6);
    }
  }
}

// The azimuth wraps, so two nearly identical directions can sit a full image
// width apart in pixels. Scoring that raw difference would reject every
// correspondence near the seam.
TEST(HomographyMatrixRay, SphericalSeam) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, /*focal_length=*/0, 2048, 1024);
  constexpr double kDelta = 0.01;
  const Eigen::Vector3d ray1(
      std::sin(M_PI - kDelta), 0, std::cos(M_PI - kDelta));
  const Eigen::Vector3d ray2(
      std::sin(-M_PI + kDelta), 0, std::cos(-M_PI + kDelta));
  const Eigen::Vector2d point1 = camera.ImgFromCam(ray1).value();
  const Eigen::Vector2d point2 = camera.ImgFromCam(ray2).value();
  ASSERT_GT((point1 - point2).norm(), 2000);

  std::vector<double> residuals;
  HomographyMatrixRayEstimator(&camera).Residuals(
      {ray1}, {{ray2, point2}}, Eigen::Matrix3d::Identity(), &residuals);

  const double expected = 2 * kDelta * camera.width / (2 * M_PI);
  EXPECT_NEAR(residuals[0], expected * expected, 1e-6);
}

// Why Estimate must resolve the sign of H. A spherical camera projects -H x1 as
// happily as H x1, only to the antipode, so a negated model is not rejected but
// silently scored against the wrong half of the sphere.
TEST(HomographyMatrixRay, SphericalSign) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, /*focal_length=*/0, 2048, 1024);
  const Eigen::Matrix3d H =
      Eigen::AngleAxisd(0.4, Eigen::Vector3d(0.2, 1, 0.3).normalized())
          .toRotationMatrix();

  std::vector<Eigen::Vector3d> cam_rays1;
  std::vector<CamRayWithImgPoint> cam_rays2;
  for (int i = 0; i < 64; ++i) {
    const Eigen::Vector3d ray1 = RandomEigenVectord<3>().normalized();
    const Eigen::Vector3d ray2 = (H * ray1).normalized();
    cam_rays1.push_back(ray1);
    cam_rays2.push_back({ray2, camera.ImgFromCam(ray2).value()});
  }

  const HomographyMatrixRayEstimator estimator(&camera);

  std::vector<double> residuals;
  estimator.Residuals(cam_rays1, cam_rays2, H, &residuals);
  for (const double residual : residuals) {
    EXPECT_LT(residual, 1e-6);
  }

  std::vector<double> negated_residuals;
  estimator.Residuals(cam_rays1, cam_rays2, -H, &negated_residuals);
  for (const double residual : negated_residuals) {
    // Finite, so nothing rejects it, but half the image away in azimuth.
    EXPECT_LT(residual, std::numeric_limits<double>::max());
    EXPECT_GT(residual, 1e4);
  }
}

INSTANTIATE_TEST_SUITE_P(HomographyMatrixRay,
                         HomographyMatrixRayTests,
                         ::testing::Values(4, 8, 64, 1024));

}  // namespace
}  // namespace colmap

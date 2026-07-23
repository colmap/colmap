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

#include "colmap/estimators/cost_functions/tiny_sampson_error.h"

#include "colmap/estimators/cost_functions/sampson_error.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"

#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// The batched functor's squared residuals match
// ComputeSquaredTangentSampsonError at several 7-parameter poses.
TEST(TinyTangentSampsonErrorCostFunctor, MatchesSquaredTangentSampsonError) {
  auto make = [](const Eigen::Vector3d& ray,
                 const Eigen::Matrix<double, 3, 2>& jac) {
    return CamRayWithJac{ray.normalized(), jac};
  };
  const std::vector<CamRayWithJac> cam_rays1_with_jac = {
      make({0.1, 0.2, 1},
           (Eigen::Matrix<double, 3, 2>() << 1.0, 0.1, 0.05, 1.0, 0.2, -0.1)
               .finished()),
      make({-0.3, 0.1, 1},
           (Eigen::Matrix<double, 3, 2>() << 0.9, -0.1, 0.15, 1.1, -0.05, 0.2)
               .finished()),
      make({0.2, -0.25, 1},
           (Eigen::Matrix<double, 3, 2>() << 1.05, 0.0, 0.0, 0.95, 0.1, 0.1)
               .finished())};
  const std::vector<CamRayWithJac> cam_rays2_with_jac = {
      make({0.15, -0.1, 1},
           (Eigen::Matrix<double, 3, 2>() << 1.0, 0.05, -0.1, 1.0, 0.2, 0.0)
               .finished()),
      make({0.05, 0.3, 1},
           (Eigen::Matrix<double, 3, 2>() << 0.8, 0.2, 0.1, 1.2, 0.0, -0.15)
               .finished()),
      make({-0.2, -0.15, 1},
           (Eigen::Matrix<double, 3, 2>() << 1.1, -0.05, 0.05, 0.9, -0.1, 0.1)
               .finished())};

  const TinyTangentSampsonErrorCostFunctor functor(cam_rays1_with_jac,
                                                   cam_rays2_with_jac);

  const Eigen::Quaterniond q0(
      Eigen::AngleAxisd(0.9, Eigen::Vector3d(-1, 0.5, 2).normalized()));
  const Eigen::Vector3d t0 = Eigen::Vector3d(1.0, -2.0, 0.5).normalized();
  const Eigen::Quaterniond q1(
      Eigen::AngleAxisd(0.3, Eigen::Vector3d(0.2, -1, 0.7).normalized()));
  const Eigen::Vector3d t1 = Eigen::Vector3d(-0.4, 0.8, 1.2).normalized();

  const std::array<Eigen::Quaterniond, 2> quaternions = {q0, q1};
  const std::array<Eigen::Vector3d, 2> translations = {t0, t1};

  for (size_t k = 0; k < quaternions.size(); ++k) {
    const Eigen::Quaterniond q = quaternions[k].normalized();
    const Eigen::Vector3d& t = translations[k];
    double cam2_from_cam1[7] = {
        q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z()};

    std::vector<double> residuals(cam_rays1_with_jac.size());
    ASSERT_TRUE(functor(cam2_from_cam1, residuals.data(), nullptr));

    const Eigen::Matrix3d E = EssentialMatrixFromPose(Rigid3d(q, t));
    for (size_t i = 0; i < cam_rays1_with_jac.size(); ++i) {
      const double expected = ComputeSquaredTangentSampsonError(
          cam_rays1_with_jac[i], cam_rays2_with_jac[i], E);
      EXPECT_NEAR(residuals[i] * residuals[i], expected, 1e-9);
    }
  }
}

// The closed-form 7-parameter Jacobian matches central finite differences.
TEST(TinyTangentSampsonErrorCostFunctor, JacobianMatchesFiniteDifference) {
  auto make = [](const Eigen::Vector3d& ray,
                 const Eigen::Matrix<double, 3, 2>& jac) {
    return CamRayWithJac{ray.normalized(), jac};
  };
  const std::vector<CamRayWithJac> cam_rays1_with_jac = {
      make({0.1, 0.2, 1},
           (Eigen::Matrix<double, 3, 2>() << 1.0, 0.1, 0.05, 1.0, 0.2, -0.1)
               .finished()),
      make({-0.3, 0.1, 1},
           (Eigen::Matrix<double, 3, 2>() << 0.9, -0.1, 0.15, 1.1, -0.05, 0.2)
               .finished())};
  const std::vector<CamRayWithJac> cam_rays2_with_jac = {
      make({0.15, -0.1, 1},
           (Eigen::Matrix<double, 3, 2>() << 1.0, 0.05, -0.1, 1.0, 0.2, 0.0)
               .finished()),
      make({0.05, 0.3, 1},
           (Eigen::Matrix<double, 3, 2>() << 0.8, 0.2, 0.1, 1.2, 0.0, -0.15)
               .finished())};
  const TinyTangentSampsonErrorCostFunctor functor(cam_rays1_with_jac,
                                                   cam_rays2_with_jac);
  const int n = static_cast<int>(cam_rays1_with_jac.size());

  const Eigen::Quaterniond q(
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(0.2, -1, 0.5).normalized()));
  const Eigen::Vector3d t = Eigen::Vector3d(0.6, -0.3, 1.0).normalized();
  double p[7] = {q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z()};

  std::vector<double> residuals(n), jacobian(n * 7);
  ASSERT_TRUE(functor(p, residuals.data(), jacobian.data()));

  constexpr double kEps = 1e-6;
  for (int l = 0; l < 7; ++l) {
    double p_plus[7], p_minus[7];
    for (int k = 0; k < 7; ++k) {
      p_plus[k] = p[k];
      p_minus[k] = p[k];
    }
    p_plus[l] += kEps;
    p_minus[l] -= kEps;
    std::vector<double> res_plus(n), res_minus(n);
    functor(p_plus, res_plus.data(), nullptr);
    functor(p_minus, res_minus.data(), nullptr);
    for (int i = 0; i < n; ++i) {
      const double finite_diff = (res_plus[i] - res_minus[i]) / (2 * kEps);
      EXPECT_NEAR(jacobian[i + l * n], finite_diff, 1e-5);
    }
  }
}

// The analytic functor agrees with the independent autodiff
// TangentSampsonErrorCostFunctor (the one RefineRelativePose uses) on both the
// residual and the 7-parameter Jacobian, pinning the hand-derived Jacobian to a
// second implementation rather than to finite differences alone.
TEST(TinyTangentSampsonErrorCostFunctor, MatchesAutodiffCostFunctor) {
  auto make = [](const Eigen::Vector3d& ray,
                 const Eigen::Matrix<double, 3, 2>& jac) {
    return CamRayWithJac{ray.normalized(), jac};
  };
  const std::vector<CamRayWithJac> cam_rays1_with_jac = {
      make({0.1, 0.2, 1},
           (Eigen::Matrix<double, 3, 2>() << 1.0, 0.1, 0.05, 1.0, 0.2, -0.1)
               .finished()),
      make({-0.3, 0.1, 1},
           (Eigen::Matrix<double, 3, 2>() << 0.9, -0.1, 0.15, 1.1, -0.05, 0.2)
               .finished())};
  const std::vector<CamRayWithJac> cam_rays2_with_jac = {
      make({0.15, -0.1, 1},
           (Eigen::Matrix<double, 3, 2>() << 1.0, 0.05, -0.1, 1.0, 0.2, 0.0)
               .finished()),
      make({0.05, 0.3, 1},
           (Eigen::Matrix<double, 3, 2>() << 0.8, 0.2, 0.1, 1.2, 0.0, -0.15)
               .finished())};
  const TinyTangentSampsonErrorCostFunctor functor(cam_rays1_with_jac,
                                                   cam_rays2_with_jac);
  const int n = static_cast<int>(cam_rays1_with_jac.size());

  const Eigen::Quaterniond q(
      Eigen::AngleAxisd(0.6, Eigen::Vector3d(0.3, -0.7, 0.5).normalized()));
  const Eigen::Vector3d t = Eigen::Vector3d(0.4, 0.9, -0.2).normalized();
  double p[7] = {q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z()};

  std::vector<double> residuals(n), jacobian(n * 7);
  ASSERT_TRUE(functor(p, residuals.data(), jacobian.data()));

  for (int i = 0; i < n; ++i) {
    std::unique_ptr<ceres::CostFunction> cost(
        TangentSampsonErrorCostFunctor::Create(cam_rays1_with_jac[i],
                                               cam_rays2_with_jac[i]));
    double residual_ad = 0.0;
    double jacobian_ad[7] = {};
    double* jacobian_ad_ptrs[] = {jacobian_ad};
    const double* param_ptrs[] = {p};
    ASSERT_TRUE(cost->Evaluate(param_ptrs, &residual_ad, jacobian_ad_ptrs));
    EXPECT_NEAR(residuals[i], residual_ad, 1e-9);
    for (int l = 0; l < 7; ++l) {
      EXPECT_NEAR(jacobian[i + l * n], jacobian_ad[l], 1e-9);
    }
  }
}

// The batched focal functor's residuals match the pixel-space squared Sampson
// error of the fundamental matrix F = diag(1/f, 1/f, 1) * E * diag(1/f, 1/f, 1)
// implied by the pose and shared focal, at several poses and focal lengths.
TEST(TinyFocalSampsonErrorCostFunctor, MatchesSquaredSampsonError) {
  // Principal-point-centered image points (u - cx, v - cy), not calibrated
  // rays.
  const std::vector<Eigen::Vector2d> points1 = {
      {120.0, -45.0}, {-200.0, 80.0}, {33.0, 210.0}};
  const std::vector<Eigen::Vector2d> points2 = {
      {95.0, -60.0}, {-180.0, 100.0}, {50.0, 190.0}};

  const TinyFocalSampsonErrorCostFunctor functor(points1, points2);

  const Eigen::Quaterniond q0(
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(0.3, -1.0, 0.5).normalized()));
  const Eigen::Vector3d t0 = Eigen::Vector3d(1.0, -0.5, 2.0).normalized();
  const Eigen::Quaterniond q1(
      Eigen::AngleAxisd(0.25, Eigen::Vector3d(-0.6, 0.4, 1.0).normalized()));
  const Eigen::Vector3d t1 = Eigen::Vector3d(-0.7, 1.1, 0.3).normalized();

  const std::array<Eigen::Quaterniond, 2> quaternions = {q0, q1};
  const std::array<Eigen::Vector3d, 2> translations = {t0, t1};
  const std::array<double, 2> focals = {900.0, 1500.0};

  for (size_t k = 0; k < quaternions.size(); ++k) {
    const Eigen::Quaterniond q = quaternions[k].normalized();
    const Eigen::Vector3d& t = translations[k];
    const double focal = focals[k];
    double params[8] = {
        q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z(), std::log(focal)};

    std::vector<double> residuals(points1.size());
    ASSERT_TRUE(functor(params, residuals.data()));

    // Independent reference: build the pixel-space fundamental matrix and
    // evaluate the squared Sampson error with a matrix-based implementation.
    const Eigen::Matrix3d E = EssentialMatrixFromPose(Rigid3d(q, t));
    const double inv_f = 1.0 / focal;
    const Eigen::DiagonalMatrix<double, 3> K_inv(inv_f, inv_f, 1.0);
    const Eigen::Matrix3d F = K_inv * E * K_inv;
    std::vector<double> expected_squared;
    ComputeSquaredSampsonError(points1, points2, F, &expected_squared);

    for (size_t i = 0; i < points1.size(); ++i) {
      EXPECT_NEAR(residuals[i] * residuals[i], expected_squared[i], 1e-9);
    }
  }
}

TEST(TinyOneSidedFocalEpipolarErrorCostFunctor, MatchesEpipolarLineDistance) {
  // Centered image points of the uncalibrated view and bearing rays of the
  // calibrated view. The rays deliberately include one with z < 0, which no
  // pinhole image plane could represent but a spherical camera observes.
  const std::vector<Eigen::Vector2d> points1 = {
      {120.0, -45.0}, {-200.0, 80.0}, {33.0, 210.0}};
  const std::vector<Eigen::Vector3d> cam_rays2 = {
      Eigen::Vector3d(0.08, -0.05, 1.0).normalized(),
      Eigen::Vector3d(-0.9, 0.4, 0.2).normalized(),
      Eigen::Vector3d(0.3, 0.7, -0.6).normalized()};

  const TinyOneSidedFocalEpipolarErrorCostFunctor functor(points1, cam_rays2);

  const Eigen::Quaterniond q0(
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(0.3, -1.0, 0.5).normalized()));
  const Eigen::Vector3d t0 = Eigen::Vector3d(1.0, -0.5, 2.0).normalized();
  const Eigen::Quaterniond q1(
      Eigen::AngleAxisd(0.25, Eigen::Vector3d(-0.6, 0.4, 1.0).normalized()));
  const Eigen::Vector3d t1 = Eigen::Vector3d(-0.7, 1.1, 0.3).normalized();

  const std::array<Eigen::Quaterniond, 2> quaternions = {q0, q1};
  const std::array<Eigen::Vector3d, 2> translations = {t0, t1};
  const std::array<double, 2> focals1 = {900.0, 1500.0};

  for (size_t k = 0; k < quaternions.size(); ++k) {
    const Eigen::Quaterniond q = quaternions[k].normalized();
    const Eigen::Vector3d& t = translations[k];
    const double focal1 = focals1[k];
    double params[8] = {
        q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z(), std::log(focal1)};

    std::vector<double> residuals(points1.size());
    ASSERT_TRUE(functor(params, residuals.data()));

    // Independent reference: build M = E * K1inv explicitly and compute the
    // distance from each point to its epipolar line in first-view pixels.
    const Eigen::Matrix3d E = EssentialMatrixFromPose(Rigid3d(q, t));
    const Eigen::DiagonalMatrix<double, 3> K1_inv(
        1.0 / focal1, 1.0 / focal1, 1.0);
    const Eigen::Matrix3d M = E * K1_inv;

    for (size_t i = 0; i < points1.size(); ++i) {
      const Eigen::Vector3d line1 = M.transpose() * cam_rays2[i];
      const double expected = cam_rays2[i].dot(M * points1[i].homogeneous()) /
                              line1.head<2>().norm();
      EXPECT_NEAR(residuals[i], expected, 1e-9);
    }
  }
}

// The residual is a true distance in first-view pixels: displacing a point by a
// known amount perpendicular to its epipolar line changes it by that amount.
TEST(TinyOneSidedFocalEpipolarErrorCostFunctor, ResidualIsInFirstViewPixels) {
  const double focal1 = 900.0;
  const Eigen::Quaterniond q(
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(0.3, -1.0, 0.5).normalized()));
  const Eigen::Vector3d t = Eigen::Vector3d(1.0, -0.5, 2.0).normalized();
  double params[8] = {
      q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z(), std::log(focal1)};

  const Eigen::Matrix3d E = EssentialMatrixFromPose(Rigid3d(q, t));
  const Eigen::Matrix3d M =
      E * Eigen::DiagonalMatrix<double, 3>(1.0 / focal1, 1.0 / focal1, 1.0);

  // A point exactly on its epipolar line, then shifted perpendicular to it.
  const Eigen::Vector3d cam_ray2 =
      Eigen::Vector3d(0.08, -0.05, 1.0).normalized();
  const Eigen::Vector3d line1 = M.transpose() * cam_ray2;
  Eigen::Vector2d point1(120.0, -45.0);
  point1 -= line1.head<2>().normalized() *
            (cam_ray2.dot(M * point1.homogeneous()) / line1.head<2>().norm());

  constexpr double kOffsetPixels = 2.5;
  const std::vector<Eigen::Vector2d> points1 = {
      point1, point1 + kOffsetPixels * line1.head<2>().normalized()};
  const std::vector<Eigen::Vector3d> cam_rays2 = {cam_ray2, cam_ray2};

  const TinyOneSidedFocalEpipolarErrorCostFunctor functor(points1, cam_rays2);
  std::vector<double> residuals(points1.size());
  ASSERT_TRUE(functor(params, residuals.data()));

  EXPECT_NEAR(std::abs(residuals[0]), 0.0, 1e-9);
  EXPECT_NEAR(std::abs(residuals[1]), kOffsetPixels, 1e-9);
}

}  // namespace
}  // namespace colmap

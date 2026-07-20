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

// The batched functor's residuals match the per-point reference
// SampsonErrorCostFunctor at several 7-parameter poses.
TEST(TinySampsonErrorCostFunctor, MatchesSampsonError) {
  const std::vector<Eigen::Vector3d> cam_rays1 = {
      Eigen::Vector3d(0.1, 0.2, 1).normalized(),
      Eigen::Vector3d(-0.3, 0.1, 1).normalized(),
      Eigen::Vector3d(0.2, -0.25, 1).normalized()};
  const std::vector<Eigen::Vector3d> cam_rays2 = {
      Eigen::Vector3d(0.15, -0.1, 1).normalized(),
      Eigen::Vector3d(0.05, 0.3, 1).normalized(),
      Eigen::Vector3d(-0.2, -0.15, 1).normalized()};

  const TinySampsonErrorCostFunctor functor(cam_rays1, cam_rays2);

  const Eigen::Quaterniond q0(
      Eigen::AngleAxisd(0.9, Eigen::Vector3d(-1, 0.5, 2).normalized()));
  const Eigen::Vector3d t0 = Eigen::Vector3d(1.0, -2.0, 0.5).normalized();
  const Eigen::Quaterniond q1(
      Eigen::AngleAxisd(0.3, Eigen::Vector3d(0.2, -1, 0.7).normalized()));
  const Eigen::Vector3d t1 = Eigen::Vector3d(-0.4, 0.8, 1.2).normalized();

  const std::array<Eigen::Quaterniond, 2> quaternions = {q0, q1};
  const std::array<Eigen::Vector3d, 2> translations = {t0, t1};

  for (size_t k = 0; k < quaternions.size(); ++k) {
    const Eigen::Quaterniond& q = quaternions[k];
    const Eigen::Vector3d& t = translations[k];
    double cam2_from_cam1[7] = {
        q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z()};

    std::vector<double> residuals(cam_rays1.size());
    ASSERT_TRUE(functor(cam2_from_cam1, residuals.data()));

    const double* parameters[1] = {cam2_from_cam1};
    for (size_t i = 0; i < cam_rays1.size(); ++i) {
      std::unique_ptr<ceres::CostFunction> cost_function(
          SampsonErrorCostFunctor::Create(cam_rays1[i], cam_rays2[i]));
      double expected_residual[1];
      ASSERT_TRUE(
          cost_function->Evaluate(parameters, expected_residual, nullptr));
      EXPECT_NEAR(residuals[i], expected_residual[0], 1e-9);
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

}  // namespace
}  // namespace colmap

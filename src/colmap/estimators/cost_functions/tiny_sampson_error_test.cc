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

#include <array>
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

}  // namespace
}  // namespace colmap

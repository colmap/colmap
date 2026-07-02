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

#include "colmap/estimators/cost_functions/tangent_sampson_error.h"

#include "colmap/estimators/cost_functions/sampson_error.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/util/eigen_matchers.h"

#include <array>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// BoxPlus at the origin recovers the base pose; at a non-zero tangent it
// matches an independently computed retraction.
TEST(TangentRelativePose, BoxPlus) {
  const Eigen::Quaterniond q0(
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()));
  const Eigen::Vector3d t0 = Eigen::Vector3d(0.5, -1.0, 2.0).normalized();
  const TangentRelativePose tangent(Rigid3d(q0, t0));

  {
    const double params[5] = {0, 0, 0, 0, 0};
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    tangent.BoxPlus(params, &R, &t);
    EXPECT_THAT(R, EigenMatrixNear(q0.toRotationMatrix(), 1e-12));
    EXPECT_THAT(t, EigenMatrixNear(t0, 1e-12));
  }

  {
    const double params[5] = {0.05, -0.1, 0.08, 0.15, -0.2};
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    tangent.BoxPlus(params, &R, &t);

    const Eigen::Vector3d b1 = t0.unitOrthogonal();
    const Eigen::Vector3d b2 = t0.cross(b1);
    const Eigen::Vector3d omega(params[0], params[1], params[2]);
    const Eigen::Matrix3d expected_R =
        q0.toRotationMatrix() *
        Eigen::AngleAxisd(omega.norm(), omega.normalized()).toRotationMatrix();
    const Eigen::Vector3d expected_t =
        (t0 + params[3] * b1 + params[4] * b2).normalized();
    EXPECT_THAT(R, EigenMatrixNear(expected_R, 1e-12));
    EXPECT_THAT(t, EigenMatrixNear(expected_t, 1e-12));
  }
}

// The functor residuals match the reference Sampson error at the retracted
// pose, both at the base pose (zero tangent) and away from it.
TEST(TangentSampsonErrorCostFunctor, MatchesSampsonError) {
  const Eigen::Quaterniond q(
      Eigen::AngleAxisd(0.9, Eigen::Vector3d(-1, 0.5, 2).normalized()));
  const Eigen::Vector3d t = Eigen::Vector3d(1.0, -2.0, 0.5).normalized();

  const std::vector<Eigen::Vector3d> cam_rays1 = {
      Eigen::Vector3d(0.1, 0.2, 1).normalized(),
      Eigen::Vector3d(-0.3, 0.1, 1).normalized(),
      Eigen::Vector3d(0.2, -0.25, 1).normalized()};
  const std::vector<Eigen::Vector3d> cam_rays2 = {
      Eigen::Vector3d(0.15, -0.1, 1).normalized(),
      Eigen::Vector3d(0.05, 0.3, 1).normalized(),
      Eigen::Vector3d(-0.2, -0.15, 1).normalized()};

  const TangentRelativePose tangent(Rigid3d(q, t));
  const TangentSampsonErrorCostFunctor functor(tangent, cam_rays1, cam_rays2);

  const std::array<double, 5> params_list[] = {{0, 0, 0, 0, 0},
                                               {0.05, -0.1, 0.08, 0.15, -0.2}};
  for (const std::array<double, 5>& params : params_list) {
    std::vector<double> residuals(cam_rays1.size());
    ASSERT_TRUE(functor(params.data(), residuals.data()));

    // Reference pose: the tangent retracted by params.
    Eigen::Matrix3d R;
    Eigen::Vector3d t_ret;
    tangent.BoxPlus(params.data(), &R, &t_ret);
    const Eigen::Quaterniond q_ret(R);
    double cam2_from_cam1[7] = {q_ret.x(),
                                q_ret.y(),
                                q_ret.z(),
                                q_ret.w(),
                                t_ret.x(),
                                t_ret.y(),
                                t_ret.z()};
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

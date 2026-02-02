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

#include "colmap/estimators/cost_functions/alignment.h"

#include "colmap/geometry/sim3.h"
#include "colmap/math/random.h"
#include "colmap/util/eigen_matchers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(Point3DAlignmentCostFunctor, UseLogScale) {
  Sim3d b_from_a = Sim3d(RandomUniformReal<double>(0.1, 10),
                         Eigen::Quaterniond::UnitRandom(),
                         Eigen::Vector3d::Random());
  const Eigen::Vector3d point_in_b_prior(1., 2., 3.);
  const Eigen::Vector3d point_in_a(3., 2., 1.);
  const Eigen::Vector3d point_in_b = b_from_a * point_in_a;
  std::unique_ptr<ceres::CostFunction> cost_function(
      Point3DAlignmentCostFunctor::Create(point_in_b_prior,
                                          /*use_log_scale=*/true));
  b_from_a.scale() = std::log(b_from_a.scale());
  const double* parameters_log_scale[2] = {point_in_a.data(),
                                           b_from_a.params.data()};
  Eigen::Vector3d residuals;
  EXPECT_TRUE(
      cost_function->Evaluate(parameters_log_scale, residuals.data(), nullptr));

  const Eigen::Vector3d error = point_in_b - point_in_b_prior;
  EXPECT_THAT(residuals, EigenMatrixNear(error, 1e-6));
}

TEST(Point3DAlignmentCostFunctor, DoNotUseLogScale) {
  const Sim3d b_from_a = Sim3d(RandomUniformReal<double>(0.1, 10),
                               Eigen::Quaterniond::UnitRandom(),
                               Eigen::Vector3d::Random());
  const Eigen::Vector3d point_in_b_prior(1., 2., 3.);
  const Eigen::Vector3d point_in_a(3., 2., 1.);
  const Eigen::Vector3d point_in_b = b_from_a * point_in_a;
  std::unique_ptr<ceres::CostFunction> cost_function(
      Point3DAlignmentCostFunctor::Create(point_in_b_prior,
                                          /*use_log_scale=*/false));
  const double* parameters_log_scale[2] = {point_in_a.data(),
                                           b_from_a.params.data()};
  Eigen::Vector3d residuals;
  EXPECT_TRUE(
      cost_function->Evaluate(parameters_log_scale, residuals.data(), nullptr));

  const Eigen::Vector3d error = point_in_b - point_in_b_prior;
  EXPECT_THAT(residuals, EigenMatrixNear(error, 1e-6));
}

}  // namespace
}  // namespace colmap

// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/cost_functions/alignment.h"

#include "colmap/geometry/sim3.h"
#include "colmap/math/random.h"
#include "colmap/math/random_eigen.h"
#include "colmap/util/eigen_matchers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(Point3DAlignmentCostFunctor, UseLogScale) {
  Sim3d b_from_a = Sim3d(RandomUniformReal<double>(0.1, 10),
                         RandomEigenQuaterniond(),
                         RandomEigenVectord<3>());
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
                               RandomEigenQuaterniond(),
                               RandomEigenVectord<3>());
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

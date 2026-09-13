// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/cost_functions/sampson_error.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/util/eigen_matchers.h"

#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(SampsonErrorCostFunctor, Nominal) {
  std::unique_ptr<ceres::CostFunction> cost_function(
      SampsonErrorCostFunctor::Create(Eigen::Vector2d(0, 0),
                                      Eigen::Vector2d(0, 0)));
  double cam_from_world[7] = {0, 0, 0, 1, 0, 1, 0};
  double residuals[1];
  const double* parameters[1] = {cam_from_world};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);

  cost_function.reset(SampsonErrorCostFunctor::Create(Eigen::Vector2d(0, 0),
                                                      Eigen::Vector2d(1, 0)));
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_NEAR(residuals[0] * residuals[0], 0.5, 1e-6);

  cost_function.reset(SampsonErrorCostFunctor::Create(Eigen::Vector2d(0, 0),
                                                      Eigen::Vector2d(1, 1)));
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_NEAR(residuals[0] * residuals[0], 0.5, 1e-6);
}

constexpr double kEps = 1e-7;

Rigid3d TestPose() {
  return Rigid3d(Eigen::Quaterniond(Eigen::AngleAxisd(
                     0.7, Eigen::Vector3d(0.2, -1, 0.5).normalized())),
                 Eigen::Vector3d(0.6, -0.3, 1).normalized());
}

TEST(TangentSampsonErrorAndJacWrtE, MatchesTemplatedErrorAndFiniteDifference) {
  const Eigen::Vector3d ray1 = Eigen::Vector3d(0.1, 0.2, 1).normalized();
  const Eigen::Vector3d ray2 = Eigen::Vector3d(0.15, -0.1, 1).normalized();
  const Eigen::Matrix3x2d J1 =
      (Eigen::Matrix3x2d() << 1.0, 0.1, 0.05, 1.0, 0.2, -0.1).finished();
  const Eigen::Matrix3x2d J2 =
      (Eigen::Matrix3x2d() << 1.0, 0.05, -0.1, 1.0, 0.2, 0.0).finished();
  const Eigen::Matrix3d E = EssentialMatrixFromPose(TestPose());
  const auto error = [&](const Eigen::Matrix3d& E) {
    return TangentSampsonError<double>(E, ray1, J1, ray2, J2);
  };

  Eigen::Matrix3d drdE;
  EXPECT_NEAR(TangentSampsonErrorAndJacWrtE(E, ray1, J1, ray2, J2, &drdE),
              error(E),
              1e-12);

  Eigen::Matrix3d finite_diff;
  for (int i = 0; i < 9; ++i) {
    Eigen::Matrix3d E_plus = E;
    Eigen::Matrix3d E_minus = E;
    E_plus(i) += kEps;
    E_minus(i) -= kEps;
    finite_diff(i) = (error(E_plus) - error(E_minus)) / (2 * kEps);
  }
  EXPECT_THAT(drdE, EigenMatrixNear(finite_diff, 1e-5));
}

TEST(TangentSampsonErrorAndJacWrtE, DegenerateDenominator) {
  Eigen::Matrix3d drdE = Eigen::Matrix3d::Ones();
  EXPECT_EQ(TangentSampsonErrorAndJacWrtE(Eigen::Matrix3d::Identity(),
                                          Eigen::Vector3d::Zero(),
                                          Eigen::Matrix3x2d::Zero(),
                                          Eigen::Vector3d::Zero(),
                                          Eigen::Matrix3x2d::Zero(),
                                          &drdE),
            0.0);
  EXPECT_TRUE(drdE.isZero(0.0));
}

TEST(EssentialMatrixAndJacFromPoseParams, MatchesTemplatedAndFiniteDifference) {
  const Eigen::Matrix<double, 7, 1> params =
      (Eigen::Matrix<double, 7, 1>() << TestPose().rotation().coeffs(),
       TestPose().translation())
          .finished();

  Eigen::Matrix3d dE[7];
  const Eigen::Matrix3d E =
      EssentialMatrixAndJacFromPoseParams(params.data(), dE);
  EXPECT_THAT(E,
              EigenMatrixNear(
                  EssentialMatrixFromPoseParams<double>(params.data()), 1e-12));

  for (int l = 0; l < 7; ++l) {
    Eigen::Matrix<double, 7, 1> params_plus = params;
    Eigen::Matrix<double, 7, 1> params_minus = params;
    params_plus[l] += kEps;
    params_minus[l] -= kEps;
    const Eigen::Matrix3d finite_diff =
        (EssentialMatrixFromPoseParams<double>(params_plus.data()) -
         EssentialMatrixFromPoseParams<double>(params_minus.data())) /
        (2 * kEps);
    EXPECT_THAT(dE[l], EigenMatrixNear(finite_diff, 1e-5));
  }
}
}  // namespace
}  // namespace colmap

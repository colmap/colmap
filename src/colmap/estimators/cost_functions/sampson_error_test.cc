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

TEST(TangentSampsonErrorAndJacWrtE, MatchesTemplatedErrorAndFiniteDifference) {
  const Eigen::Vector3d ray1 = Eigen::Vector3d(0.1, 0.2, 1).normalized();
  const Eigen::Vector3d ray2 = Eigen::Vector3d(0.15, -0.1, 1).normalized();
  const Eigen::Matrix3x2d J1 =
      (Eigen::Matrix3x2d() << 1.0, 0.1, 0.05, 1.0, 0.2, -0.1).finished();
  const Eigen::Matrix3x2d J2 =
      (Eigen::Matrix3x2d() << 1.0, 0.05, -0.1, 1.0, 0.2, 0.0).finished();
  const Eigen::Matrix3d E = EssentialMatrixFromPose(
      Rigid3d(Eigen::Quaterniond(Eigen::AngleAxisd(
                  0.7, Eigen::Vector3d(0.2, -1, 0.5).normalized())),
              Eigen::Vector3d(0.6, -0.3, 1).normalized()));

  Eigen::Matrix3d drdE;
  const double residual =
      TangentSampsonErrorAndJacWrtE(E, ray1, J1, ray2, J2, &drdE);
  EXPECT_NEAR(
      residual, TangentSampsonError<double>(E, ray1, J1, ray2, J2), 1e-12);

  constexpr double kEps = 1e-7;
  Eigen::Matrix3d finite_diff;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      Eigen::Matrix3d E_plus = E;
      Eigen::Matrix3d E_minus = E;
      E_plus(r, c) += kEps;
      E_minus(r, c) -= kEps;
      finite_diff(r, c) =
          (TangentSampsonError<double>(E_plus, ray1, J1, ray2, J2) -
           TangentSampsonError<double>(E_minus, ray1, J1, ray2, J2)) /
          (2 * kEps);
    }
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
  const Eigen::Quaterniond q(
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(0.2, -1, 0.5).normalized()));
  const Eigen::Vector3d t = Eigen::Vector3d(0.6, -0.3, 1).normalized();
  const double params[7] = {q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z()};

  Eigen::Matrix3d dE[7];
  const Eigen::Matrix3d E = EssentialMatrixAndJacFromPoseParams(params, dE);
  EXPECT_THAT(
      E, EigenMatrixNear(EssentialMatrixFromPoseParams<double>(params), 1e-12));

  constexpr double kEps = 1e-7;
  for (int l = 0; l < 7; ++l) {
    double params_plus[7], params_minus[7];
    for (int k = 0; k < 7; ++k) {
      params_plus[k] = params[k];
      params_minus[k] = params[k];
    }
    params_plus[l] += kEps;
    params_minus[l] -= kEps;
    const Eigen::Matrix3d finite_diff =
        (EssentialMatrixFromPoseParams<double>(params_plus) -
         EssentialMatrixFromPoseParams<double>(params_minus)) /
        (2 * kEps);
    EXPECT_THAT(dE[l], EigenMatrixNear(finite_diff, 1e-5));
  }
}
}  // namespace
}  // namespace colmap

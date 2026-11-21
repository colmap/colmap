// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/cost_functions/calibration.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/random.h"
#include "colmap/math/random_eigen.h"

#include <cmath>

#include <gtest/gtest.h>

namespace colmap {
namespace {

Eigen::Matrix3d CalibrationMatrix(double focal_length,
                                  const Eigen::Vector2d& principal_point) {
  Eigen::Matrix3d K;
  K << focal_length, 0, principal_point(0), 0, focal_length, principal_point(1),
      0, 0, 1;
  return K;
}

// Reference implementation of dist^2(A, E) / ||A||_F^2 via the SVD.
double NormalizedEssentialDistanceSq(const Eigen::Matrix3d& F,
                                     double focal_length1,
                                     const Eigen::Vector2d& principal_point1,
                                     double focal_length2,
                                     const Eigen::Vector2d& principal_point2) {
  const Eigen::Matrix3d A =
      CalibrationMatrix(focal_length2, principal_point2).transpose() * F *
      CalibrationMatrix(focal_length1, principal_point1);
  const Eigen::Vector3d s = A.jacobiSvd().singularValues();
  return (s(0) - s(1)) * (s(0) - s(1)) / (2 * (s(0) * s(0) + s(1) * s(1)));
}

Eigen::Vector2d EvaluateResidual(
    const EssentialManifoldFocalLengthCostFunctor& cost_functor,
    double focal_length1,
    double focal_length2) {
  const double log_focal_length1 = std::log(focal_length1);
  const double log_focal_length2 = std::log(focal_length2);
  Eigen::Vector2d residual;
  EXPECT_TRUE(
      cost_functor(&log_focal_length1, &log_focal_length2, residual.data()));
  return residual;
}

TEST(EssentialManifoldFocalLengthCostFunctor, ZeroAtTrueFocalLengths) {
  const double focal_length1 = 1200;
  const double focal_length2 = 800;
  const Eigen::Vector2d principal_point1(640, 480);
  const Eigen::Vector2d principal_point2(400, 300);
  for (int i = 0; i < 100; ++i) {
    const Eigen::Matrix3d F = FundamentalFromEssentialMatrix(
        CalibrationMatrix(focal_length2, principal_point2),
        EssentialMatrixFromPose(
            Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>())),
        CalibrationMatrix(focal_length1, principal_point1));
    // The initial focal lengths are arbitrary.
    const EssentialManifoldFocalLengthCostFunctor cost_functor(
        F,
        principal_point1,
        principal_point2,
        RandomUniformReal(100.0, 5000.0),
        RandomUniformReal(100.0, 5000.0));
    EXPECT_LT(
        EvaluateResidual(cost_functor, focal_length1, focal_length2).norm(),
        1e-8);
    EXPECT_GT(EvaluateResidual(cost_functor, 1.2 * focal_length1, focal_length2)
                  .norm(),
              1e-4);
    EXPECT_GT(EvaluateResidual(cost_functor, focal_length1, 0.8 * focal_length2)
                  .norm(),
              1e-4);
  }
}

TEST(EssentialManifoldFocalLengthCostFunctor,
     EqualsNormalizedEssentialDistance) {
  const Eigen::Vector2d principal_point1(640, 480);
  const Eigen::Vector2d principal_point2(400, 300);
  for (int i = 0; i < 100; ++i) {
    // Random rank-2 matrix, which is in general not a fundamental matrix of
    // any calibrated camera pair.
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        Eigen::Matrix3d::Random(), Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Vector3d singular_values(1, RandomUniformReal(0.01, 1.0), 0);
    const Eigen::Matrix3d F = RandomUniformReal(1e-6, 1e2) * svd.matrixU() *
                              singular_values.asDiagonal() *
                              svd.matrixV().transpose();
    const EssentialManifoldFocalLengthCostFunctor cost_functor(
        F,
        principal_point1,
        principal_point2,
        RandomUniformReal(100.0, 5000.0),
        RandomUniformReal(100.0, 5000.0));
    for (int j = 0; j < 10; ++j) {
      const double focal_length1 = RandomUniformReal(100.0, 5000.0);
      const double focal_length2 = RandomUniformReal(100.0, 5000.0);
      EXPECT_NEAR(EvaluateResidual(cost_functor, focal_length1, focal_length2)
                      .squaredNorm(),
                  NormalizedEssentialDistanceSq(F,
                                                focal_length1,
                                                principal_point1,
                                                focal_length2,
                                                principal_point2),
                  1e-10);
    }
  }
}

TEST(EssentialManifoldFocalLengthSameCameraCostFunctor, Nominal) {
  const double focal_length = 1000;
  const Eigen::Vector2d principal_point(640, 480);
  for (int i = 0; i < 10; ++i) {
    const Eigen::Matrix3d F = FundamentalFromEssentialMatrix(
        CalibrationMatrix(focal_length, principal_point),
        EssentialMatrixFromPose(
            Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>())),
        CalibrationMatrix(focal_length, principal_point));
    const double initial_focal_length = RandomUniformReal(100.0, 5000.0);
    const EssentialManifoldFocalLengthSameCameraCostFunctor cost_functor(
        F, principal_point, initial_focal_length);
    const EssentialManifoldFocalLengthCostFunctor reference_cost_functor(
        F,
        principal_point,
        principal_point,
        initial_focal_length,
        initial_focal_length);
    for (const double test_focal_length :
         {focal_length, 0.7 * focal_length, 1.3 * focal_length}) {
      const double log_focal_length = std::log(test_focal_length);
      Eigen::Vector2d residual;
      EXPECT_TRUE(cost_functor(&log_focal_length, residual.data()));
      EXPECT_EQ(
          residual,
          EvaluateResidual(
              reference_cost_functor, test_focal_length, test_focal_length));
      if (test_focal_length == focal_length) {
        EXPECT_LT(residual.norm(), 1e-8);
      } else {
        EXPECT_GT(residual.norm(), 1e-4);
      }
    }
  }
}

}  // namespace
}  // namespace colmap

// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/cost_functions/calibration.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/random_eigen.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(FetzerFocalLengthCostFunctor, ConvexCostLandscape) {
  constexpr int kNumTrials = 10;
  for (int i = 0; i < kNumTrials; ++i) {
    const double focal_length1 = 128;
    const double focal_length2 = 256;
    const Eigen::Vector2d pp1(320, 240);
    const Eigen::Vector2d pp2(480, 320);
    const Rigid3d cam2_from_cam1(RandomEigenQuaterniond(),
                                 RandomEigenVectord<3>());

    Eigen::Matrix3d K1;
    K1 << focal_length1, 0, pp1(0), 0, focal_length1, pp1(1), 0, 0, 1;
    Eigen::Matrix3d K2;
    K2 << focal_length2, 0, pp2(0), 0, focal_length2, pp2(1), 0, 0, 1;

    const Eigen::Matrix3d F = FundamentalFromEssentialMatrix(
        K2, EssentialMatrixFromPose(cam2_from_cam1), K1);

    FetzerFocalLengthCostFunctor cost_functor(F, pp1, pp2);

    Eigen::VectorXd optimal_residual(2);
    EXPECT_TRUE(
        cost_functor(&focal_length1, &focal_length2, optimal_residual.data()));
    EXPECT_LT(optimal_residual.norm(), 1e-8);

    double previous_cost = -1e-9;
    double modified_focal_length1 = focal_length1;
    double modified_focal_length2 = focal_length2;
    for (int j = 0; j < 10; ++j) {
      Eigen::VectorXd residual(2);
      EXPECT_TRUE(cost_functor(
          &modified_focal_length1, &modified_focal_length2, residual.data()));
      const double cost = residual.norm();
      EXPECT_GT(cost, previous_cost);
      previous_cost = cost;
      modified_focal_length1 *= 1.05;
      modified_focal_length2 *= 1.05;
    }

    previous_cost = -1e-9;
    modified_focal_length1 = focal_length1;
    modified_focal_length2 = focal_length2;
    for (int j = 0; j < 10; ++j) {
      Eigen::VectorXd residual(2);
      EXPECT_TRUE(cost_functor(
          &modified_focal_length1, &modified_focal_length2, residual.data()));
      const double cost = residual.norm();
      EXPECT_GT(cost, previous_cost);
      previous_cost = cost;
      modified_focal_length1 *= 0.95;
      modified_focal_length2 *= 0.95;
    }
  }
}

TEST(FetzerFocalLengthSameCameraCostFunctor, ConvexCostLandscape) {
  constexpr int kNumTrials = 10;
  for (int i = 0; i < kNumTrials; ++i) {
    const double focal_length = 128;
    const Eigen::Vector2d pp(320, 240);
    const Rigid3d cam2_from_cam1(RandomEigenQuaterniond(),
                                 RandomEigenVectord<3>());

    Eigen::Matrix3d K;
    K << focal_length, 0, pp(0), 0, focal_length, pp(1), 0, 0, 1;

    const Eigen::Matrix3d F = FundamentalFromEssentialMatrix(
        K, EssentialMatrixFromPose(cam2_from_cam1), K);

    FetzerFocalLengthSameCameraCostFunctor cost_functor(F, pp);

    Eigen::VectorXd optimal_residual(2);
    EXPECT_TRUE(cost_functor(&focal_length, optimal_residual.data()));
    EXPECT_LT(optimal_residual.norm(), 1e-8);

    double previous_cost = -1e-9;
    double modified_focal_length = focal_length;
    for (int j = 0; j < 10; ++j) {
      Eigen::VectorXd residual(2);
      EXPECT_TRUE(cost_functor(&modified_focal_length, residual.data()));
      const double cost = residual.norm();
      EXPECT_GT(cost, previous_cost);
      previous_cost = cost;
      modified_focal_length *= 1.05;
    }

    previous_cost = -1e-9;
    modified_focal_length = focal_length;
    for (int j = 0; j < 10; ++j) {
      Eigen::VectorXd residual(2);
      EXPECT_TRUE(cost_functor(&modified_focal_length, residual.data()));
      const double cost = residual.norm();
      EXPECT_GT(cost, previous_cost);
      previous_cost = cost;
      modified_focal_length *= 0.95;
    }
  }
}

}  // namespace
}  // namespace colmap

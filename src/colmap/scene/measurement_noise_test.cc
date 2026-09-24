// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/measurement_noise.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(MeasurementNoiseModel, DefaultIsUnitIsotropic) {
  MeasurementNoiseModel model;
  EXPECT_EQ(model.Covariance(1.0), Eigen::Matrix2d::Identity());
}

TEST(MeasurementNoiseModel, ScalesQuadratically) {
  MeasurementNoiseModel model;
  model.base_sigma_px = 0.5;
  model.scale_gamma = 1.0;
  // sigma = 0.5 * 4 = 2 -> variance 4.
  EXPECT_EQ(model.Covariance(4.0), 4 * Eigen::Matrix2d::Identity());
}

TEST(MeasurementNoiseModel, DefaultIsSqrtMapping) {
  MeasurementNoiseModel model;
  EXPECT_EQ(model.scale_gamma, 0.5);
  // sigma = 1 * sqrt(9) = 3 -> variance 9.
  EXPECT_EQ(model.Covariance(9.0), 9 * Eigen::Matrix2d::Identity());
}

TEST(MeasurementNoiseModel, KeypointUsesScale) {
  MeasurementNoiseModel model;
  model.scale_gamma = 1.0;
  const FeatureKeypoint keypoint =
      FeatureKeypoint::FromShapeParameters(10, 20, 2, 2, 0, 0);
  EXPECT_FLOAT_EQ(keypoint.ComputeScale(), 2.0f);
  EXPECT_EQ(model.Covariance(keypoint), 4 * Eigen::Matrix2d::Identity());
  EXPECT_EQ(model.Covariance(keypoint),
            model.Covariance(keypoint.ComputeScale()));
}

TEST(MeasurementNoiseModel, RejectsNonPositiveInputs) {
  MeasurementNoiseModel model;
  EXPECT_THROW(model.Covariance(0.0), std::invalid_argument);
  EXPECT_THROW(model.Covariance(-1.0), std::invalid_argument);
  model.base_sigma_px = 0.0;
  EXPECT_THROW(model.Covariance(1.0), std::invalid_argument);
  model.base_sigma_px = 1.0;
  model.scale_gamma = -0.5;
  EXPECT_THROW(model.Covariance(1.0), std::invalid_argument);
}

TEST(MeasurementNoiseModel, FixedSigmaAtZeroGamma) {
  MeasurementNoiseModel model;
  model.base_sigma_px = 2.0;
  model.scale_gamma = 0.0;
  // sigma = 2 * scale^0 = 2 -> variance 4, independent of scale.
  EXPECT_EQ(model.Covariance(1.0), 4 * Eigen::Matrix2d::Identity());
  EXPECT_EQ(model.Covariance(9.0), 4 * Eigen::Matrix2d::Identity());
}

TEST(MeasurementNoiseModel, SqrtScalingAtHalfGamma) {
  MeasurementNoiseModel model;
  model.scale_gamma = 0.5;
  // sigma = 1 * sqrt(9) = 3 -> variance 9.
  EXPECT_EQ(model.Covariance(9.0), 9 * Eigen::Matrix2d::Identity());
}

TEST(MeasurementNoiseModel, CovarianceIsSPD) {
  MeasurementNoiseModel model;
  const Eigen::Matrix2d cov = model.Covariance(3.0);
  EXPECT_GT(cov.determinant(), 0.0);
  EXPECT_GT(cov(0, 0), 0.0);
  EXPECT_GT(cov(1, 1), 0.0);
}

TEST(KeypointsToPoint2Ds, ConvertsXyAndCovariance) {
  const FeatureKeypoints keypoints = {
      FeatureKeypoint::FromShapeParameters(1, 2, 1, 1, 0, 0),
      FeatureKeypoint::FromShapeParameters(3, 4, 3, 3, 0, 0),
  };
  MeasurementNoiseModel model;
  model.scale_gamma = 1.0;
  const std::vector<Point2D> points2D = KeypointsToPoint2Ds(keypoints, model);
  ASSERT_EQ(points2D.size(), 2);
  EXPECT_EQ(points2D[0].xy, Eigen::Vector2d(1, 2));
  EXPECT_EQ(points2D[1].xy, Eigen::Vector2d(3, 4));
  EXPECT_EQ(points2D[0].cov, Eigen::Matrix2f::Identity());
  EXPECT_EQ(points2D[1].cov, 9 * Eigen::Matrix2f::Identity());
  EXPECT_FALSE(points2D[0].HasPoint3D());
}

TEST(KeypointsToPoint2Ds, RespectsCustomModel) {
  MeasurementNoiseModel model;
  model.base_sigma_px = 2.0;
  const FeatureKeypoints keypoints = {
      FeatureKeypoint::FromShapeParameters(1, 2, 1, 1, 0, 0),
  };
  const std::vector<Point2D> points2D = KeypointsToPoint2Ds(keypoints, model);
  ASSERT_EQ(points2D.size(), 1);
  EXPECT_EQ(points2D[0].cov, 4 * Eigen::Matrix2f::Identity());
}

TEST(KeypointsToPoint2Ds, Empty) {
  EXPECT_TRUE(KeypointsToPoint2Ds({}).empty());
}

TEST(KeypointsToPoint2Ds, DegenerateScaleTreatedAsUnit) {
  const FeatureKeypoints keypoints = {
      FeatureKeypoint::FromShapeParameters(1, 2, 0, 0, 0, 0),
  };
  ASSERT_EQ(keypoints[0].ComputeScale(), 0.0f);
  const std::vector<Point2D> points2D = KeypointsToPoint2Ds(keypoints);
  ASSERT_EQ(points2D.size(), 1);
  EXPECT_EQ(points2D[0].cov, Eigen::Matrix2f::Identity());
}

TEST(KeypointsToPoint2Ds, SubUnitScalePreserved) {
  const FeatureKeypoints keypoints = {
      FeatureKeypoint::FromShapeParameters(1, 2, 0.5, 0.5, 0, 0),
  };
  ASSERT_FLOAT_EQ(keypoints[0].ComputeScale(), 0.5f);
  MeasurementNoiseModel model;
  model.scale_gamma = 1.0;
  const std::vector<Point2D> points2D = KeypointsToPoint2Ds(keypoints, model);
  ASSERT_EQ(points2D.size(), 1);
  // sigma = 1.0 * 0.5 -> variance 0.25.
  EXPECT_EQ(points2D[0].cov, 0.25f * Eigen::Matrix2f::Identity());
}

}  // namespace
}  // namespace colmap

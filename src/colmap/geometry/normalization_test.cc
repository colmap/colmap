// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/geometry/normalization.h"

#include "colmap/util/eigen_matchers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(ComputeBoundingBoxAndCentroid, SingleCoord) {
  const auto [bbox, centroid] =
      ComputeBoundingBoxAndCentroid(0, 1, {1}, {2}, {3});
  EXPECT_EQ(bbox.min(), Eigen::Vector3d(1, 2, 3));
  EXPECT_EQ(bbox.max(), Eigen::Vector3d(1, 2, 3));
  EXPECT_EQ(centroid, Eigen::Vector3d(1, 2, 3));
}

TEST(ComputeBoundingBoxAndCentroid, TwoCoords) {
  const auto [bbox, centroid] =
      ComputeBoundingBoxAndCentroid(0, 1, {2, -1}, {3, -2}, {4, -3});
  EXPECT_EQ(bbox.min(), Eigen::Vector3d(-1, -2, -3));
  EXPECT_EQ(bbox.max(), Eigen::Vector3d(2, 3, 4));
  EXPECT_EQ(centroid, Eigen::Vector3d(0.5, 0.5, 0.5));
}

TEST(ComputeBoundingBoxAndCentroid, EqualPercentiles) {
  for (const double percentile : {0.0, 0.5, 1.0}) {
    const auto [bbox, centroid] = ComputeBoundingBoxAndCentroid(
        percentile, percentile, {5, 1, 3}, {2, 6, 4}, {7, 9, 5});
    const Eigen::Vector3d expected(
        1 + 4 * percentile, 2 + 4 * percentile, 5 + 4 * percentile);
    EXPECT_EQ(bbox.min(), expected);
    EXPECT_EQ(bbox.max(), expected);
    EXPECT_EQ(centroid, expected);
  }
}

TEST(ComputeBoundingBoxAndCentroid, EqualPercentilesBetweenRanks) {
  const auto [bbox, centroid] = ComputeBoundingBoxAndCentroid(
      0.25, 0.25, {5, 1, 3}, {2, 6, 4}, {7, 9, 5});
  EXPECT_EQ(bbox.min(), Eigen::Vector3d(1, 2, 5));
  EXPECT_EQ(bbox.max(), Eigen::Vector3d(3, 4, 7));
  EXPECT_EQ(centroid, Eigen::Vector3d(2, 3, 6));
}

TEST(ComputeBoundingBoxAndCentroid, RepeatedCoords) {
  for (const double percentile : {0.0, 0.5, 1.0}) {
    const auto [bbox, centroid] = ComputeBoundingBoxAndCentroid(
        percentile, percentile, {1, 1, 1}, {2, 2, 2}, {3, 3, 3});
    EXPECT_EQ(bbox.min(), Eigen::Vector3d(1, 2, 3));
    EXPECT_EQ(bbox.max(), Eigen::Vector3d(1, 2, 3));
    EXPECT_EQ(centroid, Eigen::Vector3d(1, 2, 3));
  }
}

TEST(ComputeBoundingBoxAndCentroid, ThreeCoords) {
  const auto [bbox, centroid] =
      ComputeBoundingBoxAndCentroid(0, 1, {2, -1, 5}, {3, -2, 5}, {4, -3, 5});
  EXPECT_EQ(bbox.min(), Eigen::Vector3d(-1, -2, -3));
  EXPECT_EQ(bbox.max(), Eigen::Vector3d(5, 5, 5));
  EXPECT_THAT(centroid, EigenMatrixNear(Eigen::Vector3d(2, 2, 2), 1e-6));
}

TEST(ComputeBoundingBoxAndCentroid, FiveCoords) {
  const auto [bbox1, centroid1] =
      ComputeBoundingBoxAndCentroid(0,
                                    1,
                                    {2, -1, 5, 100, -100},
                                    {3, -2, 5, 100, -100},
                                    {4, -3, 5, 100, -100});
  EXPECT_EQ(bbox1.min(), Eigen::Vector3d(-100, -100, -100));
  EXPECT_EQ(bbox1.max(), Eigen::Vector3d(100, 100, 100));
  EXPECT_THAT(centroid1, EigenMatrixNear(Eigen::Vector3d(1.2, 1.2, 1.2), 1e-6));

  const auto [bbox2, centroid2] =
      ComputeBoundingBoxAndCentroid(0.3,
                                    0.7,
                                    {2, -1, 5, 100, -100},
                                    {3, -2, 5, 100, -100},
                                    {4, -3, 5, 100, -100});
  EXPECT_EQ(bbox2.min(), Eigen::Vector3d(-1, -2, -3));
  EXPECT_EQ(bbox2.max(), Eigen::Vector3d(5, 5, 5));
  EXPECT_THAT(centroid2, EigenMatrixNear(Eigen::Vector3d(2, 2, 2), 1e-6));
}

TEST(CenterAndNormalizeImagePoints, Nominal) {
  constexpr size_t kNumPoints = 11;
  std::vector<Eigen::Vector2d> points;
  points.reserve(kNumPoints);
  for (size_t i = 0; i < kNumPoints; ++i) {
    points.emplace_back(i, i);
  }

  std::vector<Eigen::Vector2d> normed_points;
  Eigen::Matrix3d matrix;
  CenterAndNormalizeImagePoints(points, &normed_points, &matrix);

  EXPECT_EQ(matrix(0, 0), 0.31622776601683794);
  EXPECT_EQ(matrix(1, 1), 0.31622776601683794);
  EXPECT_EQ(matrix(0, 2), -1.5811388300841898);
  EXPECT_EQ(matrix(1, 2), -1.5811388300841898);

  Eigen::Vector2d mean_point(0, 0);
  for (const auto& point : normed_points) {
    mean_point += point;
  }
  EXPECT_LT(std::abs(mean_point[0]), 1e-6);
  EXPECT_LT(std::abs(mean_point[1]), 1e-6);
}

}  // namespace
}  // namespace colmap

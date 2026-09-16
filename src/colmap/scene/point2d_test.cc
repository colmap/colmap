// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/point2d.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(Point2D, Default) {
  Point2D point2D;
  EXPECT_EQ(point2D.xy, Eigen::Vector2d::Zero());
  EXPECT_EQ(point2D.point3D_id, kInvalidPoint3DId);
  EXPECT_FALSE(point2D.HasPoint3D());
}

TEST(Point2D, Equals) {
  Point2D point2D;
  Point2D other = point2D;
  EXPECT_EQ(point2D, other);
  point2D.xy(0) += 1;
  EXPECT_NE(point2D, other);
  other.xy(0) += 1;
  EXPECT_EQ(point2D, other);
}

TEST(Point2D, Print) {
  Point2D point2D;
  point2D.xy = Eigen::Vector2d(1, 2);
  std::ostringstream stream;
  stream << point2D;
  EXPECT_EQ(stream.str(), "Point2D(xy=[1, 2], point3D_id=-1)");
}

TEST(Point2D, Point3DId) {
  Point2D point2D;
  EXPECT_EQ(point2D.point3D_id, kInvalidPoint3DId);
  EXPECT_FALSE(point2D.HasPoint3D());
  point2D.point3D_id = 1;
  EXPECT_EQ(point2D.point3D_id, 1);
  EXPECT_TRUE(point2D.HasPoint3D());
  point2D.point3D_id = kInvalidPoint3DId;
  EXPECT_EQ(point2D.point3D_id, kInvalidPoint3DId);
  EXPECT_FALSE(point2D.HasPoint3D());
}

}  // namespace
}  // namespace colmap

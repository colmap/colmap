// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/point3d.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(Point3D, Default) {
  Point3D point3D;
  EXPECT_EQ(point3D.xyz, Eigen::Vector3d::Zero());
  EXPECT_EQ(point3D.color, Eigen::Vector3ub::Zero());
  EXPECT_EQ(point3D.error, -1.0);
  EXPECT_FALSE(point3D.HasError());
  EXPECT_EQ(point3D.track.Length(), 0);
}

TEST(Point3D, Equals) {
  Point3D point3D;
  Point3D other = point3D;
  EXPECT_EQ(point3D, other);
  point3D.xyz(0) += 1;
  EXPECT_NE(point3D, other);
  other.xyz(0) += 1;
  EXPECT_EQ(point3D, other);
}

TEST(Point3D, Print) {
  Point3D point3D;
  point3D.xyz = Eigen::Vector3d(1, 2, 3);
  std::ostringstream stream;
  stream << point3D;
  EXPECT_EQ(stream.str(), "Point3D(xyz=[1, 2, 3], track_len=0)");
}

TEST(Point3D, Error) {
  Point3D point3D;
  EXPECT_EQ(point3D.error, -1.0);
  EXPECT_FALSE(point3D.HasError());
  point3D.error = 1.0;
  EXPECT_EQ(point3D.error, 1.0);
  EXPECT_TRUE(point3D.HasError());
}

}  // namespace
}  // namespace colmap

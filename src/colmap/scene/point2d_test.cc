// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

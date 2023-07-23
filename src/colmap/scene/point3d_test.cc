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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "colmap/scene/point3d.h"

#include <gtest/gtest.h>

namespace colmap {

TEST(Point3D, Default) {
  Point3D point3D;
  EXPECT_EQ(point3D.X(), 0);
  EXPECT_EQ(point3D.Y(), 0);
  EXPECT_EQ(point3D.Z(), 0);
  EXPECT_EQ(point3D.XYZ()[0], point3D.X());
  EXPECT_EQ(point3D.XYZ()[1], point3D.Y());
  EXPECT_EQ(point3D.XYZ()[2], point3D.Z());
  EXPECT_EQ(point3D.Color()[0], 0);
  EXPECT_EQ(point3D.Color()[1], 0);
  EXPECT_EQ(point3D.Color()[2], 0);
  EXPECT_EQ(point3D.Error(), -1.0);
  EXPECT_FALSE(point3D.HasError());
  EXPECT_EQ(point3D.Track().Length(), 0);
}

TEST(Point3D, XYZ) {
  Point3D point3D;
  EXPECT_EQ(point3D.X(), 0);
  EXPECT_EQ(point3D.Y(), 0);
  EXPECT_EQ(point3D.Z(), 0);
  EXPECT_EQ(point3D.XYZ()[0], point3D.X());
  EXPECT_EQ(point3D.XYZ()[1], point3D.Y());
  EXPECT_EQ(point3D.XYZ()[2], point3D.Z());
  point3D.SetXYZ(Eigen::Vector3d(0.1, 0.2, 0.3));
  EXPECT_EQ(point3D.X(), 0.1);
  EXPECT_EQ(point3D.Y(), 0.2);
  EXPECT_EQ(point3D.Z(), 0.3);
  EXPECT_EQ(point3D.XYZ()[0], point3D.X());
  EXPECT_EQ(point3D.XYZ()[1], point3D.Y());
  EXPECT_EQ(point3D.XYZ()[2], point3D.Z());
  point3D.XYZ() = Eigen::Vector3d(0.2, 0.3, 0.4);
  EXPECT_EQ(point3D.X(), 0.2);
  EXPECT_EQ(point3D.Y(), 0.3);
  EXPECT_EQ(point3D.Z(), 0.4);
  EXPECT_EQ(point3D.XYZ()[0], point3D.X());
  EXPECT_EQ(point3D.XYZ()[1], point3D.Y());
  EXPECT_EQ(point3D.XYZ()[2], point3D.Z());
}

TEST(Point3D, RGB) {
  Point3D point3D;
  EXPECT_EQ(point3D.Color()[0], 0);
  EXPECT_EQ(point3D.Color()[1], 0);
  EXPECT_EQ(point3D.Color()[2], 0);
  point3D.SetColor(Eigen::Vector3ub(1, 2, 3));
  EXPECT_EQ(point3D.Color()[0], 1);
  EXPECT_EQ(point3D.Color()[1], 2);
  EXPECT_EQ(point3D.Color()[2], 3);
}

TEST(Point3D, Error) {
  Point3D point3D;
  EXPECT_EQ(point3D.Error(), -1.0);
  EXPECT_FALSE(point3D.HasError());
  point3D.SetError(1.0);
  EXPECT_EQ(point3D.Error(), 1.0);
  EXPECT_TRUE(point3D.HasError());
  point3D.SetError(-1.0);
  EXPECT_EQ(point3D.Error(), -1.0);
  EXPECT_FALSE(point3D.HasError());
}

TEST(Point3D, Track) {
  Point3D point3D;
  EXPECT_EQ(point3D.Track().Length(), 0);
  point3D.SetTrack(Track());
  EXPECT_EQ(point3D.Track().Length(), 0);
  Track track;
  track.AddElement(0, 1);
  track.AddElement(0, 2);
  point3D.SetTrack(track);
  EXPECT_EQ(point3D.Track().Length(), 2);
  track.AddElement(0, 3);
  EXPECT_EQ(point3D.Track().Length(), 2);
}

}  // namespace colmap

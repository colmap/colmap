// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#include "colmap/util/ply.h"

#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(PlyPoint, DefaultConstructor) {
  PlyPoint point;
  EXPECT_EQ(point.x, 0.0f);
  EXPECT_EQ(point.y, 0.0f);
  EXPECT_EQ(point.z, 0.0f);
  EXPECT_EQ(point.nx, 0.0f);
  EXPECT_EQ(point.ny, 0.0f);
  EXPECT_EQ(point.nz, 0.0f);
  EXPECT_EQ(point.r, 0);
  EXPECT_EQ(point.g, 0);
  EXPECT_EQ(point.b, 0);
}

TEST(PlyMeshVertex, DefaultConstructor) {
  PlyMeshVertex vertex;
  EXPECT_EQ(vertex.x, 0.0f);
  EXPECT_EQ(vertex.y, 0.0f);
  EXPECT_EQ(vertex.z, 0.0f);
}

TEST(PlyMeshVertex, ParameterizedConstructor) {
  PlyMeshVertex vertex(1.5f, 2.5f, 3.5f);
  EXPECT_EQ(vertex.x, 1.5f);
  EXPECT_EQ(vertex.y, 2.5f);
  EXPECT_EQ(vertex.z, 3.5f);
}

TEST(PlyMeshFace, DefaultConstructor) {
  PlyMeshFace face;
  EXPECT_EQ(face.vertex_idx1, 0);
  EXPECT_EQ(face.vertex_idx2, 0);
  EXPECT_EQ(face.vertex_idx3, 0);
}

TEST(PlyMeshFace, ParameterizedConstructor) {
  PlyMeshFace face(10, 20, 30);
  EXPECT_EQ(face.vertex_idx1, 10);
  EXPECT_EQ(face.vertex_idx2, 20);
  EXPECT_EQ(face.vertex_idx3, 30);
}

TEST(Ply, RoundTripTextPlyPointsFullData) {
  const std::string test_dir = CreateTestDir();
  const std::string test_file = test_dir + "/test.ply";

  std::vector<PlyPoint> original_points;

  // Create test points with full data
  for (int i = 0; i < 3; ++i) {
    PlyPoint p;
    p.x = i * 1.0f;
    p.y = i * 2.0f;
    p.z = i * 3.0f;
    p.nx = i * 0.1f;
    p.ny = i * 0.2f;
    p.nz = i * 0.3f;
    p.r = i * 10;
    p.g = i * 20;
    p.b = i * 30;
    original_points.push_back(p);
  }

  WriteTextPlyPoints(test_file, original_points, true, true);

  std::vector<PlyPoint> loaded_points = ReadPly(test_file);

  ASSERT_EQ(loaded_points.size(), original_points.size());

  for (size_t i = 0; i < original_points.size(); ++i) {
    EXPECT_EQ(loaded_points[i].x, original_points[i].x);
    EXPECT_EQ(loaded_points[i].y, original_points[i].y);
    EXPECT_EQ(loaded_points[i].z, original_points[i].z);
    EXPECT_EQ(loaded_points[i].nx, original_points[i].nx);
    EXPECT_EQ(loaded_points[i].ny, original_points[i].ny);
    EXPECT_EQ(loaded_points[i].nz, original_points[i].nz);
    EXPECT_EQ(loaded_points[i].r, original_points[i].r);
    EXPECT_EQ(loaded_points[i].g, original_points[i].g);
    EXPECT_EQ(loaded_points[i].b, original_points[i].b);
  }
}

TEST(Ply, RoundTripTextPlyPointsXYZOnly) {
  const std::string test_dir = CreateTestDir();
  const std::string test_file = test_dir + "/test.ply";

  std::vector<PlyPoint> original_points;

  PlyPoint p;
  p.x = 1.5f;
  p.y = 2.5f;
  p.z = 3.5f;
  p.nx = 0.15f;
  p.ny = 0.25f;
  p.nz = 0.35f;
  p.r = 15;
  p.g = 25;
  p.b = 35;
  original_points.push_back(p);

  WriteTextPlyPoints(test_file, original_points, false, false);

  std::vector<PlyPoint> loaded_points = ReadPly(test_file);

  ASSERT_EQ(loaded_points.size(), 1);
  EXPECT_EQ(loaded_points[0].x, 1.5f);
  EXPECT_EQ(loaded_points[0].y, 2.5f);
  EXPECT_EQ(loaded_points[0].z, 3.5f);
  // Normals and colors should be default (0)
  EXPECT_EQ(loaded_points[0].nx, 0.0f);
  EXPECT_EQ(loaded_points[0].ny, 0.0f);
  EXPECT_EQ(loaded_points[0].nz, 0.0f);
  EXPECT_EQ(loaded_points[0].r, 0);
  EXPECT_EQ(loaded_points[0].g, 0);
  EXPECT_EQ(loaded_points[0].b, 0);
}

TEST(Ply, RoundTripBinaryPlyPointsFullData) {
  const std::string test_dir = CreateTestDir();
  const std::string test_file = test_dir + "/test.ply";

  std::vector<PlyPoint> original_points;

  // Create test points
  for (int i = 0; i < 5; ++i) {
    PlyPoint p;
    p.x = i * 1.5f;
    p.y = i * 2.5f;
    p.z = i * 3.5f;
    p.nx = i * 0.15f;
    p.ny = i * 0.25f;
    p.nz = i * 0.35f;
    p.r = i * 15;
    p.g = i * 25;
    p.b = i * 35;
    original_points.push_back(p);
  }

  WriteBinaryPlyPoints(test_file, original_points, true, true);

  std::vector<PlyPoint> loaded_points = ReadPly(test_file);

  ASSERT_EQ(loaded_points.size(), original_points.size());

  for (size_t i = 0; i < original_points.size(); ++i) {
    EXPECT_EQ(loaded_points[i].x, original_points[i].x);
    EXPECT_EQ(loaded_points[i].y, original_points[i].y);
    EXPECT_EQ(loaded_points[i].z, original_points[i].z);
    EXPECT_EQ(loaded_points[i].nx, original_points[i].nx);
    EXPECT_EQ(loaded_points[i].ny, original_points[i].ny);
    EXPECT_EQ(loaded_points[i].nz, original_points[i].nz);
    EXPECT_EQ(loaded_points[i].r, original_points[i].r);
    EXPECT_EQ(loaded_points[i].g, original_points[i].g);
    EXPECT_EQ(loaded_points[i].b, original_points[i].b);
  }
}

TEST(Ply, RoundTripBinaryPlyPointsXYZOnly) {
  const std::string test_dir = CreateTestDir();
  const std::string test_file = test_dir + "/test.ply";

  std::vector<PlyPoint> original_points;

  PlyPoint p;
  p.x = 10.5f;
  p.y = 20.5f;
  p.z = 30.5f;
  p.nx = 0.15f;
  p.ny = 0.25f;
  p.nz = 0.35f;
  p.r = 15;
  p.g = 25;
  p.b = 35;
  original_points.push_back(p);

  WriteBinaryPlyPoints(test_file, original_points, false, false);

  std::vector<PlyPoint> loaded_points = ReadPly(test_file);

  ASSERT_EQ(loaded_points.size(), 1);
  EXPECT_EQ(loaded_points[0].x, 10.5f);
  EXPECT_EQ(loaded_points[0].y, 20.5f);
  EXPECT_EQ(loaded_points[0].z, 30.5f);
  // Normals and colors should be default (0)
  EXPECT_EQ(loaded_points[0].nx, 0.0f);
  EXPECT_EQ(loaded_points[0].ny, 0.0f);
  EXPECT_EQ(loaded_points[0].nz, 0.0f);
  EXPECT_EQ(loaded_points[0].r, 0);
  EXPECT_EQ(loaded_points[0].g, 0);
  EXPECT_EQ(loaded_points[0].b, 0);
}

}  // namespace
}  // namespace colmap

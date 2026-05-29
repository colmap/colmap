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

#include "colmap/mvs/mesh_simplification.h"

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

PlyMesh CreateTetrahedronMesh() {
  PlyMesh mesh;
  mesh.vertices.emplace_back(0.0f, 0.0f, 0.0f, 255, 0, 0);
  mesh.vertices.emplace_back(1.0f, 0.0f, 0.0f, 0, 255, 0);
  mesh.vertices.emplace_back(
      0.5f, static_cast<float>(std::sqrt(3.0) / 2.0), 0.0f, 0, 0, 255);
  mesh.vertices.emplace_back(0.5f,
                             static_cast<float>(std::sqrt(3.0) / 6.0),
                             static_cast<float>(std::sqrt(6.0) / 3.0),
                             255,
                             255,
                             0);

  mesh.faces.emplace_back(0, 2, 1);
  mesh.faces.emplace_back(0, 1, 3);
  mesh.faces.emplace_back(1, 2, 3);
  mesh.faces.emplace_back(0, 3, 2);
  return mesh;
}

void AddGridFaces(PlyMesh& mesh, const int n) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const size_t v00 = j * (n + 1) + i;
      const size_t v10 = j * (n + 1) + (i + 1);
      const size_t v01 = (j + 1) * (n + 1) + i;
      const size_t v11 = (j + 1) * (n + 1) + (i + 1);
      mesh.faces.emplace_back(v00, v10, v11);
      mesh.faces.emplace_back(v00, v11, v01);
    }
  }
}

PlyMesh CreateGridMesh(const int n) {
  PlyMesh mesh;
  for (int j = 0; j <= n; ++j) {
    for (int i = 0; i <= n; ++i) {
      mesh.vertices.emplace_back(static_cast<float>(i),
                                 static_cast<float>(j),
                                 0.0f,
                                 static_cast<uint8_t>(i * 255 / std::max(n, 1)),
                                 static_cast<uint8_t>(j * 255 / std::max(n, 1)),
                                 128);
    }
  }
  AddGridFaces(mesh, n);
  return mesh;
}

PlyMesh CreateWavyGridMesh(const int n) {
  PlyMesh mesh;
  for (int j = 0; j <= n; ++j) {
    for (int i = 0; i <= n; ++i) {
      const float x = static_cast<float>(i);
      const float y = static_cast<float>(j);
      const float z = std::sin(x * 0.5f) * std::cos(y * 0.5f);
      mesh.vertices.emplace_back(x, y, z);
    }
  }
  AddGridFaces(mesh, n);
  return mesh;
}

TEST(MeshSimplificationOptions, DefaultsAreValid) {
  MeshSimplificationOptions options;
  EXPECT_TRUE(options.Check());
}

TEST(MeshSimplificationOptions, InvalidRatio) {
  MeshSimplificationOptions options;

  options.target_face_ratio = 0.0;
  EXPECT_FALSE(options.Check());

  options.target_face_ratio = -0.5;
  EXPECT_FALSE(options.Check());

  options.target_face_ratio = 1.5;
  EXPECT_FALSE(options.Check());
}

TEST(MeshSimplificationOptions, InvalidMaxError) {
  MeshSimplificationOptions options;
  options.max_error = -1.0;
  EXPECT_FALSE(options.Check());
}

TEST(SimplifyMesh, IdentityWithRatioOne) {
  const auto mesh = CreateTetrahedronMesh();

  MeshSimplificationOptions options;
  options.target_face_ratio = 1.0;

  const auto result = SimplifyMesh(mesh, options);

  EXPECT_EQ(result.faces.size(), mesh.faces.size());
  EXPECT_EQ(result.vertices.size(), mesh.vertices.size());
}

TEST(SimplifyMesh, ReducesFaceCount) {
  const auto mesh = CreateGridMesh(10);  // 200 faces
  ASSERT_EQ(mesh.faces.size(), 200);

  MeshSimplificationOptions options;
  options.target_face_ratio = 0.5;

  const auto result = SimplifyMesh(mesh, options);

  // Should be approximately 100 faces, allow some tolerance.
  EXPECT_LE(result.faces.size(), 110);
  EXPECT_GE(result.faces.size(), 50);
  EXPECT_LT(result.faces.size(), mesh.faces.size());
}

TEST(SimplifyMesh, MaxErrorThreshold) {
  // Use a wavy surface so collapses have non-zero quadric error.
  const auto mesh = CreateWavyGridMesh(10);  // 200 faces
  ASSERT_EQ(mesh.faces.size(), 200);

  MeshSimplificationOptions options;
  options.target_face_ratio = 0.1;

  // Without error threshold.
  options.max_error = 0.0;
  const auto result_no_limit = SimplifyMesh(mesh, options);

  // With strict error threshold.
  options.max_error = 1e-6;
  const auto result_strict = SimplifyMesh(mesh, options);

  // Strict threshold should preserve more faces.
  EXPECT_GT(result_strict.faces.size(), result_no_limit.faces.size());
}

TEST(SimplifyMesh, PreservesColorsAtRatioOne) {
  const auto mesh = CreateTetrahedronMesh();

  MeshSimplificationOptions options;
  options.target_face_ratio = 1.0;

  const auto result = SimplifyMesh(mesh, options);

  ASSERT_EQ(result.vertices.size(), mesh.vertices.size());
  for (size_t i = 0; i < mesh.vertices.size(); ++i) {
    EXPECT_EQ(result.vertices[i].r, mesh.vertices[i].r);
    EXPECT_EQ(result.vertices[i].g, mesh.vertices[i].g);
    EXPECT_EQ(result.vertices[i].b, mesh.vertices[i].b);
  }
}

TEST(SimplifyMesh, OutputMeshIsValid) {
  const auto mesh = CreateGridMesh(10);

  MeshSimplificationOptions options;
  options.target_face_ratio = 0.5;

  const auto result = SimplifyMesh(mesh, options);

  EXPECT_GT(result.faces.size(), 0);
  EXPECT_GT(result.vertices.size(), 0);

  for (const auto& face : result.faces) {
    // All face indices must be in-bounds.
    EXPECT_LT(face.vertex_idx1, result.vertices.size());
    EXPECT_LT(face.vertex_idx2, result.vertices.size());
    EXPECT_LT(face.vertex_idx3, result.vertices.size());
    // No degenerate faces.
    EXPECT_NE(face.vertex_idx1, face.vertex_idx2);
    EXPECT_NE(face.vertex_idx1, face.vertex_idx3);
    EXPECT_NE(face.vertex_idx2, face.vertex_idx3);
  }
}

TEST(SimplifyMesh, BoundaryPreservation) {
  const int n = 10;
  const auto mesh = CreateGridMesh(n);

  MeshSimplificationOptions options;
  options.target_face_ratio = 0.3;

  // With high boundary weight.
  options.boundary_weight = 1e6;
  const auto result_high = SimplifyMesh(mesh, options);

  // Verify that the bounding box is approximately preserved.
  float min_x = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float min_y = std::numeric_limits<float>::max();
  float max_y = std::numeric_limits<float>::lowest();
  for (const auto& v : result_high.vertices) {
    min_x = std::min(min_x, v.x);
    max_x = std::max(max_x, v.x);
    min_y = std::min(min_y, v.y);
    max_y = std::max(max_y, v.y);
  }
  EXPECT_NEAR(min_x, 0.0f, 1.0f);
  EXPECT_NEAR(max_x, static_cast<float>(n), 1.0f);
  EXPECT_NEAR(min_y, 0.0f, 1.0f);
  EXPECT_NEAR(max_y, static_cast<float>(n), 1.0f);

  // Compare with no boundary weight: high weight should preserve
  // at least as many faces (within a small tolerance for different
  // collapse orderings due to numerical precision).
  options.boundary_weight = 0.0;
  const auto result_none = SimplifyMesh(mesh, options);
  EXPECT_GE(result_high.faces.size() + 5, result_none.faces.size());
}

TEST(SimplifyMesh, SingleTriangle) {
  PlyMesh mesh;
  mesh.vertices.emplace_back(0.0f, 0.0f, 0.0f, 255, 0, 0);
  mesh.vertices.emplace_back(1.0f, 0.0f, 0.0f, 0, 255, 0);
  mesh.vertices.emplace_back(0.0f, 1.0f, 0.0f, 0, 0, 255);
  mesh.faces.emplace_back(0, 1, 2);

  MeshSimplificationOptions options;
  options.target_face_ratio = 0.5;

  const auto result = SimplifyMesh(mesh, options);

  // A single triangle cannot be simplified below 1 face.
  EXPECT_EQ(result.faces.size(), 1);
  EXPECT_EQ(result.vertices.size(), 3);
}

TEST(SimplifyMesh, OutOfBoundsVertexIndex) {
  PlyMesh mesh;
  mesh.vertices.emplace_back(0.0f, 0.0f, 0.0f, 255, 0, 0);
  mesh.vertices.emplace_back(1.0f, 0.0f, 0.0f, 0, 255, 0);
  mesh.vertices.emplace_back(0.0f, 1.0f, 0.0f, 0, 0, 255);
  // Face references vertex index 5, which is out of bounds.
  mesh.faces.emplace_back(0, 1, 5);

  MeshSimplificationOptions options;
  options.target_face_ratio = 0.5;

  EXPECT_THROW(SimplifyMesh(mesh, options), std::exception);
}

TEST(SimplifyMesh, LargerMeshStressTest) {
  const auto mesh = CreateGridMesh(50);  // 5000 faces
  ASSERT_EQ(mesh.faces.size(), 5000);

  MeshSimplificationOptions options;
  options.target_face_ratio = 0.1;

  const auto result = SimplifyMesh(mesh, options);

  EXPECT_LE(result.faces.size(), 600);
  EXPECT_GE(result.faces.size(), 100);
  EXPECT_LT(result.faces.size(), mesh.faces.size());

  // Verify output validity.
  for (const auto& face : result.faces) {
    EXPECT_LT(face.vertex_idx1, result.vertices.size());
    EXPECT_LT(face.vertex_idx2, result.vertices.size());
    EXPECT_LT(face.vertex_idx3, result.vertices.size());
  }
}

TEST(SimplifyMesh, OptimalVertexPlacement) {
  // Simplify a flat grid mesh and verify that the QEM optimal vertex
  // positions remain on the original surface (z=0 plane) and within
  // the bounding box of the original mesh.
  const auto mesh = CreateGridMesh(4);  // 32 faces

  MeshSimplificationOptions options;
  options.target_face_ratio = 0.25;
  options.boundary_weight = 0.0;

  const auto result = SimplifyMesh(mesh, options);

  EXPECT_LT(result.faces.size(), mesh.faces.size());
  EXPECT_GT(result.vertices.size(), 0);

  for (const auto& v : result.vertices) {
    // All vertices should remain on the z=0 plane.
    EXPECT_NEAR(v.z, 0.0f, 1e-5f);
    // All vertices should remain within the original bounding box.
    EXPECT_GE(v.x, -0.01f);
    EXPECT_LE(v.x, 4.01f);
    EXPECT_GE(v.y, -0.01f);
    EXPECT_LE(v.y, 4.01f);
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap

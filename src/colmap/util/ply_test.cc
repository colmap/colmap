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

#include <fstream>

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
  EXPECT_EQ(vertex.r, 200);
  EXPECT_EQ(vertex.g, 200);
  EXPECT_EQ(vertex.b, 200);
}

TEST(PlyMeshVertex, ParameterizedConstructor) {
  PlyMeshVertex vertex(1.5f, 2.5f, 3.5f);
  EXPECT_EQ(vertex.x, 1.5f);
  EXPECT_EQ(vertex.y, 2.5f);
  EXPECT_EQ(vertex.z, 3.5f);
  EXPECT_EQ(vertex.r, 200);
  EXPECT_EQ(vertex.g, 200);
  EXPECT_EQ(vertex.b, 200);
}

TEST(PlyMeshVertex, ColorConstructor) {
  PlyMeshVertex vertex(1.0f, 2.0f, 3.0f, 10, 20, 30);
  EXPECT_EQ(vertex.x, 1.0f);
  EXPECT_EQ(vertex.y, 2.0f);
  EXPECT_EQ(vertex.z, 3.0f);
  EXPECT_EQ(vertex.r, 10);
  EXPECT_EQ(vertex.g, 20);
  EXPECT_EQ(vertex.b, 30);
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
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "test.ply";

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
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "test.ply";

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
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "test.ply";

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
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "test.ply";

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

TEST(Ply, RoundTripTextPlyMesh) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "mesh.ply";

  PlyMesh original_mesh;
  original_mesh.vertices.emplace_back(0.0f, 0.0f, 0.0f);
  original_mesh.vertices.emplace_back(1.0f, 0.0f, 0.0f);
  original_mesh.vertices.emplace_back(0.0f, 1.0f, 0.0f);
  original_mesh.vertices.emplace_back(1.0f, 1.0f, 1.0f);

  original_mesh.faces.emplace_back(0, 1, 2);
  original_mesh.faces.emplace_back(1, 3, 2);

  WriteTextPlyMesh(test_file, PlyTexturedMesh{original_mesh});

  PlyMesh loaded_mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(loaded_mesh.vertices.size(), original_mesh.vertices.size());
  ASSERT_EQ(loaded_mesh.faces.size(), original_mesh.faces.size());

  for (size_t i = 0; i < original_mesh.vertices.size(); ++i) {
    EXPECT_EQ(loaded_mesh.vertices[i].x, original_mesh.vertices[i].x);
    EXPECT_EQ(loaded_mesh.vertices[i].y, original_mesh.vertices[i].y);
    EXPECT_EQ(loaded_mesh.vertices[i].z, original_mesh.vertices[i].z);
    // No colors in file, should get defaults.
    EXPECT_EQ(loaded_mesh.vertices[i].r, 200);
    EXPECT_EQ(loaded_mesh.vertices[i].g, 200);
    EXPECT_EQ(loaded_mesh.vertices[i].b, 200);
  }

  for (size_t i = 0; i < original_mesh.faces.size(); ++i) {
    EXPECT_EQ(loaded_mesh.faces[i].vertex_idx1,
              original_mesh.faces[i].vertex_idx1);
    EXPECT_EQ(loaded_mesh.faces[i].vertex_idx2,
              original_mesh.faces[i].vertex_idx2);
    EXPECT_EQ(loaded_mesh.faces[i].vertex_idx3,
              original_mesh.faces[i].vertex_idx3);
  }
}

TEST(Ply, RoundTripBinaryPlyMesh) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "mesh.ply";

  PlyMesh original_mesh;
  original_mesh.vertices.emplace_back(0.0f, 0.0f, 0.0f);
  original_mesh.vertices.emplace_back(1.0f, 0.0f, 0.0f);
  original_mesh.vertices.emplace_back(0.0f, 1.0f, 0.0f);
  original_mesh.vertices.emplace_back(1.0f, 1.0f, 1.0f);

  original_mesh.faces.emplace_back(0, 1, 2);
  original_mesh.faces.emplace_back(1, 3, 2);

  WriteBinaryPlyMesh(test_file, PlyTexturedMesh{original_mesh});

  PlyMesh loaded_mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(loaded_mesh.vertices.size(), original_mesh.vertices.size());
  ASSERT_EQ(loaded_mesh.faces.size(), original_mesh.faces.size());

  for (size_t i = 0; i < original_mesh.vertices.size(); ++i) {
    EXPECT_EQ(loaded_mesh.vertices[i].x, original_mesh.vertices[i].x);
    EXPECT_EQ(loaded_mesh.vertices[i].y, original_mesh.vertices[i].y);
    EXPECT_EQ(loaded_mesh.vertices[i].z, original_mesh.vertices[i].z);
    // No colors in file, should get defaults.
    EXPECT_EQ(loaded_mesh.vertices[i].r, 200);
    EXPECT_EQ(loaded_mesh.vertices[i].g, 200);
    EXPECT_EQ(loaded_mesh.vertices[i].b, 200);
  }

  for (size_t i = 0; i < original_mesh.faces.size(); ++i) {
    EXPECT_EQ(loaded_mesh.faces[i].vertex_idx1,
              original_mesh.faces[i].vertex_idx1);
    EXPECT_EQ(loaded_mesh.faces[i].vertex_idx2,
              original_mesh.faces[i].vertex_idx2);
    EXPECT_EQ(loaded_mesh.faces[i].vertex_idx3,
              original_mesh.faces[i].vertex_idx3);
  }
}

PlyTexturedMesh CreateTestTexturedMesh() {
  PlyTexturedMesh textured_mesh;
  textured_mesh.texture_file = "texture.png";

  textured_mesh.mesh.vertices.emplace_back(0.0f, 0.0f, 0.0f);
  textured_mesh.mesh.vertices.emplace_back(1.0f, 0.0f, 0.0f);
  textured_mesh.mesh.vertices.emplace_back(0.0f, 1.0f, 0.0f);
  textured_mesh.mesh.vertices.emplace_back(1.0f, 1.0f, 1.0f);

  textured_mesh.mesh.faces.emplace_back(0, 1, 2);
  textured_mesh.mesh.faces.emplace_back(1, 3, 2);

  // 6 UV floats per face (u1,v1, u2,v2, u3,v3)
  // Face 0
  textured_mesh.face_uvs.push_back(0.0f);
  textured_mesh.face_uvs.push_back(0.0f);
  textured_mesh.face_uvs.push_back(1.0f);
  textured_mesh.face_uvs.push_back(0.0f);
  textured_mesh.face_uvs.push_back(0.0f);
  textured_mesh.face_uvs.push_back(1.0f);
  // Face 1
  textured_mesh.face_uvs.push_back(0.5f);
  textured_mesh.face_uvs.push_back(0.5f);
  textured_mesh.face_uvs.push_back(1.0f);
  textured_mesh.face_uvs.push_back(1.0f);
  textured_mesh.face_uvs.push_back(0.25f);
  textured_mesh.face_uvs.push_back(0.75f);

  return textured_mesh;
}

void VerifyTexturedMesh(const PlyTexturedMesh& loaded,
                        const PlyTexturedMesh& original) {
  EXPECT_EQ(loaded.texture_file, original.texture_file);

  ASSERT_EQ(loaded.mesh.vertices.size(), original.mesh.vertices.size());
  ASSERT_EQ(loaded.mesh.faces.size(), original.mesh.faces.size());
  ASSERT_EQ(loaded.face_uvs.size(), original.face_uvs.size());

  for (size_t i = 0; i < original.mesh.vertices.size(); ++i) {
    EXPECT_EQ(loaded.mesh.vertices[i].x, original.mesh.vertices[i].x);
    EXPECT_EQ(loaded.mesh.vertices[i].y, original.mesh.vertices[i].y);
    EXPECT_EQ(loaded.mesh.vertices[i].z, original.mesh.vertices[i].z);
  }

  for (size_t i = 0; i < original.mesh.faces.size(); ++i) {
    EXPECT_EQ(loaded.mesh.faces[i].vertex_idx1,
              original.mesh.faces[i].vertex_idx1);
    EXPECT_EQ(loaded.mesh.faces[i].vertex_idx2,
              original.mesh.faces[i].vertex_idx2);
    EXPECT_EQ(loaded.mesh.faces[i].vertex_idx3,
              original.mesh.faces[i].vertex_idx3);
  }

  for (size_t i = 0; i < original.face_uvs.size(); ++i) {
    EXPECT_EQ(loaded.face_uvs[i], original.face_uvs[i]);
  }
}

TEST(Ply, RoundTripTextTexturedPlyMesh) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "textured_mesh.ply";

  PlyTexturedMesh original = CreateTestTexturedMesh();
  WriteTextPlyMesh(test_file, original);
  PlyTexturedMesh loaded = ReadPlyMesh(test_file);
  VerifyTexturedMesh(loaded, original);
}

TEST(Ply, RoundTripBinaryTexturedPlyMesh) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "textured_mesh.ply";

  PlyTexturedMesh original = CreateTestTexturedMesh();
  WriteBinaryPlyMesh(test_file, original);
  PlyTexturedMesh loaded = ReadPlyMesh(test_file);
  VerifyTexturedMesh(loaded, original);
}

TEST(Ply, ReadPlyMeshWithoutTexcoords) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "mesh.ply";

  // Write a plain mesh (no texcoords) and read it with ReadPlyMesh.
  PlyMesh plain_mesh;
  plain_mesh.vertices.emplace_back(0.0f, 0.0f, 0.0f);
  plain_mesh.vertices.emplace_back(1.0f, 0.0f, 0.0f);
  plain_mesh.vertices.emplace_back(0.0f, 1.0f, 0.0f);
  plain_mesh.faces.emplace_back(0, 1, 2);

  WriteTextPlyMesh(test_file, PlyTexturedMesh{plain_mesh});
  PlyTexturedMesh loaded = ReadPlyMesh(test_file);

  ASSERT_EQ(loaded.mesh.vertices.size(), 3);
  ASSERT_EQ(loaded.mesh.faces.size(), 1);
  EXPECT_TRUE(loaded.face_uvs.empty());
  EXPECT_TRUE(loaded.texture_file.empty());
}

TEST(Ply, ReadTextPlyMeshWithVertexColors) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "mesh_colors.ply";

  // Write a PLY mesh with per-vertex colors manually.
  {
    std::ofstream file(test_file);
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex 3\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "element face 1\n";
    file << "property list uchar int vertex_index\n";
    file << "end_header\n";
    file << "0 0 0 255 0 0\n";
    file << "1 0 0 0 255 0\n";
    file << "0 1 0 0 0 255\n";
    file << "3 0 1 2\n";
  }

  PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 3);
  ASSERT_EQ(mesh.faces.size(), 1);

  EXPECT_EQ(mesh.vertices[0].r, 255);
  EXPECT_EQ(mesh.vertices[0].g, 0);
  EXPECT_EQ(mesh.vertices[0].b, 0);
  EXPECT_EQ(mesh.vertices[1].r, 0);
  EXPECT_EQ(mesh.vertices[1].g, 255);
  EXPECT_EQ(mesh.vertices[1].b, 0);
  EXPECT_EQ(mesh.vertices[2].r, 0);
  EXPECT_EQ(mesh.vertices[2].g, 0);
  EXPECT_EQ(mesh.vertices[2].b, 255);

  EXPECT_EQ(mesh.faces[0].vertex_idx1, 0);
  EXPECT_EQ(mesh.faces[0].vertex_idx2, 1);
  EXPECT_EQ(mesh.faces[0].vertex_idx3, 2);
}

TEST(Ply, ReadBinaryPlyMeshWithVertexColors) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "mesh_colors_bin.ply";

  // Write a binary PLY mesh with per-vertex colors manually.
  {
    std::ofstream text_file(test_file);
    text_file << "ply\n";
    text_file << "format binary_little_endian 1.0\n";
    text_file << "element vertex 3\n";
    text_file << "property float x\n";
    text_file << "property float y\n";
    text_file << "property float z\n";
    text_file << "property uchar red\n";
    text_file << "property uchar green\n";
    text_file << "property uchar blue\n";
    text_file << "element face 1\n";
    text_file << "property list uchar int vertex_index\n";
    text_file << "end_header\n";
    text_file.close();

    std::ofstream bin_file(test_file,
                           std::ios::out | std::ios::binary | std::ios::app);
    const float v0[] = {1.0f, 2.0f, 3.0f};
    const uint8_t c0[] = {100, 150, 200};
    bin_file.write(reinterpret_cast<const char*>(v0), sizeof(v0));
    bin_file.write(reinterpret_cast<const char*>(c0), sizeof(c0));
    const float v1[] = {4.0f, 5.0f, 6.0f};
    const uint8_t c1[] = {10, 20, 30};
    bin_file.write(reinterpret_cast<const char*>(v1), sizeof(v1));
    bin_file.write(reinterpret_cast<const char*>(c1), sizeof(c1));
    const float v2[] = {7.0f, 8.0f, 9.0f};
    const uint8_t c2[] = {50, 60, 70};
    bin_file.write(reinterpret_cast<const char*>(v2), sizeof(v2));
    bin_file.write(reinterpret_cast<const char*>(c2), sizeof(c2));
    const uint8_t num_verts = 3;
    const int idx[] = {0, 1, 2};
    bin_file.write(reinterpret_cast<const char*>(&num_verts),
                   sizeof(num_verts));
    bin_file.write(reinterpret_cast<const char*>(idx), sizeof(idx));
    bin_file.close();
  }

  const PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 3);
  ASSERT_EQ(mesh.faces.size(), 1);

  EXPECT_EQ(mesh.vertices[0].x, 1.0f);
  EXPECT_EQ(mesh.vertices[0].y, 2.0f);
  EXPECT_EQ(mesh.vertices[0].z, 3.0f);
  EXPECT_EQ(mesh.vertices[0].r, 100);
  EXPECT_EQ(mesh.vertices[0].g, 150);
  EXPECT_EQ(mesh.vertices[0].b, 200);

  EXPECT_EQ(mesh.vertices[1].x, 4.0f);
  EXPECT_EQ(mesh.vertices[1].r, 10);
  EXPECT_EQ(mesh.vertices[1].g, 20);
  EXPECT_EQ(mesh.vertices[1].b, 30);

  EXPECT_EQ(mesh.vertices[2].x, 7.0f);
  EXPECT_EQ(mesh.vertices[2].r, 50);
  EXPECT_EQ(mesh.vertices[2].g, 60);
  EXPECT_EQ(mesh.vertices[2].b, 70);
}

TEST(Ply, ReadTextPlyMeshWithExtraProperties) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "mesh_extra.ply";

  // Write a PLY mesh with normals and colors (normals should be skipped).
  {
    std::ofstream file(test_file);
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex 3\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property float nx\n";
    file << "property float ny\n";
    file << "property float nz\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "element face 1\n";
    file << "property list uchar int vertex_index\n";
    file << "end_header\n";
    file << "1 2 3 0.1 0.2 0.3 255 128 64\n";
    file << "4 5 6 0.4 0.5 0.6 32 64 96\n";
    file << "7 8 9 0.7 0.8 0.9 10 20 30\n";
    file << "3 0 1 2\n";
  }

  const PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 3);

  // Positions should be read correctly.
  EXPECT_EQ(mesh.vertices[0].x, 1.0f);
  EXPECT_EQ(mesh.vertices[0].y, 2.0f);
  EXPECT_EQ(mesh.vertices[0].z, 3.0f);
  EXPECT_EQ(mesh.vertices[1].x, 4.0f);
  EXPECT_EQ(mesh.vertices[2].x, 7.0f);

  // Colors should be read correctly despite extra normal properties.
  EXPECT_EQ(mesh.vertices[0].r, 255);
  EXPECT_EQ(mesh.vertices[0].g, 128);
  EXPECT_EQ(mesh.vertices[0].b, 64);
  EXPECT_EQ(mesh.vertices[1].r, 32);
  EXPECT_EQ(mesh.vertices[1].g, 64);
  EXPECT_EQ(mesh.vertices[1].b, 96);
  EXPECT_EQ(mesh.vertices[2].r, 10);
  EXPECT_EQ(mesh.vertices[2].g, 20);
  EXPECT_EQ(mesh.vertices[2].b, 30);
}

TEST(Ply, ReadBinaryPlyMeshWithExtraProperties) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "mesh_extra_bin.ply";

  // Write a binary PLY mesh with normals + colors.
  {
    std::ofstream text_file(test_file);
    text_file << "ply\n";
    text_file << "format binary_little_endian 1.0\n";
    text_file << "element vertex 3\n";
    text_file << "property float x\n";
    text_file << "property float y\n";
    text_file << "property float z\n";
    text_file << "property float nx\n";
    text_file << "property float ny\n";
    text_file << "property float nz\n";
    text_file << "property uchar red\n";
    text_file << "property uchar green\n";
    text_file << "property uchar blue\n";
    text_file << "element face 1\n";
    text_file << "property list uchar int vertex_index\n";
    text_file << "end_header\n";
    text_file.close();

    std::ofstream bin_file(test_file,
                           std::ios::out | std::ios::binary | std::ios::app);
    const float pos0[] = {1.0f, 2.0f, 3.0f};
    const float norm0[] = {0.1f, 0.2f, 0.3f};
    const uint8_t col0[] = {255, 128, 64};
    bin_file.write(reinterpret_cast<const char*>(pos0), sizeof(pos0));
    bin_file.write(reinterpret_cast<const char*>(norm0), sizeof(norm0));
    bin_file.write(reinterpret_cast<const char*>(col0), sizeof(col0));
    const float pos1[] = {4.0f, 5.0f, 6.0f};
    const float norm1[] = {0.4f, 0.5f, 0.6f};
    const uint8_t col1[] = {32, 64, 96};
    bin_file.write(reinterpret_cast<const char*>(pos1), sizeof(pos1));
    bin_file.write(reinterpret_cast<const char*>(norm1), sizeof(norm1));
    bin_file.write(reinterpret_cast<const char*>(col1), sizeof(col1));
    const float pos2[] = {7.0f, 8.0f, 9.0f};
    const float norm2[] = {0.7f, 0.8f, 0.9f};
    const uint8_t col2[] = {10, 20, 30};
    bin_file.write(reinterpret_cast<const char*>(pos2), sizeof(pos2));
    bin_file.write(reinterpret_cast<const char*>(norm2), sizeof(norm2));
    bin_file.write(reinterpret_cast<const char*>(col2), sizeof(col2));
    const uint8_t num_verts = 3;
    const int idx[] = {0, 1, 2};
    bin_file.write(reinterpret_cast<const char*>(&num_verts),
                   sizeof(num_verts));
    bin_file.write(reinterpret_cast<const char*>(idx), sizeof(idx));
    bin_file.close();
  }

  const PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 3);

  EXPECT_EQ(mesh.vertices[0].x, 1.0f);
  EXPECT_EQ(mesh.vertices[0].y, 2.0f);
  EXPECT_EQ(mesh.vertices[0].z, 3.0f);
  EXPECT_EQ(mesh.vertices[0].r, 255);
  EXPECT_EQ(mesh.vertices[0].g, 128);
  EXPECT_EQ(mesh.vertices[0].b, 64);

  EXPECT_EQ(mesh.vertices[1].x, 4.0f);
  EXPECT_EQ(mesh.vertices[1].r, 32);
  EXPECT_EQ(mesh.vertices[1].g, 64);
  EXPECT_EQ(mesh.vertices[1].b, 96);

  EXPECT_EQ(mesh.vertices[2].x, 7.0f);
  EXPECT_EQ(mesh.vertices[2].r, 10);
  EXPECT_EQ(mesh.vertices[2].g, 20);
  EXPECT_EQ(mesh.vertices[2].b, 30);
}

TEST(Ply, HasPlyMeshFacesWithMesh) {
  const auto test_file = CreateTestDir() / "mesh.ply";

  PlyMesh mesh;
  mesh.vertices.emplace_back(0.0f, 0.0f, 0.0f);
  mesh.vertices.emplace_back(1.0f, 0.0f, 0.0f);
  mesh.vertices.emplace_back(0.0f, 1.0f, 0.0f);
  mesh.faces.emplace_back(0, 1, 2);

  WriteTextPlyMesh(test_file, PlyTexturedMesh{mesh});
  EXPECT_TRUE(HasPlyMeshFaces(test_file));
}

TEST(Ply, HasPlyMeshFacesWithPointCloud) {
  const auto test_file = CreateTestDir() / "points.ply";

  std::vector<PlyPoint> points;
  points.emplace_back();

  WriteTextPlyPoints(test_file, points, false, false);
  EXPECT_FALSE(HasPlyMeshFaces(test_file));
}

TEST(Ply, HasPlyMeshFacesWithZeroFaces) {
  const auto test_file = CreateTestDir() / "mesh_no_faces.ply";

  PlyMesh mesh;
  WriteTextPlyMesh(test_file, PlyTexturedMesh{mesh});

  EXPECT_FALSE(HasPlyMeshFaces(test_file));
}

}  // namespace
}  // namespace colmap

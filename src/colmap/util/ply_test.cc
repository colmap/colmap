// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/ply.h"

#include "colmap/util/endian.h"
#include "colmap/util/testing.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

// Write a binary PLY mesh with the given vertex properties (e.g. "short x")
// and triangular faces. The payload callback receives a Write function that
// encodes one scalar in the file's endianness; it must write exactly
// num_vertices vertices. Pass explicitly typed values (e.g. int16_t(-300))
// so that scalars get the intended width.
template <typename WritePayload>
std::filesystem::path WriteBinaryPlyMesh(
    const std::string& file_name,
    const std::string& format,
    size_t num_vertices,
    const std::vector<std::string>& properties,
    WritePayload&& write_payload,
    const std::vector<std::array<int, 3>>& faces = {}) {
  const auto test_file = CreateTestDir() / file_name;
  {
    std::ofstream text_file(test_file);
    text_file << "ply\n";
    text_file << "format " << format << " 1.0\n";
    text_file << "element vertex " << num_vertices << "\n";
    for (const auto& property : properties) {
      text_file << "property " << property << "\n";
    }
    text_file << "element face " << faces.size() << "\n";
    text_file << "property list uchar int vertex_index\n";
    text_file << "end_header\n";
    text_file.close();

    std::ofstream binary_file(test_file,
                              std::ios::out | std::ios::binary | std::ios::app);
    const bool little_endian = (format == "binary_little_endian");
    const auto Write = [&](auto value) {
      const auto converted = little_endian ? NativeToLittleEndian(value)
                                           : NativeToBigEndian(value);
      binary_file.write(reinterpret_cast<const char*>(&converted),
                        sizeof(converted));
    };
    write_payload(Write);
    for (const auto& face : faces) {
      Write(uint8_t(3));
      for (int idx : face) {
        Write(int32_t(idx));
      }
    }
  }
  return test_file;
}

// Write an ASCII PLY mesh with the given vertex properties, vertex lines,
// and face lines.
std::filesystem::path WriteTextPlyMesh(
    const std::string& file_name,
    const std::vector<std::string>& properties,
    const std::vector<std::string>& vertex_lines,
    const std::vector<std::string>& face_lines = {}) {
  const auto test_file = CreateTestDir() / file_name;
  std::ofstream file(test_file);
  file << "ply\n";
  file << "format ascii 1.0\n";
  file << "element vertex " << vertex_lines.size() << "\n";
  for (const auto& property : properties) {
    file << "property " << property << "\n";
  }
  file << "element face " << face_lines.size() << "\n";
  file << "property list uchar int vertex_index\n";
  file << "end_header\n";
  for (const auto& vertex_line : vertex_lines) {
    file << vertex_line << "\n";
  }
  for (const auto& face_line : face_lines) {
    file << face_line << "\n";
  }
  return test_file;
}

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

TEST(Ply, ReadMeshVerticesAsPoints) {
  const auto test_dir = CreateTestDir();
  PlyMesh mesh;
  mesh.vertices.emplace_back(0.0f, 0.0f, 0.0f);
  mesh.vertices.emplace_back(1.0f, 0.0f, 0.0f);
  mesh.vertices.emplace_back(0.0f, 1.0f, 0.0f);
  mesh.faces.emplace_back(0, 1, 2);

  for (const bool binary : {false, true}) {
    SCOPED_TRACE(binary);
    const auto test_file = test_dir / (binary ? "binary.ply" : "text.ply");
    if (binary) {
      WriteBinaryPlyMesh(test_file, PlyTexturedMesh{mesh});
    } else {
      WriteTextPlyMesh(test_file, PlyTexturedMesh{mesh});
    }
    const auto points = ReadPly(test_file);
    ASSERT_EQ(points.size(), mesh.vertices.size());
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
      EXPECT_EQ(points[i].x, mesh.vertices[i].x);
      EXPECT_EQ(points[i].y, mesh.vertices[i].y);
      EXPECT_EQ(points[i].z, mesh.vertices[i].z);
    }
  }
}

TEST(Ply, ReadTextZeroVerticesIgnoresFollowingElements) {
  const auto test_file = CreateTestDir() / "empty.ply";
  std::ofstream file(test_file);
  file << "ply\nformat ascii 1.0\nelement vertex 0\n"
          "property float x\nproperty float y\nproperty float z\n"
          "element color 1\nproperty uchar red\nproperty uchar green\n"
          "property uchar blue\nend_header\n1 2 3\n";
  file.close();
  EXPECT_TRUE(ReadPly(test_file).empty());
}

TEST(Ply, ReadTextTruncatedVertices) {
  const auto test_file = CreateTestDir() / "truncated.ply";
  std::ofstream file(test_file);
  file << "ply\nformat ascii 1.0\nelement vertex 2\n"
          "property float x\nproperty float y\nproperty float z\n"
          "end_header\n1 2 3\n";
  file.close();
  EXPECT_THROW(ReadPly(test_file), std::invalid_argument);
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
  const auto test_file =
      WriteTextPlyMesh("mesh_colors.ply",
                       {"float x",
                        "float y",
                        "float z",
                        "uchar red",
                        "uchar green",
                        "uchar blue"},
                       {"0 0 0 255 0 0", "1 0 0 0 255 0", "0 1 0 0 0 255"},
                       {"3 0 1 2"});

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
  const auto test_file = WriteBinaryPlyMesh("mesh_colors_bin.ply",
                                            "binary_little_endian",
                                            3,
                                            {"float x",
                                             "float y",
                                             "float z",
                                             "uchar red",
                                             "uchar green",
                                             "uchar blue"},
                                            [](const auto& Write) {
                                              Write(1.0f);
                                              Write(2.0f);
                                              Write(3.0f);
                                              Write(uint8_t(100));
                                              Write(uint8_t(150));
                                              Write(uint8_t(200));
                                              Write(4.0f);
                                              Write(5.0f);
                                              Write(6.0f);
                                              Write(uint8_t(10));
                                              Write(uint8_t(20));
                                              Write(uint8_t(30));
                                              Write(7.0f);
                                              Write(8.0f);
                                              Write(9.0f);
                                              Write(uint8_t(50));
                                              Write(uint8_t(60));
                                              Write(uint8_t(70));
                                            },
                                            {{0, 1, 2}});

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
  // Normals should be skipped.
  const auto test_file = WriteTextPlyMesh("mesh_extra.ply",
                                          {"float x",
                                           "float y",
                                           "float z",
                                           "float nx",
                                           "float ny",
                                           "float nz",
                                           "uchar red",
                                           "uchar green",
                                           "uchar blue"},
                                          {"1 2 3 0.1 0.2 0.3 255 128 64",
                                           "4 5 6 0.4 0.5 0.6 32 64 96",
                                           "7 8 9 0.7 0.8 0.9 10 20 30"},
                                          {"3 0 1 2"});

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
  const auto test_file = WriteBinaryPlyMesh("mesh_extra_bin.ply",
                                            "binary_little_endian",
                                            3,
                                            {"float x",
                                             "float y",
                                             "float z",
                                             "float nx",
                                             "float ny",
                                             "float nz",
                                             "uchar red",
                                             "uchar green",
                                             "uchar blue"},
                                            [](const auto& Write) {
                                              Write(1.0f);
                                              Write(2.0f);
                                              Write(3.0f);
                                              Write(0.1f);
                                              Write(0.2f);
                                              Write(0.3f);
                                              Write(uint8_t(255));
                                              Write(uint8_t(128));
                                              Write(uint8_t(64));
                                              Write(4.0f);
                                              Write(5.0f);
                                              Write(6.0f);
                                              Write(0.4f);
                                              Write(0.5f);
                                              Write(0.6f);
                                              Write(uint8_t(32));
                                              Write(uint8_t(64));
                                              Write(uint8_t(96));
                                              Write(7.0f);
                                              Write(8.0f);
                                              Write(9.0f);
                                              Write(0.7f);
                                              Write(0.8f);
                                              Write(0.9f);
                                              Write(uint8_t(10));
                                              Write(uint8_t(20));
                                              Write(uint8_t(30));
                                            },
                                            {{0, 1, 2}});

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

TEST(Ply, ReadBinaryPlyMeshWithIntegerCoordinates) {
  const auto test_file = WriteBinaryPlyMesh("mesh_integer_coords.ply",
                                            "binary_little_endian",
                                            2,
                                            {"short x", "ushort y", "int z"},
                                            [](const auto& Write) {
                                              Write(int16_t(-2));
                                              Write(uint16_t(400));
                                              Write(int32_t(100000));
                                              Write(int16_t(30000));
                                              Write(uint16_t(50000));
                                              Write(int32_t(-500000));
                                            });

  const PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 2);
  EXPECT_EQ(mesh.vertices[0].x, -2.0f);
  EXPECT_EQ(mesh.vertices[0].y, 400.0f);
  EXPECT_EQ(mesh.vertices[0].z, 100000.0f);
  EXPECT_EQ(mesh.vertices[1].x, 30000.0f);
  EXPECT_EQ(mesh.vertices[1].y, 50000.0f);
  EXPECT_EQ(mesh.vertices[1].z, -500000.0f);
}

TEST(Ply, ReadBinaryPlyMeshWithDoubleCoordinates) {
  const auto test_file =
      WriteBinaryPlyMesh("mesh_double_coords.ply",
                         "binary_little_endian",
                         2,
                         {"double x", "double y", "double z"},
                         [](const auto& Write) {
                           Write(1.5);
                           Write(-2.25);
                           Write(100000.0);
                           Write(-0.5);
                           Write(0.0);
                           Write(-30000.0);
                         });

  const PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 2);
  EXPECT_EQ(mesh.vertices[0].x, 1.5f);
  EXPECT_EQ(mesh.vertices[0].y, -2.25f);
  EXPECT_EQ(mesh.vertices[0].z, 100000.0f);
  EXPECT_EQ(mesh.vertices[1].x, -0.5f);
  EXPECT_EQ(mesh.vertices[1].y, 0.0f);
  EXPECT_EQ(mesh.vertices[1].z, -30000.0f);
}

TEST(Ply, ReadBinaryPlyMeshWith8And32BitCoordinates) {
  const auto test_file = WriteBinaryPlyMesh("mesh_small_coords.ply",
                                            "binary_little_endian",
                                            2,
                                            {"char x", "uchar y", "uint z"},
                                            [](const auto& Write) {
                                              Write(int8_t(-100));
                                              Write(uint8_t(0));
                                              Write(uint32_t(12345678));
                                              Write(int8_t(100));
                                              Write(uint8_t(255));
                                              Write(uint32_t(100));
                                            });

  const PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 2);
  EXPECT_EQ(mesh.vertices[0].x, -100.0f);
  EXPECT_EQ(mesh.vertices[0].y, 0.0f);
  EXPECT_EQ(mesh.vertices[0].z, 12345678.0f);
  EXPECT_EQ(mesh.vertices[1].x, 100.0f);
  EXPECT_EQ(mesh.vertices[1].y, 255.0f);
  EXPECT_EQ(mesh.vertices[1].z, 100.0f);
}

TEST(Ply, ReadBinaryPlyMeshWithBigEndianIntegerCoordinates) {
  const auto test_file = WriteBinaryPlyMesh("mesh_big_endian_coords.ply",
                                            "binary_big_endian",
                                            2,
                                            {"short x", "int y", "float z"},
                                            [](const auto& Write) {
                                              Write(int16_t(-300));
                                              Write(int32_t(-500000));
                                              Write(1.5f);
                                              Write(int16_t(30000));
                                              Write(int32_t(500000));
                                              Write(-2.5f);
                                            });

  const PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 2);
  EXPECT_EQ(mesh.vertices[0].x, -300.0f);
  EXPECT_EQ(mesh.vertices[0].y, -500000.0f);
  EXPECT_EQ(mesh.vertices[0].z, 1.5f);
  EXPECT_EQ(mesh.vertices[1].x, 30000.0f);
  EXPECT_EQ(mesh.vertices[1].y, 500000.0f);
  EXPECT_EQ(mesh.vertices[1].z, -2.5f);
}

TEST(Ply, ReadBinaryPlyMeshWithMixedColorTypes) {
  const auto test_file = WriteBinaryPlyMesh("mesh_mixed_colors.ply",
                                            "binary_little_endian",
                                            2,
                                            {"float x",
                                             "float y",
                                             "float z",
                                             "uchar red",
                                             "ushort green",
                                             "float blue"},
                                            [](const auto& Write) {
                                              Write(1.0f);
                                              Write(2.0f);
                                              Write(3.0f);
                                              Write(uint8_t(200));
                                              Write(uint16_t(100));
                                              Write(1.0f);
                                              Write(4.0f);
                                              Write(5.0f);
                                              Write(6.0f);
                                              Write(uint8_t(10));
                                              Write(uint16_t(300));
                                              Write(0.5f);
                                            });

  const PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 2);
  // Out-of-range integer channels clamp to 255; 0-1 float channels scale.
  EXPECT_EQ(mesh.vertices[0].r, 200);
  EXPECT_EQ(mesh.vertices[0].g, 100);
  EXPECT_EQ(mesh.vertices[0].b, 255);
  EXPECT_EQ(mesh.vertices[1].r, 10);
  EXPECT_EQ(mesh.vertices[1].g, 255);
  EXPECT_EQ(mesh.vertices[1].b, 127);
}

TEST(Ply, ReadTextPlyMeshWithFloatColors) {
  const auto test_file =
      WriteTextPlyMesh("mesh_float_colors.ply",
                       {"float x",
                        "float y",
                        "float z",
                        "float red",
                        "float green",
                        "float blue"},
                       {"0 0 0 1.0 0.5 0.0", "1 0 0 0.0 1.0 0.25"});

  const PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 2);
  EXPECT_EQ(mesh.vertices[0].r, 255);
  EXPECT_EQ(mesh.vertices[0].g, 127);
  EXPECT_EQ(mesh.vertices[0].b, 0);
  EXPECT_EQ(mesh.vertices[1].r, 0);
  EXPECT_EQ(mesh.vertices[1].g, 255);
  EXPECT_EQ(mesh.vertices[1].b, 63);
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

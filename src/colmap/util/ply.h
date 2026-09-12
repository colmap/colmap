// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace colmap {

struct PlyPoint {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float nx = 0.0f;
  float ny = 0.0f;
  float nz = 0.0f;
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
};

struct PlyMeshVertex {
  PlyMeshVertex() : x(0), y(0), z(0) {}
  PlyMeshVertex(const float x, const float y, const float z)
      : x(x), y(y), z(z) {}
  PlyMeshVertex(const float x,
                const float y,
                const float z,
                const uint8_t r,
                const uint8_t g,
                const uint8_t b)
      : x(x), y(y), z(z), r(r), g(g), b(b) {}

  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;

  // Default color is gray.
  uint8_t r = 200;
  uint8_t g = 200;
  uint8_t b = 200;
};

struct PlyMeshFace {
  PlyMeshFace() : vertex_idx1(0), vertex_idx2(0), vertex_idx3(0) {}
  PlyMeshFace(const size_t vertex_idx1,
              const size_t vertex_idx2,
              const size_t vertex_idx3)
      : vertex_idx1(vertex_idx1),
        vertex_idx2(vertex_idx2),
        vertex_idx3(vertex_idx3) {}

  size_t vertex_idx1 = 0;
  size_t vertex_idx2 = 0;
  size_t vertex_idx3 = 0;
};

struct PlyMesh {
  std::vector<PlyMeshVertex> vertices;
  std::vector<PlyMeshFace> faces;
};

struct PlyTexturedMesh {
  PlyMesh mesh;
  // Per-face UV coordinates: 6 floats per face (u1,v1, u2,v2, u3,v3).
  std::vector<float> face_uvs;
  // Texture image filename referenced via "comment TextureFile ..." in header.
  std::string texture_file;
};

// Read PLY point cloud from text or binary file.
std::vector<PlyPoint> ReadPly(const std::filesystem::path& path);

// Write PLY point cloud to text or binary file.
void WriteTextPlyPoints(const std::filesystem::path& path,
                        const std::vector<PlyPoint>& points,
                        bool write_normal = true,
                        bool write_rgb = true);
void WriteBinaryPlyPoints(const std::filesystem::path& path,
                          const std::vector<PlyPoint>& points,
                          bool write_normal = true,
                          bool write_rgb = true);

// Read PLY mesh from text or binary file. Supports both plain and textured
// meshes (with per-face UV coordinates and "comment TextureFile" header).
PlyTexturedMesh ReadPlyMesh(const std::filesystem::path& path);

// Write PLY mesh to text or binary file. Writes texture coordinates and
// TextureFile comment when present in the mesh.
void WriteTextPlyMesh(const std::filesystem::path& path,
                      const PlyTexturedMesh& mesh);
void WriteBinaryPlyMesh(const std::filesystem::path& path,
                        const PlyTexturedMesh& mesh);

// Returns true if the PLY file contains face elements (i.e., is a mesh).
bool HasPlyMeshFaces(const std::filesystem::path& path);

}  // namespace colmap

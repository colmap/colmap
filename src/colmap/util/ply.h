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

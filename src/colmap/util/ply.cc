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

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"

#include <cstring>
#include <fstream>

#include <Eigen/Core>

namespace colmap {
namespace {

template <typename T>
T ReadFromBuffer(const char* buffer, size_t offset) {
  T value;
  std::memcpy(&value, buffer + offset, sizeof(T));
  return value;
}

}  // namespace

std::vector<PlyPoint> ReadPly(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);

  std::vector<PlyPoint> points;

  std::string line;

  // The index of the property for ASCII PLY files.
  int X_index = -1;
  int Y_index = -1;
  int Z_index = -1;
  int NX_index = -1;
  int NY_index = -1;
  int NZ_index = -1;
  int R_index = -1;
  int G_index = -1;
  int B_index = -1;

  // The position in number of bytes of the property for binary PLY files.
  int X_byte_pos = -1;
  int Y_byte_pos = -1;
  int Z_byte_pos = -1;
  int NX_byte_pos = -1;
  int NY_byte_pos = -1;
  int NZ_byte_pos = -1;
  int R_byte_pos = -1;
  int G_byte_pos = -1;
  int B_byte_pos = -1;

  // Flag to use double precision in binary PLY files
  bool X_double = false;
  bool Y_double = false;
  bool Z_double = false;
  bool NX_double = false;
  bool NY_double = false;
  bool NZ_double = false;

  bool in_vertex_section = false;
  bool is_binary = false;
  bool is_little_endian = false;
  size_t num_bytes_per_line = 0;
  size_t num_vertices = 0;

  int index = 0;
  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty()) {
      continue;
    }

    if (line == "end_header") {
      break;
    }

    if (line.size() >= 6 && line.substr(0, 6) == "format") {
      if (line == "format ascii 1.0") {
        is_binary = false;
      } else if (line == "format binary_little_endian 1.0") {
        is_binary = true;
        is_little_endian = true;
      } else if (line == "format binary_big_endian 1.0") {
        is_binary = true;
        is_little_endian = false;
      }
    }

    const std::vector<std::string> line_elems = StringSplit(line, " ");

    if (line_elems.size() >= 3 && line_elems[0] == "element") {
      in_vertex_section = false;
      if (line_elems[1] == "vertex") {
        num_vertices = std::stoll(line_elems[2]);
        in_vertex_section = true;
      } else if (std::stoll(line_elems[2]) > 0) {
        LOG(WARNING) << "Only vertex elements supported; ignoring "
                     << line_elems[1];
      }
    }

    if (!in_vertex_section) {
      continue;
    }

    // Show diffuse, ambient, specular colors as regular colors.

    if (line_elems.size() >= 3 && line_elems[0] == "property") {
      THROW_CHECK(line_elems[1] == "float" || line_elems[1] == "float32" ||
                  line_elems[1] == "double" || line_elems[1] == "float64" ||
                  line_elems[1] == "uchar")
          << "PLY import only supports float, double, and uchar data types";

      if (line == "property float x" || line == "property float32 x" ||
          line == "property double x" || line == "property float64 x") {
        X_index = index;
        X_byte_pos = num_bytes_per_line;
        X_double = (line_elems[1] == "double" || line_elems[1] == "float64");
      } else if (line == "property float y" || line == "property float32 y" ||
                 line == "property double y" || line == "property float64 y") {
        Y_index = index;
        Y_byte_pos = num_bytes_per_line;
        Y_double = (line_elems[1] == "double" || line_elems[1] == "float64");
      } else if (line == "property float z" || line == "property float32 z" ||
                 line == "property double z" || line == "property float64 z") {
        Z_index = index;
        Z_byte_pos = num_bytes_per_line;
        Z_double = (line_elems[1] == "double" || line_elems[1] == "float64");
      } else if (line == "property float nx" || line == "property float32 nx" ||
                 line == "property double nx" ||
                 line == "property float64 nx") {
        NX_index = index;
        NX_byte_pos = num_bytes_per_line;
        NX_double = (line_elems[1] == "double" || line_elems[1] == "float64");
      } else if (line == "property float ny" || line == "property float32 ny" ||
                 line == "property double ny" ||
                 line == "property float64 ny") {
        NY_index = index;
        NY_byte_pos = num_bytes_per_line;
        NY_double = (line_elems[1] == "double" || line_elems[1] == "float64");
      } else if (line == "property float nz" || line == "property float32 nz" ||
                 line == "property double nz" ||
                 line == "property float64 nz") {
        NZ_index = index;
        NZ_byte_pos = num_bytes_per_line;
        NZ_double = (line_elems[1] == "double" || line_elems[1] == "float64");
      } else if (line == "property uchar r" || line == "property uchar red" ||
                 line == "property uchar diffuse_red" ||
                 line == "property uchar ambient_red" ||
                 line == "property uchar specular_red") {
        R_index = index;
        R_byte_pos = num_bytes_per_line;
      } else if (line == "property uchar g" || line == "property uchar green" ||
                 line == "property uchar diffuse_green" ||
                 line == "property uchar ambient_green" ||
                 line == "property uchar specular_green") {
        G_index = index;
        G_byte_pos = num_bytes_per_line;
      } else if (line == "property uchar b" || line == "property uchar blue" ||
                 line == "property uchar diffuse_blue" ||
                 line == "property uchar ambient_blue" ||
                 line == "property uchar specular_blue") {
        B_index = index;
        B_byte_pos = num_bytes_per_line;
      }

      index += 1;
      if (line_elems[1] == "float" || line_elems[1] == "float32") {
        num_bytes_per_line += 4;
      } else if (line_elems[1] == "double" || line_elems[1] == "float64") {
        num_bytes_per_line += 8;
      } else if (line_elems[1] == "uchar") {
        num_bytes_per_line += 1;
      } else {
        LOG(FATAL_THROW) << "Invalid data type: " << line_elems[1];
      }
    }
  }

  const bool is_normal_missing =
      (NX_index == -1) || (NY_index == -1) || (NZ_index == -1);
  const bool is_rgb_missing =
      (R_index == -1) || (G_index == -1) || (B_index == -1);

  THROW_CHECK(X_index != -1 && Y_index != -1 && Z_index != -1)
      << "Invalid PLY file format: x, y, z properties missing";

  points.reserve(num_vertices);

  if (is_binary) {
    auto ReadCoord = [&](const char* buf, bool is_double, size_t byte_pos) {
      if (is_double) {
        const auto val = ReadFromBuffer<double>(buf, byte_pos);
        return static_cast<float>(is_little_endian ? LittleEndianToNative(val)
                                                   : BigEndianToNative(val));
      } else {
        const auto val = ReadFromBuffer<float>(buf, byte_pos);
        return is_little_endian ? LittleEndianToNative(val)
                                : BigEndianToNative(val);
      }
    };

    std::vector<char> buffer(num_bytes_per_line);
    for (size_t i = 0; i < num_vertices; ++i) {
      file.read(buffer.data(), num_bytes_per_line);
      THROW_CHECK(file.good()) << "Unexpected end of PLY file at vertex " << i;

      PlyPoint point;

      point.x = ReadCoord(buffer.data(), X_double, X_byte_pos);
      point.y = ReadCoord(buffer.data(), Y_double, Y_byte_pos);
      point.z = ReadCoord(buffer.data(), Z_double, Z_byte_pos);

      if (!is_normal_missing) {
        point.nx = ReadCoord(buffer.data(), NX_double, NX_byte_pos);
        point.ny = ReadCoord(buffer.data(), NY_double, NY_byte_pos);
        point.nz = ReadCoord(buffer.data(), NZ_double, NZ_byte_pos);
      }

      if (!is_rgb_missing) {
        point.r = ReadFromBuffer<uint8_t>(buffer.data(), R_byte_pos);
        point.g = ReadFromBuffer<uint8_t>(buffer.data(), G_byte_pos);
        point.b = ReadFromBuffer<uint8_t>(buffer.data(), B_byte_pos);
      }

      points.push_back(point);
    }
  } else {
    while (std::getline(file, line)) {
      StringTrim(&line);
      std::stringstream line_stream(line);

      std::string item;
      std::vector<std::string> items;
      while (line_stream >> item) {
        items.push_back(item);
      }

      PlyPoint point;

      point.x = std::stold(items.at(X_index));
      point.y = std::stold(items.at(Y_index));
      point.z = std::stold(items.at(Z_index));

      if (!is_normal_missing) {
        point.nx = std::stold(items.at(NX_index));
        point.ny = std::stold(items.at(NY_index));
        point.nz = std::stold(items.at(NZ_index));
      }

      if (!is_rgb_missing) {
        point.r = std::stoi(items.at(R_index));
        point.g = std::stoi(items.at(G_index));
        point.b = std::stoi(items.at(B_index));
      }

      points.push_back(point);
    }
  }

  return points;
}

void WriteTextPlyPoints(const std::filesystem::path& path,
                        const std::vector<PlyPoint>& points,
                        const bool write_normal,
                        const bool write_rgb) {
  std::ofstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "ply\n";
  file << "format ascii 1.0\n";
  file << "element vertex " << points.size() << '\n';

  file << "property float x\n";
  file << "property float y\n";
  file << "property float z\n";

  if (write_normal) {
    file << "property float nx\n";
    file << "property float ny\n";
    file << "property float nz\n";
  }

  if (write_rgb) {
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
  }

  file << "end_header\n";

  for (const auto& point : points) {
    file << point.x << " " << point.y << " " << point.z;

    if (write_normal) {
      file << " " << point.nx << " " << point.ny << " " << point.nz;
    }

    if (write_rgb) {
      file << " " << static_cast<int>(point.r) << " "
           << static_cast<int>(point.g) << " " << static_cast<int>(point.b);
    }

    file << '\n';
  }

  file.close();
}

void WriteBinaryPlyPoints(const std::filesystem::path& path,
                          const std::vector<PlyPoint>& points,
                          const bool write_normal,
                          const bool write_rgb) {
  std::fstream text_file(path, std::ios::out);
  THROW_CHECK_FILE_OPEN(text_file, path);

  text_file << "ply\n";
  text_file << "format binary_little_endian 1.0\n";
  text_file << "element vertex " << points.size() << '\n';

  text_file << "property float x\n";
  text_file << "property float y\n";
  text_file << "property float z\n";

  if (write_normal) {
    text_file << "property float nx\n";
    text_file << "property float ny\n";
    text_file << "property float nz\n";
  }

  if (write_rgb) {
    text_file << "property uchar red\n";
    text_file << "property uchar green\n";
    text_file << "property uchar blue\n";
  }

  text_file << "end_header\n";
  text_file.close();

  std::fstream binary_file(path,
                           std::ios::out | std::ios::binary | std::ios::app);
  THROW_CHECK_FILE_OPEN(binary_file, path);

  for (const auto& point : points) {
    WriteBinaryLittleEndian<float>(&binary_file, point.x);
    WriteBinaryLittleEndian<float>(&binary_file, point.y);
    WriteBinaryLittleEndian<float>(&binary_file, point.z);

    if (write_normal) {
      WriteBinaryLittleEndian<float>(&binary_file, point.nx);
      WriteBinaryLittleEndian<float>(&binary_file, point.ny);
      WriteBinaryLittleEndian<float>(&binary_file, point.nz);
    }

    if (write_rgb) {
      WriteBinaryLittleEndian<uint8_t>(&binary_file, point.r);
      WriteBinaryLittleEndian<uint8_t>(&binary_file, point.g);
      WriteBinaryLittleEndian<uint8_t>(&binary_file, point.b);
    }
  }

  binary_file.close();
}

PlyMesh ReadPlyMesh(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);

  PlyMesh mesh;

  std::string line;

  bool is_binary = false;
  bool is_little_endian = false;
  size_t num_vertices = 0;
  size_t num_faces = 0;

  // Track vertex properties for proper parsing.
  bool in_vertex_section = false;
  bool in_face_section = false;
  std::string face_count_type = "uchar";
  std::string face_index_type = "int";
  size_t num_bytes_per_vertex = 0;
  int num_vertex_props = 0;

  // Vertex property indices (for ASCII) and byte positions (for binary).
  int X_index = -1, Y_index = -1, Z_index = -1;
  int R_index = -1, G_index = -1, B_index = -1;
  int X_byte_pos = -1, Y_byte_pos = -1, Z_byte_pos = -1;
  int R_byte_pos = -1, G_byte_pos = -1, B_byte_pos = -1;
  bool X_double = false, Y_double = false, Z_double = false;

  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty()) {
      continue;
    }

    if (line == "end_header") {
      break;
    }

    if (line.size() >= 6 && line.substr(0, 6) == "format") {
      if (line == "format ascii 1.0") {
        is_binary = false;
      } else if (line == "format binary_little_endian 1.0") {
        is_binary = true;
        is_little_endian = true;
      } else if (line == "format binary_big_endian 1.0") {
        is_binary = true;
        is_little_endian = false;
      }
    }

    const std::vector<std::string> line_elems = StringSplit(line, " ");

    if (line_elems.size() >= 3 && line_elems[0] == "element") {
      in_vertex_section = false;
      in_face_section = false;
      if (line_elems[1] == "vertex") {
        num_vertices = std::stoll(line_elems[2]);
        in_vertex_section = true;
      } else if (line_elems[1] == "face") {
        num_faces = std::stoll(line_elems[2]);
        in_face_section = true;
      }
    }

    // Parse face property list to determine the index data type.
    // Format: property list <count_type> <index_type> vertex_index
    if (in_face_section && line_elems.size() >= 5 &&
        line_elems[0] == "property" && line_elems[1] == "list") {
      face_count_type = line_elems[2];
      face_index_type = line_elems[3];
    }

    if (in_vertex_section && line_elems.size() >= 3 &&
        line_elems[0] == "property") {
      const std::string& dtype = line_elems[1];
      const std::string& pname = line_elems[2];

      if (pname == "x") {
        X_index = num_vertex_props;
        X_byte_pos = num_bytes_per_vertex;
        X_double = (dtype == "double" || dtype == "float64");
      } else if (pname == "y") {
        Y_index = num_vertex_props;
        Y_byte_pos = num_bytes_per_vertex;
        Y_double = (dtype == "double" || dtype == "float64");
      } else if (pname == "z") {
        Z_index = num_vertex_props;
        Z_byte_pos = num_bytes_per_vertex;
        Z_double = (dtype == "double" || dtype == "float64");
      } else if (pname == "red" || pname == "r" || pname == "diffuse_red") {
        R_index = num_vertex_props;
        R_byte_pos = num_bytes_per_vertex;
      } else if (pname == "green" || pname == "g" || pname == "diffuse_green") {
        G_index = num_vertex_props;
        G_byte_pos = num_bytes_per_vertex;
      } else if (pname == "blue" || pname == "b" || pname == "diffuse_blue") {
        B_index = num_vertex_props;
        B_byte_pos = num_bytes_per_vertex;
      }

      num_vertex_props += 1;
      if (dtype == "float" || dtype == "float32" || dtype == "int" ||
          dtype == "int32" || dtype == "uint" || dtype == "uint32") {
        num_bytes_per_vertex += 4;
      } else if (dtype == "double" || dtype == "float64") {
        num_bytes_per_vertex += 8;
      } else if (dtype == "short" || dtype == "int16" || dtype == "ushort" ||
                 dtype == "uint16") {
        num_bytes_per_vertex += 2;
      } else if (dtype == "uchar" || dtype == "uint8" || dtype == "char" ||
                 dtype == "int8") {
        num_bytes_per_vertex += 1;
      } else {
        LOG(FATAL_THROW) << "Invalid vertex data type: " << dtype;
      }
    }
  }

  THROW_CHECK(X_index != -1 && Y_index != -1 && Z_index != -1)
      << "Invalid PLY mesh format: x, y, z properties missing";

  const bool has_colors = (R_index != -1) && (G_index != -1) && (B_index != -1);

  mesh.vertices.reserve(num_vertices);
  mesh.faces.reserve(num_faces);

  if (is_binary) {
    auto ReadCoord = [&](const char* buf, bool is_double, size_t byte_pos) {
      if (is_double) {
        const auto val = ReadFromBuffer<double>(buf, byte_pos);
        return static_cast<float>(is_little_endian ? LittleEndianToNative(val)
                                                   : BigEndianToNative(val));
      } else {
        const auto val = ReadFromBuffer<float>(buf, byte_pos);
        return is_little_endian ? LittleEndianToNative(val)
                                : BigEndianToNative(val);
      }
    };

    auto ReadInt = [&](const char* buf, size_t num_bytes) {
      if (num_bytes == 4) {
        const auto val = ReadFromBuffer<int32_t>(buf, 0);
        return static_cast<int>(is_little_endian ? LittleEndianToNative(val)
                                                 : BigEndianToNative(val));
      } else if (num_bytes == 2) {
        const auto val = ReadFromBuffer<int16_t>(buf, 0);
        return static_cast<int>(is_little_endian ? LittleEndianToNative(val)
                                                 : BigEndianToNative(val));
      } else {
        return static_cast<int>(ReadFromBuffer<uint8_t>(buf, 0));
      }
    };

    // Read binary vertex data
    std::vector<char> buffer(num_bytes_per_vertex);
    for (size_t i = 0; i < num_vertices; ++i) {
      file.read(buffer.data(), num_bytes_per_vertex);
      THROW_CHECK(file.good()) << "Unexpected end of PLY file at vertex " << i;

      const float x = ReadCoord(buffer.data(), X_double, X_byte_pos);
      const float y = ReadCoord(buffer.data(), Y_double, Y_byte_pos);
      const float z = ReadCoord(buffer.data(), Z_double, Z_byte_pos);

      if (has_colors) {
        const uint8_t r = ReadFromBuffer<uint8_t>(buffer.data(), R_byte_pos);
        const uint8_t g = ReadFromBuffer<uint8_t>(buffer.data(), G_byte_pos);
        const uint8_t b = ReadFromBuffer<uint8_t>(buffer.data(), B_byte_pos);
        mesh.vertices.emplace_back(x, y, z, r, g, b);
      } else {
        mesh.vertices.emplace_back(x, y, z);
      }
    }

    // Read binary face data
    // Determine byte sizes of the face count and index types from the header.
    auto PlyTypeBytes = [](const std::string& type) -> size_t {
      if (type == "int" || type == "int32" || type == "uint" ||
          type == "uint32") {
        return 4;
      } else if (type == "short" || type == "int16" || type == "ushort" ||
                 type == "uint16") {
        return 2;
      } else if (type == "char" || type == "int8" || type == "uchar" ||
                 type == "uint8") {
        return 1;
      }
      LOG(FATAL_THROW) << "Unsupported PLY face data type: " << type;
      return 0;
    };

    const size_t face_count_bytes = PlyTypeBytes(face_count_type);
    const size_t face_index_bytes = PlyTypeBytes(face_index_type);

    std::vector<char> face_buffer(std::max(face_count_bytes, face_index_bytes));
    for (size_t i = 0; i < num_faces; ++i) {
      file.read(face_buffer.data(), face_count_bytes);
      THROW_CHECK(file.good()) << "Unexpected end of PLY file at face " << i;

      const int num_face_vertices =
          ReadInt(face_buffer.data(), face_count_bytes);
      THROW_CHECK_EQ(num_face_vertices, 3)
          << "Only triangular faces are supported";

      int indices[3];
      for (int j = 0; j < 3; ++j) {
        file.read(face_buffer.data(), face_index_bytes);
        THROW_CHECK(file.good())
            << "Unexpected end of PLY file at face " << i << " index " << j;
        indices[j] = ReadInt(face_buffer.data(), face_index_bytes);
      }

      mesh.faces.emplace_back(indices[0], indices[1], indices[2]);
    }
  } else {
    // Read ASCII vertex data
    for (size_t i = 0; i < num_vertices; ++i) {
      std::getline(file, line);
      StringTrim(&line);
      std::stringstream line_stream(line);

      std::string item;
      std::vector<std::string> items;
      while (line_stream >> item) {
        items.push_back(item);
      }

      const float x = std::stof(items.at(X_index));
      const float y = std::stof(items.at(Y_index));
      const float z = std::stof(items.at(Z_index));

      if (has_colors) {
        const uint8_t r = static_cast<uint8_t>(std::stoi(items.at(R_index)));
        const uint8_t g = static_cast<uint8_t>(std::stoi(items.at(G_index)));
        const uint8_t b = static_cast<uint8_t>(std::stoi(items.at(B_index)));
        mesh.vertices.emplace_back(x, y, z, r, g, b);
      } else {
        mesh.vertices.emplace_back(x, y, z);
      }
    }

    // Read ASCII face data
    for (size_t i = 0; i < num_faces; ++i) {
      std::getline(file, line);
      StringTrim(&line);
      std::stringstream line_stream(line);

      int num_face_vertices;
      line_stream >> num_face_vertices;
      THROW_CHECK_EQ(num_face_vertices, 3)
          << "Only triangular faces are supported";

      int idx1, idx2, idx3;
      line_stream >> idx1 >> idx2 >> idx3;
      mesh.faces.emplace_back(idx1, idx2, idx3);
    }
  }

  return mesh;
}

void WriteTextPlyMesh(const std::filesystem::path& path, const PlyMesh& mesh) {
  std::fstream file(path, std::ios::out);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "ply\n";
  file << "format ascii 1.0\n";
  file << "element vertex " << mesh.vertices.size() << '\n';
  file << "property float x\n";
  file << "property float y\n";
  file << "property float z\n";
  file << "element face " << mesh.faces.size() << '\n';
  file << "property list uchar int vertex_index\n";
  file << "end_header\n";

  for (const auto& vertex : mesh.vertices) {
    file << vertex.x << " " << vertex.y << " " << vertex.z << '\n';
  }

  for (const auto& face : mesh.faces) {
    file << "3 " << face.vertex_idx1 << " " << face.vertex_idx2 << " "
         << face.vertex_idx3 << '\n';
  }
}

void WriteBinaryPlyMesh(const std::filesystem::path& path,
                        const PlyMesh& mesh) {
  std::fstream text_file(path, std::ios::out);
  THROW_CHECK_FILE_OPEN(text_file, path);

  text_file << "ply\n";
  text_file << "format binary_little_endian 1.0\n";
  text_file << "element vertex " << mesh.vertices.size() << '\n';
  text_file << "property float x\n";
  text_file << "property float y\n";
  text_file << "property float z\n";
  text_file << "element face " << mesh.faces.size() << '\n';
  text_file << "property list uchar int vertex_index\n";
  text_file << "end_header\n";
  text_file.close();

  std::fstream binary_file(path,
                           std::ios::out | std::ios::binary | std::ios::app);
  THROW_CHECK_FILE_OPEN(binary_file, path);

  for (const auto& vertex : mesh.vertices) {
    WriteBinaryLittleEndian<float>(&binary_file, vertex.x);
    WriteBinaryLittleEndian<float>(&binary_file, vertex.y);
    WriteBinaryLittleEndian<float>(&binary_file, vertex.z);
  }

  for (const auto& face : mesh.faces) {
    THROW_CHECK_LT(face.vertex_idx1, mesh.vertices.size());
    THROW_CHECK_LT(face.vertex_idx2, mesh.vertices.size());
    THROW_CHECK_LT(face.vertex_idx3, mesh.vertices.size());
    const uint8_t kNumVertices = 3;
    WriteBinaryLittleEndian<uint8_t>(&binary_file, kNumVertices);
    WriteBinaryLittleEndian<int>(&binary_file, face.vertex_idx1);
    WriteBinaryLittleEndian<int>(&binary_file, face.vertex_idx2);
    WriteBinaryLittleEndian<int>(&binary_file, face.vertex_idx3);
  }

  binary_file.close();
}

bool HasPlyMeshFaces(const std::filesystem::path& path) {
  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);

  std::string line;
  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line == "end_header") {
      break;
    }

    const std::vector<std::string> line_elems = StringSplit(line, " ");
    if (line_elems.size() >= 3 && line_elems[0] == "element" &&
        line_elems[1] == "face" && std::stoll(line_elems[2]) > 0) {
      return true;
    }
  }

  return false;
}

}  // namespace colmap

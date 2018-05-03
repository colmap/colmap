// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "util/ply.h"

#include <fstream>

#include <Eigen/Core>

#include "util/logging.h"
#include "util/misc.h"

namespace colmap {

std::vector<PlyPoint> ReadPly(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.is_open()) << path;

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
        LOG(FATAL) << "Only vertex elements supported";
      }
    }

    if (!in_vertex_section) {
      continue;
    }

    // Show diffuse, ambient, specular colors as regular colors.

    if (line_elems.size() >= 3 && line_elems[0] == "property") {
      CHECK(line_elems[1] == "float" || line_elems[1] == "float32" ||
            line_elems[1] == "uchar")
          << "PLY import only supports the float and uchar data types";

      if (line == "property float x" || line == "property float32 x") {
        X_index = index;
        X_byte_pos = num_bytes_per_line;
      } else if (line == "property float y" || line == "property float32 y") {
        Y_index = index;
        Y_byte_pos = num_bytes_per_line;
      } else if (line == "property float z" || line == "property float32 z") {
        Z_index = index;
        Z_byte_pos = num_bytes_per_line;
      } else if (line == "property float nx" || line == "property float32 nx") {
        NX_index = index;
        NX_byte_pos = num_bytes_per_line;
      } else if (line == "property float ny" || line == "property float32 ny") {
        NY_index = index;
        NY_byte_pos = num_bytes_per_line;
      } else if (line == "property float nz" || line == "property float32 nz") {
        NZ_index = index;
        NZ_byte_pos = num_bytes_per_line;
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
      } else if (line_elems[1] == "uchar") {
        num_bytes_per_line += 1;
      } else {
        LOG(FATAL) << "Invalid data type: " << line_elems[1];
      }
    }
  }

  const bool is_normal_missing =
      (NX_index == -1) || (NY_index == -1) || (NZ_index == -1);
  const bool is_rgb_missing =
      (R_index == -1) || (G_index == -1) || (B_index == -1);

  CHECK(X_index != -1 && Y_index != -1 && Z_index)
      << "Invalid PLY file format: x, y, z properties missing";

  points.reserve(num_vertices);

  if (is_binary) {
    std::vector<char> buffer(num_bytes_per_line);
    for (size_t i = 0; i < num_vertices; ++i) {
      file.read(buffer.data(), num_bytes_per_line);

      PlyPoint point;

      if (is_little_endian) {
        point.x = LittleEndianToNative(
            *reinterpret_cast<float*>(&buffer[X_byte_pos]));
        point.y = LittleEndianToNative(
            *reinterpret_cast<float*>(&buffer[Y_byte_pos]));
        point.z = LittleEndianToNative(
            *reinterpret_cast<float*>(&buffer[Z_byte_pos]));

        if (!is_normal_missing) {
          point.nx = LittleEndianToNative(
              *reinterpret_cast<float*>(&buffer[NX_byte_pos]));
          point.ny = LittleEndianToNative(
              *reinterpret_cast<float*>(&buffer[NY_byte_pos]));
          point.nz = LittleEndianToNative(
              *reinterpret_cast<float*>(&buffer[NZ_byte_pos]));
        }

        if (!is_rgb_missing) {
          point.r = LittleEndianToNative(
              *reinterpret_cast<uint8_t*>(&buffer[R_byte_pos]));
          point.g = LittleEndianToNative(
              *reinterpret_cast<uint8_t*>(&buffer[G_byte_pos]));
          point.b = LittleEndianToNative(
              *reinterpret_cast<uint8_t*>(&buffer[B_byte_pos]));
        }
      } else {
        point.x =
            BigEndianToNative(*reinterpret_cast<float*>(&buffer[X_byte_pos]));
        point.y =
            BigEndianToNative(*reinterpret_cast<float*>(&buffer[Y_byte_pos]));
        point.z =
            BigEndianToNative(*reinterpret_cast<float*>(&buffer[Z_byte_pos]));

        if (!is_normal_missing) {
          point.nx = BigEndianToNative(
              *reinterpret_cast<float*>(&buffer[NX_byte_pos]));
          point.ny = BigEndianToNative(
              *reinterpret_cast<float*>(&buffer[NY_byte_pos]));
          point.nz = BigEndianToNative(
              *reinterpret_cast<float*>(&buffer[NZ_byte_pos]));
        }

        if (!is_rgb_missing) {
          point.r = BigEndianToNative(
              *reinterpret_cast<uint8_t*>(&buffer[R_byte_pos]));
          point.g = BigEndianToNative(
              *reinterpret_cast<uint8_t*>(&buffer[G_byte_pos]));
          point.b = BigEndianToNative(
              *reinterpret_cast<uint8_t*>(&buffer[B_byte_pos]));
        }
      }

      points.push_back(point);
    }
  } else {
    while (std::getline(file, line)) {
      StringTrim(&line);
      std::stringstream line_stream(line);

      std::string item;
      std::vector<std::string> items;
      while (!line_stream.eof()) {
        std::getline(line_stream, item, ' ');
        StringTrim(&item);
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

void WriteTextPly(const std::string& path, const std::vector<PlyPoint>& points,
                  const bool write_normal, const bool write_rgb) {
  std::ofstream file(path);
  CHECK(file.is_open()) << path;

  file << "ply" << std::endl;
  file << "format ascii 1.0" << std::endl;
  file << "element vertex " << points.size() << std::endl;

  file << "property float x" << std::endl;
  file << "property float y" << std::endl;
  file << "property float z" << std::endl;

  if (write_normal) {
    file << "property float nx" << std::endl;
    file << "property float ny" << std::endl;
    file << "property float nz" << std::endl;
  }

  if (write_rgb) {
    file << "property uchar red" << std::endl;
    file << "property uchar green" << std::endl;
    file << "property uchar blue" << std::endl;
  }

  file << "end_header" << std::endl;

  for (const auto& point : points) {
    file << point.x << " " << point.y << " " << point.z;

    if (write_normal) {
      file << " " << point.nx << " " << point.ny << " " << point.nz;
    }

    if (write_rgb) {
      file << " " << static_cast<int>(point.r) << " "
           << static_cast<int>(point.g) << " " << static_cast<int>(point.b);
    }

    file << std::endl;
  }

  file.close();
}

void WriteBinaryPly(const std::string& path,
                    const std::vector<PlyPoint>& points,
                    const bool write_normal, const bool write_rgb) {
  std::fstream text_file(path, std::ios::out);
  CHECK(text_file.is_open()) << path;

  text_file << "ply" << std::endl;
  text_file << "format binary_little_endian 1.0" << std::endl;
  text_file << "element vertex " << points.size() << std::endl;

  text_file << "property float x" << std::endl;
  text_file << "property float y" << std::endl;
  text_file << "property float z" << std::endl;

  if (write_normal) {
    text_file << "property float nx" << std::endl;
    text_file << "property float ny" << std::endl;
    text_file << "property float nz" << std::endl;
  }

  if (write_rgb) {
    text_file << "property uchar red" << std::endl;
    text_file << "property uchar green" << std::endl;
    text_file << "property uchar blue" << std::endl;
  }

  text_file << "end_header" << std::endl;
  text_file.close();

  std::fstream binary_file(path,
                           std::ios::out | std::ios::binary | std::ios::app);
  CHECK(binary_file.is_open()) << path;

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

}  // namespace colmap

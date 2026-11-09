// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mat.h"

#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/string.h"

#include <cstdint>
#include <fstream>
#include <vector>

namespace colmap {
namespace mvs {

template <>
void Mat<float>::Read(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  SetFullPrecTextStream(file);

  size_t width, height, depth;
  char sep1, sep2, sep3;
  THROW_CHECK(file >> width >> sep1 >> height >> sep2 >> depth >> sep3)
      << "Invalid matrix header in " << path;
  THROW_CHECK(sep1 == '&' && sep2 == '&' && sep3 == '&') << path;
  THROW_CHECK_GT(width, 0) << path;
  THROW_CHECK_GT(height, 0) << path;
  THROW_CHECK_GT(depth, 0) << path;

  const auto data_start = file.tellg();
  file.seekg(0, std::ios::end);
  const std::streamoff num_bytes = file.tellg() - data_start;
  THROW_CHECK_GE(num_bytes, 0) << path;
  const auto available = static_cast<std::uintmax_t>(num_bytes) / sizeof(float);
  THROW_CHECK_LE(width, available / height / depth)
      << "Matrix dimensions exceed the data in " << path;
  file.seekg(data_start);
  data_.resize(width * height * depth);
  ReadBinaryLittleEndian<float>(&file, &data_);
  THROW_CHECK(file) << path;
  width_ = width;
  height_ = height;
  depth_ = depth;
  file.close();
}

template <>
void Mat<float>::Write(const std::filesystem::path& path) const {
  std::ofstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  SetFullPrecTextStream(file);
  file << width_ << "&" << height_ << "&" << depth_ << "&";
  WriteBinaryLittleEndian<float>(&file, {data_.data(), data_.size()});
  file.close();
}

}  // namespace mvs
}  // namespace colmap

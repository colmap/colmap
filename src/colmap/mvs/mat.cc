// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mat.h"

#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/string.h"

#include <fstream>
#include <vector>

namespace colmap {
namespace mvs {

template <>
void Mat<float>::Read(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  SetFullPrecTextStream(file);

  char unused_char;
  file >> width_ >> unused_char >> height_ >> unused_char >> depth_ >>
      unused_char;
  THROW_CHECK(file) << path;
  THROW_CHECK_GT(width_, 0) << path;
  THROW_CHECK_GT(height_, 0) << path;
  THROW_CHECK_GT(depth_, 0) << path;
  THROW_CHECK_LE(width_, data_.max_size() / height_ / depth_) << path;
  data_.resize(width_ * height_ * depth_);

  try {
    file.exceptions(std::ios::failbit | std::ios::badbit);
    ReadBinaryLittleEndian<float>(&file, &data_);
    file.close();
  } catch (const std::ios_base::failure& error) {
    throw std::ios_base::failure(path.string() + ": " + error.what(),
                                 error.code());
  }
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

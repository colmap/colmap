// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/consistency_graph.h"

#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"

#include <fstream>

namespace colmap {
namespace mvs {

const int ConsistencyGraph::kNoConsistentImageIds = -1;

ConsistencyGraph::ConsistencyGraph() {}

ConsistencyGraph::ConsistencyGraph(const size_t width,
                                   const size_t height,
                                   const std::vector<int>& data)
    : data_(data) {
  InitializeMap(width, height);
}

size_t ConsistencyGraph::GetNumBytes() const {
  return (data_.size() + map_.size()) * sizeof(int);
}

void ConsistencyGraph::GetImageIdxs(const int row,
                                    const int col,
                                    int* num_images,
                                    const int** image_idxs) const {
  const int index = map_(row, col);
  if (index == kNoConsistentImageIds) {
    *num_images = 0;
    *image_idxs = nullptr;
  } else {
    *num_images = data_.at(index);
    *image_idxs = &data_.at(index + 1);
  }
}

void ConsistencyGraph::Read(const std::filesystem::path& path) {
  std::fstream text_file(path, std::ios::in | std::ios::binary);
  THROW_CHECK_FILE_OPEN(text_file, path);

  size_t width = 0;
  size_t height = 0;
  size_t depth = 0;
  char unused_char;

  text_file >> width >> unused_char >> height >> unused_char >> depth >>
      unused_char;
  const std::streampos pos = text_file.tellg();
  text_file.close();

  THROW_CHECK_GT(width, 0);
  THROW_CHECK_GT(height, 0);
  THROW_CHECK_GT(depth, 0);

  std::fstream binary_file(path, std::ios::in | std::ios::binary);
  THROW_CHECK_FILE_OPEN(binary_file, path);

  binary_file.seekg(0, std::ios::end);
  const size_t num_bytes = binary_file.tellg() - pos;

  data_.resize(num_bytes / sizeof(int));

  binary_file.seekg(pos);
  ReadBinaryLittleEndian<int>(&binary_file, &data_);
  binary_file.close();

  InitializeMap(width, height);
}

void ConsistencyGraph::Write(const std::filesystem::path& path) const {
  std::fstream text_file(path, std::ios::out);
  THROW_CHECK_FILE_OPEN(text_file, path);
  text_file << map_.cols() << "&" << map_.rows() << "&" << 1 << "&";
  text_file.close();

  std::fstream binary_file(path,
                           std::ios::out | std::ios::binary | std::ios::app);
  THROW_CHECK_FILE_OPEN(binary_file, path);
  WriteBinaryLittleEndian<int>(&binary_file, {data_.data(), data_.size()});
  binary_file.close();
}

void ConsistencyGraph::InitializeMap(const size_t width, const size_t height) {
  map_.resize(height, width);
  map_.setConstant(kNoConsistentImageIds);
  for (size_t i = 0; i < data_.size();) {
    THROW_CHECK_LT(i + 2, data_.size())
        << "Corrupt consistency graph: insufficient data at offset " << i;
    const int col = data_.at(i);
    const int row = data_.at(i + 1);
    const int num_images = data_.at(i + 2);
    THROW_CHECK_GE(num_images, 0)
        << "Corrupt consistency graph: negative num_images at offset " << i;
    THROW_CHECK_GE(col, 0);
    THROW_CHECK_LT(col, static_cast<int>(width));
    THROW_CHECK_GE(row, 0);
    THROW_CHECK_LT(row, static_cast<int>(height));
    if (num_images > 0) {
      map_(row, col) = i + 2;
    }
    i += 3 + num_images;
  }
}

}  // namespace mvs
}  // namespace colmap

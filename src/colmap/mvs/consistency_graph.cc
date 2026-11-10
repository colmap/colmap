// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/consistency_graph.h"

#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/string.h"

#include <fstream>
#include <limits>

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
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  file.exceptions(std::ios::failbit | std::ios::badbit);
  SetFullPrecTextStream(file);

  size_t width = 0;
  size_t height = 0;
  size_t depth = 0;
  char width_separator, height_separator, depth_separator;
  const auto read_positive = [&file](size_t& value, char& separator) {
    file >> std::ws;
    THROW_CHECK_NE(file.peek(), '-');
    file >> value >> separator;
    THROW_CHECK_GT(value, 0);
    THROW_CHECK_EQ(separator, '&');
  };
  read_positive(width, width_separator);
  read_positive(height, height_separator);
  read_positive(depth, depth_separator);
  const std::streampos pos = file.tellg();
  file.seekg(0, std::ios::end);
  const std::streamoff num_bytes = file.tellg() - pos;
  THROW_CHECK_GE(num_bytes, 0);
  THROW_CHECK_EQ(num_bytes % sizeof(int), 0);
  ConsistencyGraph graph;
  graph.data_.resize(num_bytes / sizeof(int));
  file.seekg(pos);
  ReadBinaryLittleEndian<int>(&file, &graph.data_);
  file.close();
  graph.InitializeMap(width, height);
  map_.swap(graph.map_);
  data_.swap(graph.data_);
}

void ConsistencyGraph::Write(const std::filesystem::path& path) const {
  std::fstream text_file(path, std::ios::out);
  THROW_CHECK_FILE_OPEN(text_file, path);
  SetFullPrecTextStream(text_file);
  text_file << map_.cols() << "&" << map_.rows() << "&" << 1 << "&";
  text_file.close();

  std::fstream binary_file(path,
                           std::ios::out | std::ios::binary | std::ios::app);
  THROW_CHECK_FILE_OPEN(binary_file, path);
  WriteBinaryLittleEndian<int>(&binary_file, {data_.data(), data_.size()});
  binary_file.close();
}

void ConsistencyGraph::InitializeMap(const size_t width, const size_t height) {
  THROW_CHECK_LE(width, static_cast<size_t>(std::numeric_limits<int>::max()));
  THROW_CHECK_LE(height, static_cast<size_t>(std::numeric_limits<int>::max()));
  map_.resize(height, width);
  map_.setConstant(kNoConsistentImageIds);
  for (size_t i = 0; i < data_.size();) {
    THROW_CHECK_GE(data_.size() - i, 3)
        << "Corrupt consistency graph: insufficient data at offset " << i;
    const int col = data_.at(i);
    const int row = data_.at(i + 1);
    const int num_images = data_.at(i + 2);
    THROW_CHECK_GE(num_images, 0)
        << "Corrupt consistency graph: negative num_images at offset " << i;
    THROW_CHECK_LE(static_cast<size_t>(num_images), data_.size() - i - 3)
        << "Corrupt consistency graph: insufficient image indices at offset "
        << i;
    THROW_CHECK_GE(col, 0);
    THROW_CHECK_LT(col, static_cast<int>(width));
    THROW_CHECK_GE(row, 0);
    THROW_CHECK_LT(row, static_cast<int>(height));
    if (num_images > 0) {
      THROW_CHECK_LE(i + 2,
                     static_cast<size_t>(std::numeric_limits<int>::max() - 1));
      map_(row, col) = static_cast<int>(i + 2);
    }
    i += 3 + static_cast<size_t>(num_images);
  }
}

}  // namespace mvs
}  // namespace colmap

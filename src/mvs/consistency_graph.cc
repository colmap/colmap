// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "mvs/consistency_graph.h"

#include <fstream>
#include <iostream>

#include "util/logging.h"

namespace colmap {
namespace mvs {

const int ConsistencyGraph::kNoConsistentImageIds = -1;

ConsistencyGraph::ConsistencyGraph() {}

ConsistencyGraph::ConsistencyGraph(const size_t width, const size_t height,
                                   const std::vector<int>& data)
    : data_(data), map_(height, width) {
  map_.setConstant(kNoConsistentImageIds);
  for (size_t i = 0; i < data_.size();) {
    const int col = data_[i++];
    const int row = data_[i++];
    map_(row, col) = i;
    const int num_images = data_[i++];
    i += num_images;
  }
}

void ConsistencyGraph::GetImageIds(const int row, const int col,
                                   int* num_images,
                                   const int** image_ids) const {
  const int kNoConsistentImageIds = -1;
  const int index = map_(row, col);
  if (index == kNoConsistentImageIds) {
    *num_images = 0;
    *image_ids = nullptr;
  } else {
    *num_images = data_.at(index);
    *image_ids = &data_.at(index + 1);
  }
}

void ConsistencyGraph::Read(const std::string& path) {
  std::fstream text_file(path, std::ios_base::in | std::ios_base::binary);
  CHECK(text_file.is_open()) << path;

  size_t width;
  size_t height;
  size_t depth;
  char unused_char;

  text_file >> width >> unused_char >> height >> unused_char >> depth >>
      unused_char;
  std::streampos pos = text_file.tellg();
  text_file.close();

  CHECK_GT(width, 0);
  CHECK_GT(height, 0);
  CHECK_GT(depth, 0);

  std::fstream binary_file(path, std::ios_base::in | std::ios_base::binary);
  CHECK(binary_file.is_open()) << path;

  binary_file.seekg(0, std::ios::end);
  const size_t num_bytes = binary_file.tellg() - pos;

  std::vector<int> data(num_bytes / sizeof(int));

  binary_file.seekg(pos);
  binary_file.read(reinterpret_cast<char*>(data.data()), num_bytes);
  binary_file.close();

  *this = ConsistencyGraph(width, height, data);
}

void ConsistencyGraph::Write(const std::string& path) const {
  std::fstream text_file(path, std::ios_base::out);
  CHECK(text_file.is_open()) << path;
  text_file << map_.cols() << "&" << map_.rows() << "&" << 1 << "&";
  text_file.close();

  std::fstream binary_file(
      path, std::ios_base::out | std::ios_base::binary | std::ios_base::app);
  CHECK(binary_file.is_open()) << path;
  binary_file.write(reinterpret_cast<const char*>(data_.data()),
                    sizeof(int) * data_.size());
  binary_file.close();
}

}  // namespace mvs
}  // namespace colmap

// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/mvs/consistency_graph.h"

#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <fstream>
#include <numeric>

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

void ConsistencyGraph::Read(const std::string& path) {
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

void ConsistencyGraph::Write(const std::string& path) const {
  std::fstream text_file(path, std::ios::out);
  THROW_CHECK_FILE_OPEN(text_file, path);
  text_file << map_.cols() << "&" << map_.rows() << "&" << 1 << "&";
  text_file.close();

  std::fstream binary_file(path,
                           std::ios::out | std::ios::binary | std::ios::app);
  THROW_CHECK_FILE_OPEN(binary_file, path);
  WriteBinaryLittleEndian<int>(&binary_file, data_);
  binary_file.close();
}

void ConsistencyGraph::InitializeMap(const size_t width, const size_t height) {
  map_.resize(height, width);
  map_.setConstant(kNoConsistentImageIds);
  for (size_t i = 0; i < data_.size();) {
    const int num_images = data_.at(i + 2);
    if (num_images > 0) {
      const int col = data_.at(i);
      const int row = data_.at(i + 1);
      map_(row, col) = i + 2;
    }
    i += 3 + num_images;
  }
}

}  // namespace mvs
}  // namespace colmap

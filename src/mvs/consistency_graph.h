// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_MVS_CONSISTENCY_GRAPH_H_
#define COLMAP_SRC_MVS_CONSISTENCY_GRAPH_H_

#include <string>
#include <vector>

#include <Eigen/Core>

#include "util/types.h"

namespace colmap {
namespace mvs {

// List of geometrically consistent images, in the following format:
//
//    r_1, c_1, N_1, i_11, i_12, ..., i_1N_1,
//    r_2, c_2, N_2, i_21, i_22, ..., i_2N_2, ...
//
// where r, c are the row and column image coordinates of the pixel,
// N is the number of consistent images, followed by the N image indices.
// Note that only pixels are listed which are not filtered and that the
// consistency graph is only filled if filtering is enabled.
class ConsistencyGraph {
 public:
  ConsistencyGraph();
  ConsistencyGraph(const size_t width, const size_t height,
                   const std::vector<int>& data);

  size_t GetNumBytes() const;

  void GetImageIdxs(const int row, const int col, int* num_images,
                    const int** image_idxs) const;

  void Read(const std::string& path);
  void Write(const std::string& path) const;

 private:
  void InitializeMap(const size_t width, const size_t height);

  const static int kNoConsistentImageIds;
  std::vector<int> data_;
  Eigen::MatrixXi map_;
};

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_CONSISTENCY_GRAPH_H_

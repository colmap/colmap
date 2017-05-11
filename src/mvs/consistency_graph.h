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
// N is the number of consistent images, followed by the N image identifiers.
// Note that only pixels are listed which are not filtered and that the
// consistency graph is only filled if filtering is enabled.
class ConsistencyGraph {
 public:
  ConsistencyGraph();
  ConsistencyGraph(const size_t width, const size_t height,
                   const std::vector<int>& data);

  size_t GetNumBytes() const;

  void GetImageIds(const int row, const int col, int* num_images,
                   const int** image_ids) const;

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

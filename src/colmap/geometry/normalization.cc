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

#include "colmap/geometry/normalization.h"

#include "colmap/util/logging.h"

#include <algorithm>

namespace colmap {

std::pair<Eigen::AlignedBox3d, Eigen::Vector3d> ComputeBoundingBoxAndCentroid(
    double min_percentile,
    double max_percentile,
    std::vector<double> coords_x,
    std::vector<double> coords_y,
    std::vector<double> coords_z) {
  THROW_CHECK(!coords_x.empty());
  THROW_CHECK_EQ(coords_x.size(), coords_y.size());
  THROW_CHECK_EQ(coords_x.size(), coords_z.size());
  THROW_CHECK_GE(min_percentile, 0);
  THROW_CHECK_LE(min_percentile, 1);
  THROW_CHECK_GE(max_percentile, 0);
  THROW_CHECK_LE(max_percentile, 1);
  THROW_CHECK_LE(min_percentile, max_percentile);

  const size_t end_idx = coords_x.size() - 1;
  const size_t min_idx = std::min<size_t>(
      end_idx, static_cast<size_t>(std::floor(min_percentile * end_idx)));
  const size_t max_idx = std::min<size_t>(
      end_idx, static_cast<size_t>(std::ceil(max_percentile * end_idx)));

  std::nth_element(
      coords_x.begin(), coords_x.begin() + min_idx, coords_x.end());
  std::nth_element(coords_x.begin() + min_idx + 1,
                   coords_x.begin() + max_idx,
                   coords_x.end());
  std::nth_element(
      coords_y.begin(), coords_y.begin() + min_idx, coords_y.end());
  std::nth_element(coords_y.begin() + min_idx + 1,
                   coords_y.begin() + max_idx,
                   coords_y.end());
  std::nth_element(
      coords_z.begin(), coords_z.begin() + min_idx, coords_z.end());
  std::nth_element(coords_z.begin() + min_idx + 1,
                   coords_z.begin() + max_idx,
                   coords_z.end());

  const Eigen::Vector3d bbox_min(
      coords_x[min_idx], coords_y[min_idx], coords_z[min_idx]);
  const Eigen::Vector3d bbox_max(
      coords_x[max_idx], coords_y[max_idx], coords_z[max_idx]);

  Eigen::Vector3d centroid(0, 0, 0);
  const double normalization = 1.0 / (max_idx - min_idx + 1);
  for (size_t i = min_idx; i <= max_idx; ++i) {
    centroid(0) += normalization * coords_x[i];
    centroid(1) += normalization * coords_y[i];
    centroid(2) += normalization * coords_z[i];
  }

  return std::make_pair(Eigen::AlignedBox3d(bbox_min, bbox_max), centroid);
}

}  // namespace colmap

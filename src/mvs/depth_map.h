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

#ifndef COLMAP_SRC_MVS_DEPTH_MAP_H_
#define COLMAP_SRC_MVS_DEPTH_MAP_H_

#include <string>
#include <vector>

#include "mvs/mat.h"
#include "util/bitmap.h"

namespace colmap {
namespace mvs {

class DepthMap : public Mat<float> {
 public:
  DepthMap();
  DepthMap(const size_t width, const size_t height, const float depth_min,
           const float depth_max);
  DepthMap(const Mat<float>& mat, const float depth_min, const float depth_max);

  inline float GetDepthMin() const;
  inline float GetDepthMax() const;

  inline float Get(const size_t row, const size_t col) const;

  void Rescale(const float factor);
  void Downsize(const size_t max_width, const size_t max_height);

  Bitmap ToBitmap(const float min_percentile, const float max_percentile) const;

 private:
  float depth_min_ = -1.0f;
  float depth_max_ = -1.0f;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

float DepthMap::GetDepthMin() const { return depth_min_; }

float DepthMap::GetDepthMax() const { return depth_max_; }

float DepthMap::Get(const size_t row, const size_t col) const {
  return data_.at(row * width_ + col);
}

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_DEPTH_MAP_H_

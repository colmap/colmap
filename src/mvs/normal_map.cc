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

#include "mvs/normal_map.h"

#include <algorithm>
#include <fstream>

#include "base/warp.h"
#include "util/logging.h"

namespace colmap {
namespace mvs {

NormalMap::NormalMap() : Mat<float>(0, 0, 3) {}

NormalMap::NormalMap(const size_t width, const size_t height)
    : Mat<float>(width, height, 3) {}

NormalMap::NormalMap(const Mat<float>& mat)
    : Mat<float>(mat.GetWidth(), mat.GetHeight(), mat.GetDepth()) {
  CHECK_EQ(mat.GetDepth(), 3);
  data_ = mat.GetData();
}

void NormalMap::Rescale(const float factor) {
  const size_t new_width = std::round(width_ * factor);
  const size_t new_height = std::round(height_ * factor);
  std::vector<float> new_data(new_width * new_height);

  for (size_t i = 0; i < 3; ++i) {
    const int offset = i * width_ * height_;
    DownsampleImage(data_.data() + offset, height_, width_,
                    new_height, new_width, new_data.data());
    std::copy(new_data.begin(), new_data.end(), data_.begin() + offset);
  }

  data_ = new_data;
  width_ = new_width;
  height_ = new_height;
}

void NormalMap::Downsize(const size_t max_width, const size_t max_height) {
  if (height_ <= max_height && width_ <= max_width) {
    return;
  }
  const float factor_x = static_cast<float>(max_width) / width_;
  const float factor_y = static_cast<float>(max_height) / height_;
  const float factor = std::min(factor_x, factor_y);
  Rescale(factor);
}

}  // namespace mvs
}  // namespace colmap

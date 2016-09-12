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
  if (width_ * height_ == 0) {
    return;
  }

  const size_t new_width = std::round(width_ * factor);
  const size_t new_height = std::round(height_ * factor);
  std::vector<float> new_data(new_width * new_height * 3);

  // Resample the normal map.
  for (size_t d = 0; d < 3; ++d) {
    const int offset = d * width_ * height_;
    DownsampleImage(data_.data() + offset, height_, width_, new_height,
                    new_width, new_data.data());
    std::copy(new_data.begin(), new_data.end(), data_.begin() + offset);
  }

  data_ = new_data;
  width_ = new_width;
  height_ = new_height;

  // Re-normalize the normal vectors.
  for (size_t r = 0; r < height_; ++r) {
    for (size_t c = 0; c < width_; ++c) {
      Eigen::Vector3f normal(Get(r, c, 0), Get(r, c, 1), Get(r, c, 2));
      normal /= normal.norm();
      Set(r, c, 0, normal(0));
      Set(r, c, 1, normal(1));
      Set(r, c, 2, normal(2));
    }
  }
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

Bitmap NormalMap::ToBitmap() const {
  CHECK_GT(width_, 0);
  CHECK_GT(height_, 0);
  CHECK_EQ(depth_, 3);

  Bitmap bitmap;
  bitmap.Allocate(width_, height_, true);

  for (size_t y = 0; y < height_; ++y) {
    for (size_t x = 0; x < width_; ++x) {
      float normal[3];
      GetSlice(y, x, normal);
      if (normal[0] != 0 || normal[1] != 0 || normal[2] != 0) {
        const BitmapColor<float> color(127.5f * (-normal[0] + 1),
                                       127.5f * (-normal[1] + 1),
                                       -255.0f * normal[2]);
        bitmap.SetPixel(x, y, color.Cast<uint8_t>());
      } else {
        bitmap.SetPixel(x, y, BitmapColor<uint8_t>(0, 0, 0));
      }
    }
  }

  return bitmap;
}

}  // namespace mvs
}  // namespace colmap

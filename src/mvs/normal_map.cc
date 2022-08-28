// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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

#include "mvs/normal_map.h"

#include "base/warp.h"

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
    const size_t offset = d * width_ * height_;
    const size_t new_offset = d * new_width * new_height;
    DownsampleImage(data_.data() + offset, height_, width_, new_height,
                    new_width, new_data.data() + new_offset);
  }

  data_ = new_data;
  width_ = new_width;
  height_ = new_height;

  data_.shrink_to_fit();

  // Re-normalize the normal vectors.
  for (size_t r = 0; r < height_; ++r) {
    for (size_t c = 0; c < width_; ++c) {
      Eigen::Vector3f normal(Get(r, c, 0), Get(r, c, 1), Get(r, c, 2));
      const float squared_norm = normal.squaredNorm();
      if (squared_norm > 0) {
        normal /= std::sqrt(squared_norm);
      }
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
  Rescale(std::min(factor_x, factor_y));
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
        bitmap.SetPixel(x, y, BitmapColor<uint8_t>(0));
      }
    }
  }

  return bitmap;
}

}  // namespace mvs
}  // namespace colmap

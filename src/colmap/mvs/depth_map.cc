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

#include "colmap/mvs/depth_map.h"

#include "colmap/image/warp.h"
#include "colmap/math/math.h"

namespace colmap {
namespace mvs {

DepthMap::DepthMap() : DepthMap(0, 0, -1.0f, -1.0f) {}

DepthMap::DepthMap(const size_t width,
                   const size_t height,
                   const float depth_min,
                   const float depth_max)
    : Mat<float>(width, height, 1),
      depth_min_(depth_min),
      depth_max_(depth_max) {}

DepthMap::DepthMap(const Mat<float>& mat,
                   const float depth_min,
                   const float depth_max)
    : Mat<float>(mat.GetWidth(), mat.GetHeight(), mat.GetDepth()),
      depth_min_(depth_min),
      depth_max_(depth_max) {
  THROW_CHECK_EQ(mat.GetDepth(), 1);
  data_ = mat.GetData();
}

void DepthMap::Rescale(const float factor) {
  if (width_ * height_ == 0) {
    return;
  }

  const size_t new_width = std::round(width_ * factor);
  const size_t new_height = std::round(height_ * factor);
  std::vector<float> new_data(new_width * new_height);
  DownsampleImage(
      data_.data(), height_, width_, new_height, new_width, new_data.data());

  data_ = new_data;
  width_ = new_width;
  height_ = new_height;

  data_.shrink_to_fit();
}

void DepthMap::Downsize(const size_t max_width, const size_t max_height) {
  if (height_ <= max_height && width_ <= max_width) {
    return;
  }
  const float factor_x = static_cast<float>(max_width) / width_;
  const float factor_y = static_cast<float>(max_height) / height_;
  Rescale(std::min(factor_x, factor_y));
}

Bitmap DepthMap::ToBitmap(const float min_percentile,
                          const float max_percentile) const {
  THROW_CHECK_GT(width_, 0);
  THROW_CHECK_GT(height_, 0);

  Bitmap bitmap;
  bitmap.Allocate(width_, height_, true);

  std::vector<float> valid_depths;
  valid_depths.reserve(data_.size());
  for (const float depth : data_) {
    if (depth > 0) {
      valid_depths.push_back(depth);
    }
  }

  if (valid_depths.empty()) {
    bitmap.Fill(BitmapColor<uint8_t>(0));
    return bitmap;
  }

  const float robust_depth_min = Percentile(valid_depths, min_percentile);
  const float robust_depth_max = Percentile(valid_depths, max_percentile);

  const float robust_depth_range = robust_depth_max - robust_depth_min;
  for (size_t y = 0; y < height_; ++y) {
    for (size_t x = 0; x < width_; ++x) {
      const float depth = Get(y, x);
      if (depth > 0) {
        const float robust_depth =
            std::max(robust_depth_min, std::min(robust_depth_max, depth));
        const float gray =
            (robust_depth - robust_depth_min) / robust_depth_range;
        const BitmapColor<float> color(255 * JetColormap::Red(gray),
                                       255 * JetColormap::Green(gray),
                                       255 * JetColormap::Blue(gray));
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

// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/mat.h"
#include "colmap/sensor/bitmap.h"

#include <vector>

namespace colmap {
namespace mvs {

class DepthMap : public Mat<float> {
 public:
  DepthMap();
  DepthMap(size_t width, size_t height, float depth_min, float depth_max);
  DepthMap(const Mat<float>& mat, float depth_min, float depth_max);

  inline float GetDepthMin() const;
  inline float GetDepthMax() const;

  inline float Get(size_t row, size_t col) const;

  void Rescale(float factor);
  void Downsize(size_t max_width, size_t max_height);

  Bitmap ToBitmap(float min_percentile, float max_percentile) const;

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

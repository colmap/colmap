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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#pragma once

#include "colmap/mvs/mat.h"
#include "colmap/sensor/bitmap.h"

#include <string>
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

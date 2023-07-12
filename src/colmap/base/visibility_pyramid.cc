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

#include "colmap/base/visibility_pyramid.h"

#include "colmap/math/math.h"
#include "colmap/util/logging.h"

namespace colmap {

VisibilityPyramid::VisibilityPyramid() : VisibilityPyramid(0, 0, 0) {}

VisibilityPyramid::VisibilityPyramid(const size_t num_levels,
                                     const size_t width,
                                     const size_t height)
    : width_(width), height_(height), score_(0), max_score_(0) {
  pyramid_.resize(num_levels);
  for (size_t level = 0; level < num_levels; ++level) {
    const size_t level_plus_one = level + 1;
    const int dim = 1 << level_plus_one;
    pyramid_[level].setZero(dim, dim);
    max_score_ += dim * dim * dim * dim;
  }
}

void VisibilityPyramid::SetPoint(const double x, const double y) {
  CHECK_GT(pyramid_.size(), 0);

  size_t cx = 0;
  size_t cy = 0;
  CellForPoint(x, y, &cx, &cy);

  for (int i = static_cast<int>(pyramid_.size() - 1); i >= 0; --i) {
    auto& level = pyramid_[i];

    level(cy, cx) += 1;
    if (level(cy, cx) == 1) {
      score_ += level.size();
    }

    cx = cx >> 1;
    cy = cy >> 1;
  }

  CHECK_LE(score_, max_score_);
}

void VisibilityPyramid::ResetPoint(const double x, const double y) {
  CHECK_GT(pyramid_.size(), 0);

  size_t cx = 0;
  size_t cy = 0;
  CellForPoint(x, y, &cx, &cy);

  for (int i = static_cast<int>(pyramid_.size() - 1); i >= 0; --i) {
    auto& level = pyramid_[i];

    level(cy, cx) -= 1;
    if (level(cy, cx) == 0) {
      score_ -= level.size();
    }

    cx = cx >> 1;
    cy = cy >> 1;
  }

  CHECK_LE(score_, max_score_);
}

void VisibilityPyramid::CellForPoint(const double x,
                                     const double y,
                                     size_t* cx,
                                     size_t* cy) const {
  CHECK_GT(width_, 0);
  CHECK_GT(height_, 0);
  const int max_dim = 1 << pyramid_.size();
  *cx = Clamp<size_t>(max_dim * x / width_, 0, max_dim - 1);
  *cy = Clamp<size_t>(max_dim * y / height_, 0, max_dim - 1);
}

}  // namespace colmap

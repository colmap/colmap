// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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

#ifndef COLMAP_SRC_BASE_VISIBILITY_PYRAMID_H_
#define COLMAP_SRC_BASE_VISIBILITY_PYRAMID_H_

#include <vector>

#include <Eigen/Core>

#include "util/alignment.h"

namespace colmap {

// A class that captures the distribution of points in a 2D grid.
// For example, to capture the distribution of visible 3D points in an image.
//
// The class captures the distribution of points by a score. A higher score
// corresponds to a more uniform distribution of the points in the grid.
//
// The score is computed by the number of populated cells in a multi-resolution
// pyramid. A populated cell contributes to the overall score if it is
// populated by at least one point and the contributed score is according
// to its resolution in the pyramid. A cell in a higher resolution level
// contributes a higher score to the overall score.
class VisibilityPyramid {
 public:
  VisibilityPyramid();
  VisibilityPyramid(const size_t num_levels, const size_t width,
                    const size_t height);

  void SetPoint(const double x, const double y);
  void ResetPoint(const double x, const double y);

  inline size_t NumLevels() const;
  inline size_t Width() const;
  inline size_t Height() const;

  inline size_t Score() const;
  inline size_t MaxScore() const;

 private:
  void CellForPoint(const double x, const double y, size_t* cx,
                    size_t* cy) const;

  // Range of the input points.
  size_t width_;
  size_t height_;

  // The overall visibility score.
  size_t score_;

  // The maximum score when all cells are populated.
  size_t max_score_;

  // The visibilty pyramid with multiple levels.
  std::vector<Eigen::MatrixXi> pyramid_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

size_t VisibilityPyramid::NumLevels() const { return pyramid_.size(); }

size_t VisibilityPyramid::Width() const { return width_; }

size_t VisibilityPyramid::Height() const { return height_; }

size_t VisibilityPyramid::Score() const { return score_; }

size_t VisibilityPyramid::MaxScore() const { return max_score_; }

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_VISIBILITY_PYRAMID_H_

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

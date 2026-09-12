// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/eigen_alignment.h"

#include <vector>

#include <Eigen/Core>

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
  VisibilityPyramid(size_t num_levels, size_t width, size_t height);

  void SetPoint(double x, double y);
  void ResetPoint(double x, double y);

  inline size_t NumLevels() const;
  inline size_t Width() const;
  inline size_t Height() const;

  inline size_t Score() const;
  inline size_t MaxScore() const;

 private:
  void CellForPoint(double x, double y, size_t* cx, size_t* cy) const;

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

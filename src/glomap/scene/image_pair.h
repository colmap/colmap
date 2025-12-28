#pragma once

#include "colmap/estimators/two_view_geometry.h"

#include "glomap/scene/types.h"

#include <Eigen/Core>

namespace glomap {

// ImagePair extends TwoViewGeometry.
struct ImagePair : public colmap::TwoViewGeometry {
  ImagePair() = default;

  explicit ImagePair(Rigid3d cam2_from_cam1) {
    this->cam2_from_cam1 = std::move(cam2_from_cam1);
  }

  // indicator whether the image pair is valid
  bool is_valid = true;

  // weight is the initial inlier rate
  double weight = -1;

  // Matches between the two images.
  // First column is the index of the feature in the first image.
  // Second column is the index of the feature in the second image.
  Eigen::MatrixXi matches;

  // Row index of inliers in the matches matrix.
  std::vector<int> inliers;

  // Invert the geometry to swap image order. Extends TwoViewGeometry::Invert()
  // to also swap columns of matches matrix.
  void Invert() {
    TwoViewGeometry::Invert();
    if (matches.rows() > 0) {
      matches.col(0).swap(matches.col(1));
    }
  }
};

}  // namespace glomap

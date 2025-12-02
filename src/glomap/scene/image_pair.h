#pragma once

#include "colmap/estimators/two_view_geometry.h"

#include "glomap/scene/types.h"
#include "glomap/types.h"

#include <Eigen/Core>

namespace glomap {

// TODO: add covariance to the relative pose
struct ImagePair {
  ImagePair()
      : image_id1(colmap::kInvalidImageId),
        image_id2(colmap::kInvalidImageId),
        pair_id(colmap::kInvalidImagePairId) {}
  ImagePair(image_t image_id1,
            image_t image_id2,
            Rigid3d cam2_from_cam1 = Rigid3d())
      : image_id1(image_id1),
        image_id2(image_id2),
        pair_id(colmap::ImagePairToPairId(image_id1, image_id2)),
        cam2_from_cam1(std::move(cam2_from_cam1)) {}

  // Ids are kept constant
  const image_t image_id1;
  const image_t image_id2;
  const image_pair_t pair_id;

  // indicator whether the image pair is valid
  bool is_valid = true;

  // weight is the initial inlier rate
  double weight = -1;

  // one of `ConfigurationType`.
  int config = colmap::TwoViewGeometry::UNDEFINED;

  // Essential matrix.
  Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
  // Fundamental matrix.
  Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
  // Homography matrix.
  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();

  // Relative pose.
  Rigid3d cam2_from_cam1 = Rigid3d();

  // Matches between the two images.
  // First column is the index of the feature in the first image.
  // Second column is the index of the feature in the second image.
  Eigen::MatrixXi matches;

  // Row index of inliers in the matches matrix.
  std::vector<int> inliers;
};

}  // namespace glomap

#pragma once

#include "colmap/scene/image.h"

#include <vector>

#include <Eigen/Core>

namespace glomap {

class Image : public colmap::Image {
 public:
  // Distorted feature points in pixels.
  std::vector<Eigen::Vector2d> features;
  // Normalized feature rays, can be obtained by calling UndistortImages.
  std::vector<Eigen::Vector3d> features_undist;
};

}  // namespace glomap

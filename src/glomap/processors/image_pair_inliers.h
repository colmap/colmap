#pragma once

#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/models.h"

#include "glomap/scene/view_graph.h"

namespace glomap {

struct InlierThresholdOptions {
  // Thresholds for 3D-2D matches
  double max_angle_error = 1.;           // in degree, for global positioning
  double max_reprojection_error = 1e-2;  // for bundle adjustment
  double min_triangulation_angle = 1.;   // in degree, for triangulation

  // Thresholds for image_pair
  double max_epipolar_error_E = 1.;
  double max_epipolar_error_F = 4.;
  double max_epipolar_error_H = 4.;

  // Thresholds for edges
  double min_inlier_num = 30;
  double min_inlier_ratio = 0.25;
  double max_rotation_error = 10.;  // in degree, for rotation averaging
};

class ImagePairInliers {
 public:
  ImagePairInliers(image_t image_id1,
                   image_t image_id2,
                   ImagePair& image_pair,
                   const colmap::Reconstruction& reconstruction,
                   const InlierThresholdOptions& options)
      : image_id1(image_id1),
        image_id2(image_id2),
        image_pair(image_pair),
        reconstruction(reconstruction),
        options(options) {}

  // use the sampson error and put the inlier result into the image pair
  double ScoreError();

 protected:
  // Error for the case of essential matrix
  double ScoreErrorEssential();

  // Error for the case of fundamental matrix
  double ScoreErrorFundamental();

  // Error for the case of homography matrix
  double ScoreErrorHomography();

  image_t image_id1;
  image_t image_id2;
  ImagePair& image_pair;
  const colmap::Reconstruction& reconstruction;
  const InlierThresholdOptions& options;
};

void ImagePairsInlierCount(ViewGraph& view_graph,
                           const colmap::Reconstruction& reconstruction,
                           const InlierThresholdOptions& options,
                           bool clean_inliers);

}  // namespace glomap

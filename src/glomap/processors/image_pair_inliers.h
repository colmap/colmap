#pragma once

#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/models.h"

#include "glomap/scene/view_graph.h"
#include "glomap/types.h"

namespace glomap {

class ImagePairInliers {
 public:
  ImagePairInliers(ImagePair& image_pair,
                   const colmap::Reconstruction& reconstruction,
                   const InlierThresholdOptions& options)
      : image_pair(image_pair),
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

  ImagePair& image_pair;
  const colmap::Reconstruction& reconstruction;
  const InlierThresholdOptions& options;
};

void ImagePairsInlierCount(ViewGraph& view_graph,
                           const colmap::Reconstruction& reconstruction,
                           const InlierThresholdOptions& options,
                           bool clean_inliers);

}  // namespace glomap

#pragma once

#include "glomap/math/rigid3d.h"
#include "glomap/scene/types_sfm.h"
#include "glomap/types.h"

namespace glomap {

class ImagePairInliers {
 public:
  ImagePairInliers(
      ImagePair& image_pair,
      const std::unordered_map<image_t, Image>& images,
      const InlierThresholdOptions& options,
      const std::unordered_map<camera_t, Camera>* cameras = nullptr)
      : image_pair(image_pair),
        images(images),
        cameras(cameras),
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
  const std::unordered_map<image_t, Image>& images;
  const std::unordered_map<camera_t, Camera>* cameras;
  const InlierThresholdOptions& options;
};

void ImagePairsInlierCount(ViewGraph& view_graph,
                           const std::unordered_map<camera_t, Camera>& cameras,
                           const std::unordered_map<image_t, Image>& images,
                           const InlierThresholdOptions& options,
                           bool clean_inliers);

}  // namespace glomap

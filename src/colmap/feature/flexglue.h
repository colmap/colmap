// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/matcher.h"
#include "colmap/feature/resources.h"

#include <memory>
#include <string>

namespace colmap {

// FlexGlue: Sparse-to-Dense Feature Matching and Visual Disambiguation.
// Reference: https://github.com/cvg/FlexGlue
//
// FlexGlue matches through dense image features: each image is encoded into a
// stride-8 feature map by a DINOv3-ConvNeXt backbone, a transformer refines
// both maps jointly, and every keypoint finds its correspondence in the other
// image by correlation. Keypoints are then matched by mutual nearest neighbors
// between the keypoints and the regressed correspondences ("sparse" mode).
// FlexGlue only uses the keypoint locations, so it works with the keypoints of
// any feature extractor, and it ignores the descriptors. Since it requires the
// image pixels, the images are read from image_path during matching.
//
// FlexGlue also predicts whether the two images are covisible at all, which
// rejects look-alike pairs: all matches of a pair whose score is below
// min_covisibility are rejected inside the model.
struct FlexGlueMatchingOptions {
  // Root folder of the images (required). The images are read from this
  // folder during matching, as FlexGlue computes dense features from the image
  // pixels. Matching fails if an image cannot be read.
  std::string image_path;

  // Minimum match confidence, i.e., the geometric mean of the confidences of
  // both matching directions. Increase for fewer but more confident matches.
  double filter_threshold = 0.1;

  // Maximum distance between a keypoint and the predicted correspondence of its
  // mutual nearest neighbor, averaged over both matching directions, in pixels
  // at the descriptor resolution (512x512 by default): in the original images,
  // the threshold is scaled by the long image side / 512, averaged over both
  // images. Non-positive values disable the check.
  double max_distance_px = 3.0;

  // Reject all matches of a pair whose covisibility score is below this value,
  // which suppresses look-alike pairs (e.g., of repeated structures). On ETH3D
  // and IMC 2023, values in 0.5-0.8 clearly improve the reconstructions over no
  // rejection. A value of 0 disables the pair rejection.
  double min_covisibility = 0.5;

  // Early stopping of non-covisible pairs: after each checked layer k (2, 4, 6,
  // 8 for the default models), the pair is rejected without running the
  // remaining layers if its covisibility score is below
  // min_covisibility * (1 - early_stop_decay)^(9 - k). Lower values stop
  // earlier and save more time, but may reject pairs that would pass
  // min_covisibility after the last layer. 1 disables early stopping, as does
  // min_covisibility = 0.
  double early_stop_decay = 0.5;

  // Maximum number of images whose dense features are cached in memory, on the
  // GPU if the models run on a CUDA GPU, which avoids copying them to the GPU
  // for every pair. Each entry takes ~4MB with the default model.
  int feature_cache_size = 128;

  // Dense image descriptor: image [1, 3, S, S] -> features [1, C, S/8, S/8].
  std::string descriptor_model_path = kDefaultFlexGlueDescriptorUri;

  // Sparse matcher operating on the dense features and keypoints.
  std::string matcher_model_path = kDefaultFlexGlueMatcherUri;

  // Variant of the matcher whose transformer layers run in float16, used
  // instead of matcher_model_path when use_fp16 is set and the matcher runs on
  // a CUDA GPU. About 2.5x faster with practically identical matches.
  std::string matcher_model_path_fp16 = kDefaultFlexGlueMatcherFp16Uri;

  // Whether to use the float16 matcher on CUDA GPUs. Falls back to the float32
  // matcher if the float16 matcher cannot be initialized. Ignored on the CPU.
  bool use_fp16 = true;

  bool Check() const;
};

std::unique_ptr<FeatureMatcher> CreateFlexGlueFeatureMatcher(
    const FeatureMatchingOptions& options);

}  // namespace colmap

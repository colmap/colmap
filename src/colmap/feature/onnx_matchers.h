// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/matcher.h"

#include <vector>

namespace colmap {

struct BruteForceONNXMatchingOptions {
  // The minimum cosine similarity for a match to be considered valid
  // in brute-force matching.
  double min_cossim = 0.85;

  // Maximum ratio for Lowe's ratio test (second-best / best distance).
  double max_ratio = 1.0;

  // Enable cross-checking (mutual nearest neighbor).
  bool cross_check = true;

  // The path to the ONNX model file for the brute-force matcher.
  std::string model_path;

  bool Check() const;
};

// The matcher rejects descriptors whose type is not listed in
// `supported_feature_types`. Set `normalize_descriptors` when the extractor
// does not produce unit-length descriptors.
std::unique_ptr<FeatureMatcher> CreateBruteForceONNXFeatureMatcher(
    const FeatureMatchingOptions& options,
    const BruteForceONNXMatchingOptions& brute_force_options,
    std::vector<FeatureExtractorType> supported_feature_types,
    bool normalize_descriptors);

}  // namespace colmap

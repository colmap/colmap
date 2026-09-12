// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/extractor.h"
#include "colmap/feature/matcher.h"
#include "colmap/feature/onnx_matchers.h"
#include "colmap/feature/resources.h"

namespace colmap {

// The ALIKED torch model was exported to ONNX using the following codebase:
// https://github.com/colmap/ALIKED-ONNX/tree/user/jsch/onnx-export
// Follow instructions in export/README.md and see standalone C++
// implementation in cpp_test/README.md.

struct AlikedExtractionOptions {
  // Maximum number of features to detect, keeping higher-score features.
  int max_num_features = 2048;

  // Minimum score threshold for keypoint detection.
  double min_score = 0.2;

  // The path to the ONNX model file for the ALIKED extractor.
  // Must be a sparse model with max_keypoints and min_score inputs.
  std::string n16rot_model_path = kDefaultAlikedN16RotFeatureExtractorUri;
  std::string n32_model_path = kDefaultAlikedN32FeatureExtractorUri;

  bool Check() const;
};

std::unique_ptr<FeatureExtractor> CreateAlikedFeatureExtractor(
    const FeatureExtractionOptions& options);

struct AlikedMatchingOptions {
  // Brute-force matching options.
  BruteForceONNXMatchingOptions brute_force = []() {
    BruteForceONNXMatchingOptions options;
    options.min_cossim = 0.85;
    options.max_ratio = 1.0;
    options.cross_check = true;
    options.model_path = kDefaultBruteForceONNXMatcherUri;
    return options;
  }();

  // LightGlue matching options.
  LightGlueONNXMatchingOptions lightglue = []() {
    LightGlueONNXMatchingOptions options;
    options.min_score = 0.1;
    options.model_path = kDefaultAlikedLightGlueFeatureMatcherUri;
    return options;
  }();

  bool Check() const;
};

std::unique_ptr<FeatureMatcher> CreateAlikedFeatureMatcher(
    const FeatureMatchingOptions& options);

}  // namespace colmap

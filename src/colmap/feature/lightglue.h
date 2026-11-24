// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/matcher.h"

namespace colmap {

// The LightGlue torch model was exported to ONNX using the following codebase:
// https://github.com/colmap/LightGlue-ONNX/tree/user/jsch/onnx-export
// Follow instructions in export/README.md and see standalone C++
// implementation in cpp_test/README.md.

struct LightGlueONNXMatchingOptions {
  // Minimum match score threshold. Matches with scores below this
  // value are discarded (post-model filtering).
  double min_score = 0.1;

  // Path to the LightGlue ONNX model file.
  std::string model_path;

  bool Check() const;
};

std::unique_ptr<FeatureMatcher> CreateLightGlueONNXFeatureMatcher(
    const FeatureMatchingOptions& options,
    const LightGlueONNXMatchingOptions& lightglue_options);

}  // namespace colmap

// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/feature/extractor.h"
#include "colmap/feature/matcher.h"
#include "colmap/feature/resources.h"

namespace colmap {

// Reference: https://github.com/davnords/LoMa

struct LomaExtractionOptions {
  // Number of keypoints to detect. Recommended is to use either 2048 or 4096.
  int max_num_features = 2048;

  // The DaD detector does not filter in the same way as ALIKED. Thus, we set min_score = 0.0.
  double min_score = 0.0;

  // NOTE: Do not change. Internal descriptor dimension. The ONNX graph is exported with a fixed descriptor dimension.
  int descriptor_size = 784;

  std::string detector_model_path = kDefaultLomaBDetectorUri;
  std::string descriptor_model_path = kDefaultLomaBDescriptorUri;

  bool Check() const;
};

std::unique_ptr<FeatureExtractor> CreateLomaFeatureExtractor(
    const FeatureExtractionOptions& options);

struct LomaMatchingOptions {
  // Matching filter, matches LG
  double min_score = 0.1;

  std::string model_path = kDefaultLomaBMatcherUri;

  bool Check() const;
};

std::unique_ptr<FeatureMatcher> CreateLomaFeatureMatcher(
    const FeatureMatchingOptions& options);

}  // namespace colmap

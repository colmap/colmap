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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#pragma once

#include "colmap/feature/extractor.h"
#include "colmap/feature/matcher.h"
#include "colmap/feature/resources.h"

namespace colmap {

struct ALIKEDExtractionOptions {
  // Maximum image size, otherwise image will be down-scaled.
  int max_image_size = 2000;

  // Maximum number of features to detect (in combination with score threshold).
  int max_num_features = 4096;

  // Minimum detection score threshold.
  double score_threshold = 0.2;

  // Detect fixed number of top k features independent of detection score.
  // Ignored if negative.
  int top_k = -1;

  // The ALIKED model name.
  std::string model_name = "aliked-n32";

  // The path to the ALIKED model.
  std::string model_path = kDefaultAlikedUri;

  bool Check() const;
};

std::unique_ptr<FeatureExtractor> CreateALIKEDFeatureExtractor(
    const FeatureExtractionOptions& options);

struct ALIKEDMatchingOptions {
  // Minimum dot product similarity score.
  double min_similarity = 0.75;

  // Path to .pt ALIKED model file in torch model format.
  std::string lightglue_model_path = kDefaultAlikedLightGlueUri;

  bool Check() const;
};

std::unique_ptr<FeatureMatcher> CreateALIKEDFeatureMatcher(
    const FeatureMatchingOptions& options);

}  // namespace colmap

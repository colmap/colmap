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

struct AlikedExtractionOptions {
  // Maximum number of features to detect, keeping higher-score features.
  int max_num_features = 2048;

  // The path to the ONNX model file for the ALIKED extractor.
  // Must be a sparse model with max_keypoints input.
  std::string model_path = kDefaultALIKEDN32FeatureExtractorUri;

  bool Check() const;
};

std::unique_ptr<FeatureExtractor> CreateAlikedFeatureExtractor(
    const FeatureExtractionOptions& options);

struct AlikedMatchingOptions {
  // The minimum cosine similarity for a match to be considered valid
  // in brute-force matching.
  double min_cossim = 0.85;

  // The path to the ONNX model file for the brute-force matcher.
  std::string bruteforce_model_path = kDefaultONNXBruteForceMatcherUri;

  bool Check() const;
};

std::unique_ptr<FeatureMatcher> CreateAlikedFeatureMatcher(
    const FeatureMatchingOptions& options);

}  // namespace colmap

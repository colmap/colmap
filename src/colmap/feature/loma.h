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
#include "colmap/feature/onnx_matchers.h"
#include "colmap/feature/resources.h"

namespace colmap {

// Reference: https://github.com/davnords/LoMa

struct LomaExtractionOptions {
  // Number of keypoints to detect. Recommended is to use either 2048 or 4096.
  int max_num_features = 2048;

  // The DaD detector does not filter in the same way as ALIKED. Thus, we set
  // min_score = 0.0.
  double min_score = 0.0;

  // NOTE: Do not change. Internal descriptor dimension. The ONNX graph is
  // exported with a fixed descriptor dimension.
  int descriptor_size = 784;

  // Use the bf16 descriptor variant for faster inference. Only affects
  // LOMA_B -- no-op for LOMA_B128 (no DINO backbone to cast).
  bool use_bf16 = false;

  // Use a fast bilinear resample instead of the filtered resize for the
  // descriptor's fixed input size. ~2-3x faster extraction; small accuracy
  // cost only at the tightest rotation thresholds. Defaults off for accuracy.
  bool use_fast_resize = false;

  // Detector (DaD) is shared across all LoMa variants -- one file, always
  // fp32 (cheap, not a speed bottleneck).
  std::string detector_model_path = kDefaultLomaBDetectorUri;

  // Descriptor for FeatureExtractorType::LOMA_B (dedode_g, also used by the
  // LOMA_B/LOMA_R/LOMA_L/LOMA_G matcher variants).
  std::string descriptor_model_path = kDefaultLomaBDescriptorUri;
  std::string descriptor_model_path_bf16 = kDefaultLomaBDescriptorBf16Uri;

  // Descriptor for FeatureExtractorType::LOMA_B128 (dedode_b, VGG-only).
  std::string descriptor_b128_model_path = kDefaultLomaB128DescriptorUri;

  bool Check() const;
};

std::unique_ptr<FeatureExtractor> CreateLomaFeatureExtractor(
    const FeatureExtractionOptions& options);

// Model paths for one LoMa matcher variant (B/B128/R/L/G), at both
// precisions.
struct LomaVariantMatcherOptions {
  std::string model_path;
  std::string model_path_bf16;
};

struct LomaMatchingOptions {
  // Matching filter, matches LG
  double min_score = 0.1;

  // Use the bf16 matcher variant for faster inference.
  bool use_bf16 = false;

  // One entry per FeatureMatcherType::LOMA_* dedicated-matcher variant.
  LomaVariantMatcherOptions b = {kDefaultLomaBMatcherUri,
                                 kDefaultLomaBMatcherBf16Uri};
  LomaVariantMatcherOptions b128 = {kDefaultLomaB128MatcherUri,
                                    kDefaultLomaB128MatcherBf16Uri};
  LomaVariantMatcherOptions r = {kDefaultLomaRMatcherUri,
                                 kDefaultLomaRMatcherBf16Uri};
  LomaVariantMatcherOptions l = {kDefaultLomaLMatcherUri,
                                 kDefaultLomaLMatcherBf16Uri};
  LomaVariantMatcherOptions g = {kDefaultLomaGMatcherUri,
                                 kDefaultLomaGMatcherBf16Uri};

  // Brute-force matching options (reuses the generic ONNX matcher, which
  // is descriptor-dimension agnostic).
  BruteForceONNXMatchingOptions brute_force = []() {
    BruteForceONNXMatchingOptions options;
    options.min_cossim = 0.85;
    options.max_ratio = 1.0;
    options.cross_check = true;
    options.model_path = kDefaultBruteForceONNXMatcherUri;
    return options;
  }();

  bool Check() const;
};

std::unique_ptr<FeatureMatcher> CreateLomaFeatureMatcher(
    const FeatureMatchingOptions& options);

}  // namespace colmap

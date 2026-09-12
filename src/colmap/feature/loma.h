// SPDX-License-Identifier: BSD-3-Clause

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

  // Prefer bf16 for LOMA_B when supported; otherwise use fp32. No-op for
  // LOMA_B128.
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

struct LomaMatchingOptions {
  // Per-variant fp32 and bf16 model paths.
  struct Variant {
    std::string model_path;
    std::string model_path_bf16;
  };

  // Matching filter, matches LG
  double min_score = 0.1;

  // Prefer bf16 when supported; otherwise use fp32.
  bool use_bf16 = false;

  // One entry per FeatureMatcherType::LOMA_* dedicated-matcher variant.
  Variant b = {kDefaultLomaBMatcherUri, kDefaultLomaBMatcherBf16Uri};
  Variant b128 = {kDefaultLomaB128MatcherUri, kDefaultLomaB128MatcherBf16Uri};
  Variant r = {kDefaultLomaRMatcherUri, kDefaultLomaRMatcherBf16Uri};
  Variant l = {kDefaultLomaLMatcherUri, kDefaultLomaLMatcherBf16Uri};
  Variant g = {kDefaultLomaGMatcherUri, kDefaultLomaGMatcherBf16Uri};

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

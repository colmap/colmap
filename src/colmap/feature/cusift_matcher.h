// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/matcher.h"
#include "colmap/util/enum_utils.h"

#include <memory>

namespace colmap {

// Metric space of the ratio test and of sift.max_distance. L2: ratio of
// unit-vector L2 distances, max_distance in unit-vector L2, as COLMAP's
// index-based SIFT matcher. ANGULAR: ratio of angles, max_distance in radians,
// as COLMAP's brute-force SIFT matcher and SiftGPU.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(CusiftDistanceType, 0, L2, ANGULAR);

// Options for FeatureMatcherType::CUSIFT_GEMM beyond FeatureMatchingOptions::
// sift, whose max_ratio, max_distance and cross_check it uses.
struct CusiftMatchingOptions {
  // With sift.cross_check, also apply the ratio test in the reverse direction,
  // as both of COLMAP's SIFT matchers do. Disabled, cross-checking requires
  // only mutual best matches.
  bool cross_check_reverse_ratio = true;

  // ANGULAR is what COLMAP's exact brute-force matcher does
  // (FindBestMatchesOneWayBruteForce).
  CusiftDistanceType distance_type = CusiftDistanceType::ANGULAR;

  bool Check() const;
};

// CUDA (cuBLAS SGEMM) brute-force matcher for FeatureExtractorType::SIFT
// descriptors. Candidates are ranked on the raw inner product with COLMAP's
// fixed 512^2 descriptor norm, as COLMAP's brute-force matcher and SiftGPU do.
// max_num_matches keeps the first matches in image1 keypoint order, as SiftGPU
// does. Guided matching is not supported. Returns nullptr if COLMAP was built
// without cusift (CUSIFT_ENABLED, which needs CUDA >= 11.2).
std::unique_ptr<FeatureMatcher> CreateCusiftFeatureMatcher(
    const FeatureMatchingOptions& options);

}  // namespace colmap

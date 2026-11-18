// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/matcher.h"
#include "colmap/util/enum_utils.h"

#include <memory>

namespace colmap {

// Metric of the matcher: it picks the winning candidate and sets the space of
// the ratio test and of sift.max_distance. L2: ranks candidates on true
// distance, ratio of unit-vector L2 distances, max_distance in unit-vector L2,
// as COLMAP's index-based SIFT matcher. ANGULAR: ranks on the raw inner
// product, ratio of angles, max_distance in radians, as COLMAP's brute-force
// SIFT matcher and SiftGPU. The two can pick different winners when
// descriptor norms differ.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(CusiftDistanceType, 0, L2, ANGULAR);

// Options for FeatureMatcherType::CUSIFT_GEMM beyond FeatureMatchingOptions::
// sift, whose max_ratio, max_distance and cross_check it uses.
struct CusiftMatchingOptions {
  // With sift.cross_check, also apply the ratio test in the reverse direction,
  // as both of COLMAP's SIFT matchers do. Disabled, cross-checking requires
  // only mutual best matches.
  bool cross_check_reverse_ratio = true;

  // L2, the default, applies the ratio test of COLMAP's default CPU matcher
  // (FindBestMatchesIndex) with exact instead of approximate search. ANGULAR
  // is what COLMAP's exact brute-force matcher does
  // (FindBestMatchesOneWayBruteForce).
  CusiftDistanceType distance_type = CusiftDistanceType::L2;

  // Set by COLMAP's matching workers after clamping max_num_matches to the
  // database's largest keypoint count: Create then sizes the device buffers
  // for that many descriptors upfront. Otherwise they grow to each pair.
  bool reserve_max_num_matches = false;

  bool Check() const;
};

// |q - c|^2 from |q|^2 and the GEMM matcher's L2 reduction key 2 q.c - |c|^2.
// Exact for uint8 descriptors; clamped at 0, as FAISS does, in case
// reduced-precision math rounds it below 0.
float SqDistanceFromGemmKey(float sq_norm, float key);

// CUDA (cuBLAS SGEMM) brute-force matcher for FeatureExtractorType::SIFT
// descriptors. By default (distance_type L2), candidates are ranked and
// accepted on L2 distance, as COLMAP's default CPU matcher does. With ANGULAR,
// they are ranked on the raw inner product with COLMAP's fixed 512^2
// descriptor norm, as COLMAP's brute-force matcher and SiftGPU do.
// max_num_matches keeps the first matches in image1 keypoint order, as SiftGPU
// does. Guided matching is not supported. Returns nullptr if COLMAP was built
// without cusift (CUSIFT_ENABLED, which needs CUDA >= 11.2).
//
// The matcher is bound to the GPU named by options.gpu_index, which Create
// makes the calling thread's current device (BindCusiftDevice); Match binds
// its calling thread to that GPU again. Use one matcher from one thread at a
// time, as COLMAP's controllers do.
std::unique_ptr<FeatureMatcher> CreateCusiftFeatureMatcher(
    const FeatureMatchingOptions& options);

}  // namespace colmap

// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/matcher.h"
#include "colmap/util/enum_utils.h"

#include <memory>

namespace colmap {

// Metric space in which the ratio test and max_distance threshold are
// evaluated. Top-2 selection is identical either way (the three spaces are
// monotone in one another), and max_distance converts exactly between them;
// only the ratio test genuinely differs, since a ratio of angles is not a ratio
// of distances.
//
// Note this changes the UNITS of max_distance: unit-vector L2 distance under
// L2, radians under ANGULAR.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(CusiftDistanceType,
                                0,
                                // Ratio of unit-vector L2 distances, the
                                // convention of COLMAP's index-based CPU SIFT
                                // matcher and VLFeat's vl_ubcmatch. Cheapest
                                // on GPU: the GEMM already produced the cosine.
                                L2,
                                // Ratio of angles via acos -- behaviourally
                                // what SiftGPU does (its behaviour only; see
                                // the LICENSING note in
                                // src/thirdparty/cusift/src/matcher.h).
                                ANGULAR);

// Matching options for FeatureMatcherType::CUSIFT_GEMM that SIFT matching has
// no equivalent for. The ratio test, distance threshold and cross-check come
// from FeatureMatchingOptions::sift (max_ratio, max_distance, cross_check),
// with the same meaning as for SIFT_BRUTEFORCE: max_distance is in radians
// under ANGULAR, as on SIFT's brute-force CPU path, and unit-vector L2 distance
// under L2, as on its index-based path.
struct CusiftMatchingOptions {
  // When sift.cross_check is enabled, whether the ratio test is ALSO applied in
  // the reverse direction, which is what both of COLMAP's existing SIFT
  // matchers do -- the CPU path runs FindBestMatchesOneWayIndex once per
  // direction before intersecting (sift.cc, FindBestMatchesIndex), and SiftGPU
  // passes distmax/ratiomax into GetColMatch as well as GetRowMatch
  // (SiftGPU/ProgramCU.cu). Leave enabled for behavioural parity with the
  // matcher this one replaces.
  //
  // Disabling it requires only mutual-best-index agreement, which is weaker at
  // rejecting matches between images that do not overlap.
  bool cross_check_reverse_ratio = true;

  // Metric space for the ratio test and max_distance -- see CusiftDistanceType.
  // Defaults to ANGULAR because that is what COLMAP's EXACT reference matcher
  // does: FindBestMatchesOneWayBruteForce (sift.cc) ranks on the raw inner
  // product and thresholds acos(dot / 512^2), and SiftGPU does the same. L2
  // instead follows COLMAP's index-based CPU matcher, whose results differ
  // slightly from that exact reference.
  CusiftDistanceType distance_type = CusiftDistanceType::ANGULAR;

  bool Check() const;
};

// Create a cusift GEMM feature matcher: a CUDA (cuBLAS SGEMM) alternative to
// SiftGPUFeatureMatcher for FeatureExtractorType::SIFT descriptors (uint8, UBC
// layout).
//
//  - Candidates are ranked on the raw inner product of the uint8 descriptors,
//    scaled by COLMAP's fixed 512^2 descriptor norm rather than per-row norms,
//    as COLMAP's brute-force SIFT matcher and SiftGPU do.
//  - max_ratio, max_distance and cross_check are applied in the kernel, in the
//    space distance_type selects.
//  - max_num_matches keeps the first matches in ascending image1 keypoint
//    order, as SiftGPU does (SiftMatchCU::GetBestMatch), not the best-scoring.
//  - guided_matching is not supported and is rejected.
//
// Requires CUDA and a GPU; returns nullptr if COLMAP was built without CUDA.
std::unique_ptr<FeatureMatcher> CreateCusiftFeatureMatcher(
    const FeatureMatchingOptions& options);

}  // namespace colmap

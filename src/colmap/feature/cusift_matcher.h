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
                                // Ratio of unit-vector L2 distances -- what
                                // COLMAP's CPU SIFT matcher and VLFeat's
                                // vl_ubcmatch do. Cheapest on GPU: the GEMM
                                // already produced the cosine.
                                L2,
                                // Ratio of angles via acos -- behaviourally
                                // what SiftGPU does (its behaviour only; see
                                // the LICENSING note in
                                // src/thirdparty/cusift/src/matcher.h).
                                ANGULAR);

// Matching options for FeatureMatcherType::CUSIFT_GEMM. A separate struct
// from SiftMatchingOptions -- even though the numerics below are provably
// equivalent -- to match the codebase's own precedent (AlikedMatchingOptions
// vs SiftMatchingOptions, aliked.h) of never letting the same field name mean
// something subtly different depending on FeatureMatching.type.
struct CusiftMatchingOptions {
  // Maximum distance ratio between first and second best match. Same
  // convention as SiftMatchingOptions::max_ratio: cusift's GEMM matcher
  // squares this internally and compares against squared unit-vector
  // distances, algebraically identical to COLMAP's own ratio test once both
  // sides are expressed in that space (see cusift_matcher.cc).
  double max_ratio = 0.8;

  // Maximum distance to the best match, in the units distance_type implies:
  // radians under ANGULAR (identical to SiftMatchingOptions::max_distance,
  // which SiftGPU also treats as radians), unit-vector L2 distance under L2.
  double max_distance = 0.7;

  // Whether to enable cross-checking (mutual nearest neighbor).
  bool cross_check = true;

  // When cross_check is enabled, whether the ratio test is ALSO applied in the
  // reverse direction, which is what both of COLMAP's existing SIFT matchers
  // do -- the CPU path runs FindBestMatchesOneWayIndex once per direction
  // before intersecting (sift.cc, FindBestMatchesIndex), and SiftGPU passes
  // distmax/ratiomax into GetColMatch as well as GetRowMatch
  // (SiftGPU/ProgramCU.cu). Leave enabled for behavioural parity with the
  // matcher this one replaces.
  //
  // Disabling it requires only mutual-best-index agreement: the original
  // upstream cusift behaviour, and measurably weaker at rejecting matches
  // between images that do not overlap (see src/thirdparty/cusift/README.md
  // for the measured comparison). Retained as an option so the difference is
  // testable, not because it is a recommended operating point.
  bool cross_check_reverse_ratio = true;

  // Metric space for the ratio test and max_distance -- see CusiftDistanceType.
  // Defaults to ANGULAR because that is what COLMAP's EXACT reference matcher
  // does: FindBestMatchesOneWayBruteForce (sift.cc) ranks on the raw inner
  // product and thresholds acos(dot / 512^2), and SiftGPU does the same. L2
  // instead reproduces COLMAP's other, inconsistent convention -- the
  // faiss/index CPU path -- which was measured at only 0.976 Jaccard against
  // the exact reference. See src/thirdparty/cusift/README.md.
  CusiftDistanceType distance_type = CusiftDistanceType::ANGULAR;

  bool Check() const;
};

// Create a cusift GEMM-based feature matcher: a CUDA cuBLAS SGEMM matcher
// that is a drop-in alternative to SiftGPUFeatureMatcher, operating on the
// same FeatureExtractorType::SIFT descriptors (uint8, UBC layout) that every
// other SIFT matcher consumes.
//
// Functional parity notes (see src/thirdparty/cusift/README.md for the
// numeric derivation and gate numbers):
//  - max_ratio and cross_check map directly onto the vendored
//    MatchDescriptorsGemm's ratio/mutual parameters once descriptors are
//    L2-normalized to true unit vectors -- no conversion needed beyond that
//    normalization, which is the same one onnx_matchers.cc already applies
//    for LightGlue matching (ToFloat() + L2NormalizeFeatureDescriptors).
//  - max_distance has no upstream equivalent; enforced here as a wrapper
//    post-filter on the per-match squared unit-distance the vendored
//    function already returns (Match::dist2).
//  - max_num_matches (FeatureMatchingOptions::max_num_matches) has no
//    upstream equivalent either; enforced here as a wrapper truncation that
//    replicates SiftMatchGPU's own cap-binds selection rule -- ascending
//    image1 keypoint-index order, first-encountered wins, NOT a
//    distance-sorted top-K (see SiftMatchCU::GetBestMatch,
//    src/thirdparty/SiftGPU/SiftMatchCU.cpp).
//  - guided_matching is rejected loudly (THROW_CHECK) -- not implemented in
//    this phase; a real epipolar-windowed candidate filter is a separate,
//    later workstream.
//
// Requires CUDA and a GPU; returns nullptr if COLMAP was built without CUDA.
std::unique_ptr<FeatureMatcher> CreateCusiftFeatureMatcher(
    const FeatureMatchingOptions& options);

}  // namespace colmap

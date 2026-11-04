// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yonatan Simson (see ../LICENSE)

#pragma once
#include <vector>

struct Match { int i, j; float dist2; };

// Metric space in which the ratio test is evaluated. Top-2 SELECTION is
// unaffected (cosine, unit-vector L2 and angle are monotone in one another, so
// the argmax is identical), and a max-distance THRESHOLD converts exactly
// between the spaces. Only the RATIO test genuinely differs, because a ratio of
// angles is not a ratio of distances.
//
//   kL2      - ratio of unit-vector L2 distances. What COLMAP's CPU matcher
//              does (sift.cc, FindBestMatchesOneWayIndex) and what VLFeat's
//              vl_ubcmatch does. Cheapest: d^2 = 2-2*cos is one FMA on the
//              value the GEMM already produced.
//   kAngular - ratio of angles, via acos. Behaviourally what SiftGPU does. Costs
//              a transcendental, but only in the O(n) reduction epilogue, never
//              in the O(n1*n2) inner loop.
//
// LICENSING: kAngular replicates SiftGPU's observable BEHAVIOUR only; no
// SiftGPU code is copied or adapted here. This implementation works from true
// normalised cosines produced by cuBLAS (SiftGPU instead scales raw integer dot
// products by a hardcoded reciprocal), clamps both ends of the domain, and
// evaluates the test in a host-side epilogue over separately-reduced top-2
// arrays. The shared element is the mathematical idea -- compare a ratio of
// angles, and test both directions -- not any expression of it.
//
// NOTE this also changes the UNITS of any max-distance threshold applied by the
// caller: radians under kAngular, unit-vector L2 distance under kL2.
enum GemmDistanceType { kGemmDistanceL2 = 0, kGemmDistanceAngular = 1 };

// GEMM-class matcher (cuBLAS SGEMM + tiled top-2 reductions) for uint8
// descriptors (row-major n x 128). The host->device transfer carries 128 B per
// descriptor and the widening to float runs on the device. Returns kernel time
// in ms; matches are appended to `out`.
//
// ratio: Lowe ratio test, evaluated in the space distanceType selects.
// mutual: cross-check. reverseRatio (only meaningful when `mutual` is set)
// selects how strong the cross-check is:
//   true  - also apply the ratio test in the reverse (column) direction, which
//           is what COLMAP's CPU matcher and SiftGPU both do. Costs nothing
//           extra on the device: the column runner-up falls out of the same
//           reduction that already computes the column best.
//   false - require only mutual-best-index agreement. Weaker on repetitive
//           structure; see ColTop2's comment in matcher_gemm.cu.
// `out[k].dist2` is the squared L2 distance in the INPUT descriptors' own scale
// (so ~512^2-scaled for COLMAP descriptors), regardless of distanceType.
// maxDistance is the absolute threshold on the best match, in the units the
// chosen distanceType implies: RADIANS under kAngular, and descriptor-scale
// distance under kL2 (the same units the input descriptors are in).
// sqDescriptorNorm is the squared norm the descriptors are nominally scaled to;
// kAngular divides the raw inner product by it to form the pseudo-cosine, which
// is how both of COLMAP's references do it (a FIXED constant, not per-row
// norms). Pass 512*512 for COLMAP-scale descriptors.
float MatchDescriptorsGemmU8(const unsigned char* h_desc1, int n1,
                             const unsigned char* h_desc2, int n2,
                             float ratio, bool mutual, std::vector<Match>& out,
                             bool reverseRatio = true,
                             int distanceType = kGemmDistanceAngular,
                             float maxDistance = 3.4028235e38f,
                             float sqDescriptorNorm = 512.0f * 512.0f);

// Sizes the thread-local device buffers up front, for `maxNumDescriptors` per
// side. Optional but worth calling once per matching thread when an upper bound
// is known: the buffers are otherwise grown on demand, and every growth does a
// batch of cudaFree/cudaMalloc. cudaFree synchronises the WHOLE DEVICE, so in a
// pipelined host (COLMAP runs verification and I/O alongside matching) that
// stalls unrelated queued work.
void ReserveGemmMatcher(int maxNumDescriptors);

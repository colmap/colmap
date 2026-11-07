// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yonatan Simson (see ../LICENSE)

#pragma once
#include <memory>
#include <vector>

struct Match { int i, j; float dist2; };

// Distance used to choose and test each descriptor's best match. It changes
// three things, not only the ratio test:
//   - which candidate wins: kL2 ranks on the true distance,
//     |d1|^2 + |d2|^2 - 2 d1.d2, and kAngular on the raw inner product d1.d2,
//     so when candidate norms differ the two can pick different winners
//     (RowTop2Update and ColTop2 rank kL2 on the mirrored key 2 d1.d2 - |c|^2,
//     c the candidate, for that reason);
//   - the space of the ratio test: a ratio of distances under kL2, a ratio of
//     angles acos(d1.d2 / sqDescriptorNorm) under kAngular;
//   - the units of maxDistance: unit-vector L2 distance under kL2, radians
//     under kAngular.
//
//   kL2      - what COLMAP's index-based matcher does (sift.cc,
//              FindBestMatchesOneWayIndex).
//   kAngular - what COLMAP's brute-force matcher and SiftGPU do, with a fixed
//              descriptor norm. The acos runs only in the O(n) epilogue.
//
// No SiftGPU code is copied or adapted here; kAngular reproduces its
// observable behaviour.
enum GemmDistanceType { kGemmDistanceL2 = 0, kGemmDistanceAngular = 1 };

struct GemmMatcherContext;

// GEMM-class matcher (cuBLAS SGEMM + tiled top-2 reductions) for uint8
// descriptors (row-major n x 128). Owns a cuBLAS handle and grow-only device
// buffers, freed when it is destroyed; use one per thread. Not thread_local:
// on Windows every CUDA call made from a thread-exit destructor fails
// ("initialization error"), so buffers released there leaked.
class GemmMatcher {
 public:
  GemmMatcher();
  ~GemmMatcher();
  GemmMatcher(const GemmMatcher&) = delete;
  GemmMatcher& operator=(const GemmMatcher&) = delete;

  // Sizes the device buffers up front, for `maxNumDescriptors` per side.
  // Optional but worth calling once when an upper bound is known: the buffers
  // are otherwise grown on demand, and every growth does a batch of
  // cudaFree/cudaMalloc. cudaFree synchronises the WHOLE DEVICE, so in a
  // pipelined host (COLMAP runs verification and I/O alongside matching) that
  // stalls unrelated queued work.
  void Reserve(int maxNumDescriptors);

  // Matches two descriptor sets (see below for the parameters).
  float MatchPair(const unsigned char* h_desc1, int n1,
                  const unsigned char* h_desc2, int n2,
                  float ratio, bool mutual, std::vector<Match>& out,
                  bool reverseRatio = true,
                  int distanceType = kGemmDistanceAngular,
                  float maxDistance = 3.4028235e38f,
                  float sqDescriptorNorm = 512.0f * 512.0f);

 private:
  GemmMatcherContext& Context();
  std::unique_ptr<GemmMatcherContext> ctx_;
};

// GemmMatcher::MatchPair: The host->device transfer carries 128 B per
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
// chosen distanceType implies: RADIANS under kAngular, and unit-vector L2
// distance under kL2 (the kernel scales it by sqDescriptorNorm).
// sqDescriptorNorm is the squared norm the descriptors are nominally scaled to;
// kAngular divides the raw inner product by it to form the pseudo-cosine, which
// is how both of COLMAP's references do it (a FIXED constant, not per-row
// norms). Pass 512*512 for COLMAP-scale descriptors.

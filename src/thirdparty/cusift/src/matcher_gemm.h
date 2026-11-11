// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yonatan Simson (see ../LICENSE)

#pragma once
#include <memory>
#include <vector>

// Distance used to rank each descriptor's candidates. kL2 ranks on the true
// distance, |d1|^2 + |d2|^2 - 2 d1.d2, and kAngular on the raw inner product
// d1.d2, so when candidate norms differ the two can pick different winners
// (RowTop2Update and ColTop2 rank kL2 on the mirrored key 2 d1.d2 - |c|^2, c
// the candidate, for that reason). The caller's acceptance test then works in
// the matching space: a ratio of distances and a unit-vector max distance
// under kL2, a ratio of angles acos(d1.d2 / 512^2) and a max angle under
// kAngular.
//
//   kL2      - what COLMAP's index-based matcher does (sift.cc,
//              FindBestMatchesOneWayIndex).
//   kAngular - what COLMAP's brute-force matcher and SiftGPU do, with a fixed
//              descriptor norm.
//
// No SiftGPU code is copied or adapted here; kAngular reproduces its
// observable behaviour.
enum GemmDistanceType { kGemmDistanceL2 = 0, kGemmDistanceAngular = 1 };

struct GemmMatcherContext;

// Top-2 candidates per query descriptor, as reduction keys: the raw inner
// product q.c under kAngular, and 2 q.c - |c|^2 under kL2 (so the squared
// distance is sqNorm - key). A missing candidate has key -FLT_MAX, and
// bestIdx is -1 when there is no candidate at all.
struct GemmTop2 {
  std::vector<float> best;
  std::vector<float> second;
  std::vector<int> bestIdx;
  std::vector<float> sqNorm;  // |q|^2 of each query descriptor

  void Resize(int n) {
    best.resize(n);
    second.resize(n);
    bestIdx.resize(n);
    sqNorm.resize(n);
  }
  void Clear() { Resize(0); }
};

// GEMM-class matcher (cuBLAS SGEMM + tiled top-2 reductions) for uint8
// descriptors (row-major n x 128). Owns a cuBLAS handle and grow-only device
// buffers, freed when it is destroyed; use one per thread. Not thread_local:
// on Windows every CUDA call made from a thread-exit destructor fails
// ("initialization error"), so buffers released there leaked.
//
// Default stream only: the cuBLAS handle keeps its default stream and every
// kernel launch and copy uses the default stream too. cusift is built without
// --default-stream per-thread, so that is the legacy default stream: Top2 is
// serialised against all other blocking-stream work on the device, including
// GemmMatchers on other threads, and returns once its results are on the host.
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

  // Best and second-best candidate of every descriptor of side 1 among side 2
  // (`rows`) and, when `columns` is set, of every descriptor of side 2 among
  // side 1 (`cols`). Descriptors are uint8, row-major n x 128. Clears both
  // outputs and fills them (their contents are unspecified if it throws); an
  // empty side launches nothing. A descriptor pointer must be non-null when its
  // side is non-empty. Ranking follows distanceType (see above); accepting a
  // match from the result is up to the caller.
  void Top2(const unsigned char* h_desc1, int n1,
            const unsigned char* h_desc2, int n2, int distanceType,
            bool columns, GemmTop2& rows, GemmTop2& cols);

 private:
  GemmMatcherContext& Context();
  std::unique_ptr<GemmMatcherContext> ctx_;
};

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yonatan Simson (see ../LICENSE)

// matcher_gemm.cu — brute-force matcher built on one cuBLAS SGEMM per tile.
//
// sim = D1 (n1x128) * D2^T (128xn2) via cuBLAS SGEMM, tiled over n2 to bound
// memory. After each tile, two lightweight kernels update running
// best/second-best per query row (for matching) and best per target column
// (for the mutual cross-check). The tile reductions dominate the cost, not the
// GEMM.
#include "matcher_gemm.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>
#include <stdexcept>
#include <string>

#include "cudautils.h"  // safeCall / checkMsg: throw on CUDA errors

namespace {
void CheckCublas(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cusift: ") + what +
                             " failed (cuBLAS status " +
                             std::to_string(static_cast<int>(status)) + ")");
  }
}
}  // namespace

// Widens uint8 descriptors to float in place on the device. The uint8 buffer is
// the actual upload target (4x less PCIe traffic than float32) and is retained
// deliberately: row-major, 128 contiguous bytes per descriptor, so lda = 128.
//
// (An integer GEMM would need a signed-int8 correction or a custom kernel:
// cuBLAS has no unsigned 8-bit GEMM and descriptor values reach 255.)
__global__ void WidenU8(const unsigned char* __restrict__ src, int n, int dim,
                        float* __restrict__ dst) {
  const size_t total = (size_t)n * dim;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total;
       i += (size_t)gridDim.x * blockDim.x) {
    dst[i] = (float)src[i];
  }
}

// Per-descriptor squared norm, needed to rank by true distance rather than by
// raw inner product (see the reduction-key note on RowTop2Update).
__global__ void SqNorms(const float* __restrict__ d, int n, int dim,
                        float* __restrict__ out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const float* row = d + (size_t)i * dim;
    float acc = 0.0f;
    for (int k = 0; k < dim; k++) acc += row[k] * row[k];
    out[i] = acc;
  }
}

// Row-wise: for each query row, scan a (n1 x tw) tile and merge into running
// (best, second, idx). One thread per query, grid-stride.
//
// REDUCTION KEY (L2 mode; ANGULAR ranks on the raw inner product, as the
// reference does). Ranking by the raw inner product only agrees with ranking by
// distance when every candidate has the same norm. COLMAP's descriptors sit
// NEAR its 512 convention but not exactly on it, and
//   ||q - c||^2 = ||q||^2 + ||c||^2 - 2 q.c
// so for a fixed query the distance-minimising candidate maximises
//   key = 2 q.c - ||c||^2
// which is what this kernel reduces on. Reducing on q.c instead silently
// prefers longer candidates and disagrees with an exact brute-force matcher on
// a thin band of near-ties.
// The caller recovers the distance as ||q||^2 - key.
//
// Note this is exact in fp32 for COLMAP-scale descriptors: with ||d||^2 ~ 2^18
// every intermediate stays below 2^24, so no mantissa bits are lost. This
// assumes cuBLAS's default math mode. SGEMM then computes and accumulates in
// FP32. A reduced-precision mode could round the keys. The differential fuzz
// tests in cusift_matcher_test guard this. SqDistanceFromGemmKey clamps a
// recovered distance at 0.
//
// CANONICAL TIE-BREAK: strictly better key, or an equal key at a LOWER index.
// The column reduction merges in a shared-memory tree, so without this rule a
// tie would resolve by thread layout; keys are integers of magnitude at most
// about 2^18, so exact ties do occur, and the cross-check compares column
// winners exactly.
__device__ __forceinline__ bool CanonicallyBetter(float v, int i, float b,
                                                  int bi) {
  return v > b || (v == b && (bi < 0 || i < bi));
}

__global__ void RowTop2Update(const float* __restrict__ sim, int n1, int tw,
                              int tileOff, const float* __restrict__ colSqNorm,
                              float* __restrict__ best,
                              float* __restrict__ second, int* __restrict__ idx,
                              int* __restrict__ idx2) {
  for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < n1;
       q += gridDim.x * blockDim.x) {
    float b = best[q], s = second[q];
    int bi = idx[q], si = idx2[q];
    const float* row = sim + (size_t)q * tw;
    for (int t = 0; t < tw; t++) {
      // colSqNorm == nullptr selects raw-inner-product ranking, which is what
      // COLMAP's exact brute-force reference and SiftGPU both do. Non-null
      // selects distance ranking (COLMAP's faiss/index path). See matcher_gemm.h.
      const float v = colSqNorm ? (2.0f * row[t] - colSqNorm[tileOff + t])
                                : row[t];
      const int t_glob = tileOff + t;
      if (CanonicallyBetter(v, t_glob, b, bi)) {
        s = b; si = bi; b = v; bi = t_glob;
      } else if (CanonicallyBetter(v, t_glob, s, si)) {
        s = v; si = t_glob;
      }
    }
    best[q] = b; second[q] = s; idx[q] = bi; idx2[q] = si;
  }
}

// Column-wise: best AND second-best query for each target column in the tile.
// One block per column; block-stride rows; shared-mem reduction.
//
// Second-best is tracked (not just the best) because
// the cross-check has to apply the ratio test in the REVERSE direction too, not
// merely confirm mutual-best. Both of COLMAP's own matchers behave that way:
// its CPU path runs FindBestMatchesOneWayIndex once per direction before
// intersecting (colmap/feature/sift.cc, FindBestMatchesIndex), and SiftGPU
// applies its distance and ratio thresholds in the column direction as well as
// the row direction. Mutual-best alone is materially weaker: on repetitive
// structure it admits matches whose reverse-direction runner-up is nearly as
// close, which is exactly the spurious-match regime the ratio test suppresses.
//
// The reduction below is an independent implementation -- a standard top-2
// merge -- written from the behavioural requirement, not ported from either
// reference.
//
// Each column lives in exactly one tile, so a column's top-2 is complete within
// its own tile -- no cross-tile merge is needed here (unlike the row direction).
__global__ void ColTop2(const float* __restrict__ sim, int n1, int tw,
                        const float* __restrict__ rowSqNorm,
                        float* __restrict__ colBest,
                        float* __restrict__ colSecond,
                        int* __restrict__ colIdx) {
  int c = blockIdx.x;
  if (c >= tw) return;
  float b = -FLT_MAX, s = -FLT_MAX; int bi = -1;
  for (int r = threadIdx.x; r < n1; r += blockDim.x) {
    // Mirror of the row key: for a fixed column the distance-minimising row
    // maximises 2*sim - ||row||^2. The caller recovers the distance as
    // ||col||^2 - key.
    const float v = rowSqNorm ? (2.0f * sim[(size_t)r * tw + c] - rowSqNorm[r])
                              : sim[(size_t)r * tw + c];
    // Only the BEST needs an index-aware comparison. The runner-up's index is
    // never used (callers read colSecond's value only), and max over
    // values is order-independent, so the second is canonical by construction.
    if (CanonicallyBetter(v, r, b, bi)) { s = b; b = v; bi = r; }
    else if (v > s) { s = v; }
  }
  __shared__ float sb[256]; __shared__ float ss[256]; __shared__ int si[256];
  sb[threadIdx.x] = b; ss[threadIdx.x] = s; si[threadIdx.x] = bi;
  __syncthreads();
  // Standard top-2 merge: the winner's best survives, and the second is the
  // larger of (loser's best, both seconds). With the canonical comparison the
  // result does not depend on which thread happened to scan which rows: ties
  // go to the lowest row index, not the lowest threadIdx.x.
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (threadIdx.x < off) {
      const float ob = sb[threadIdx.x + off], os = ss[threadIdx.x + off];
      const int oi = si[threadIdx.x + off];
      const float mb = sb[threadIdx.x], ms = ss[threadIdx.x];
      const int mi = si[threadIdx.x];
      if (CanonicallyBetter(ob, oi, mb, mi)) {
        sb[threadIdx.x] = ob;
        si[threadIdx.x] = oi;
        ss[threadIdx.x] = fmaxf(mb, fmaxf(ms, os));
      } else {
        ss[threadIdx.x] = fmaxf(ms, fmaxf(ob, os));
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    colBest[c] = sb[0]; colSecond[c] = ss[0]; colIdx[c] = si[0];
  }
}

__global__ void InitTop2(int n, float* best, float* second, int* idx,
                        int* idx2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    best[i] = -FLT_MAX; second[i] = -FLT_MAX; idx[i] = -1; idx2[i] = -1;
  }
}

// A per-pair `cublasCreate` and device-synchronizing `cudaMalloc`/`cudaFree`
// would dominate the cost of a pair loop, so everything is hoisted into one
// grow-only context, created on first use and
// reused for every later pair. Buffers only ever grow, so a batch settles after
// its largest pair. Owned by one GemmMatcher: one cuBLAS handle per matcher,
// since cuBLAS handles are not safe to share across threads.
struct GemmMatcherContext {
  cublasHandle_t blas = nullptr;
  float *d1 = nullptr, *d2 = nullptr, *dSim = nullptr;
  float *dBest = nullptr, *dSecond = nullptr, *dColBest = nullptr;
  float *dColSecond = nullptr;
  float *d1Sq = nullptr, *d2Sq = nullptr;
  unsigned char *d1U8 = nullptr, *d2U8 = nullptr;
  int *dIdx = nullptr, *dIdx2 = nullptr, *dColIdx = nullptr;
  int cap1 = 0, cap2 = 0, capTile = 0;
  int device = -1;  // where the handle and buffers live

  GemmMatcherContext() = default;
  GemmMatcherContext(const GemmMatcherContext&) = delete;
  GemmMatcherContext& operator=(const GemmMatcherContext&) = delete;

  // Runs when the owning GemmMatcher is destroyed. Errors are ignored: there
  // is nothing to do about a failed free in a destructor.
  ~GemmMatcherContext() {
    ScopedCudaDeviceNoThrow scoped(device);
    ReleaseAll();
  }

  void ReleaseAll() {
    Release(&d1); Release(&d2); Release(&dSim);
    Release(&dBest); Release(&dSecond); Release(&dColBest);
    Release(&dColSecond); Release(&d1Sq); Release(&d2Sq);
    Release(&d1U8); Release(&d2U8);
    Release(&dIdx); Release(&dIdx2); Release(&dColIdx);
    if (blas) { cublasDestroy(blas); blas = nullptr; }
    cap1 = cap2 = capTile = 0;
  }

  void Reserve(int n1, int n2, int tile) {
    // A capacity is updated only after all of its buffers were allocated, so
    // a failed allocation (which throws) never leaves a buffer smaller than
    // its recorded capacity; the next call retries.
    if (!blas) CheckCublas(cublasCreate(&blas), "cublasCreate");
    if (n1 > cap1) {
      Release(&d1); Release(&dBest); Release(&dSecond); Release(&dIdx);
      Release(&dIdx2); Release(&d1Sq); Release(&d1U8);
      Release(&dSim);   // dSim is cap1 x capTile -- both bounds matter
      cap1 = 0;
      safeCall(cudaMalloc(&d1U8, (size_t)n1 * 128 * sizeof(unsigned char)));
      safeCall(cudaMalloc(&d1, (size_t)n1 * 128 * sizeof(float)));
      safeCall(cudaMalloc(&dBest, (size_t)n1 * sizeof(float)));
      safeCall(cudaMalloc(&dSecond, (size_t)n1 * sizeof(float)));
      safeCall(cudaMalloc(&dIdx, (size_t)n1 * sizeof(int)));
      safeCall(cudaMalloc(&dIdx2, (size_t)n1 * sizeof(int)));
      safeCall(cudaMalloc(&d1Sq, (size_t)n1 * sizeof(float)));
      cap1 = n1;
    }
    if (n2 > cap2) {
      Release(&d2); Release(&dColBest); Release(&dColSecond);
      Release(&dColIdx); Release(&d2Sq); Release(&d2U8);
      cap2 = 0;
      safeCall(cudaMalloc(&d2U8, (size_t)n2 * 128 * sizeof(unsigned char)));
      safeCall(cudaMalloc(&d2, (size_t)n2 * 128 * sizeof(float)));
      safeCall(cudaMalloc(&d2Sq, (size_t)n2 * sizeof(float)));
      safeCall(cudaMalloc(&dColBest, (size_t)n2 * sizeof(float)));
      safeCall(cudaMalloc(&dColSecond, (size_t)n2 * sizeof(float)));
      safeCall(cudaMalloc(&dColIdx, (size_t)n2 * sizeof(int)));
      cap2 = n2;
    }
    if (tile > capTile) { Release(&dSim); capTile = tile; }
    if (!dSim) safeCall(cudaMalloc(&dSim, (size_t)cap1 * capTile * sizeof(float)));
  }

  template <typename T>
  static void Release(T** p) {
    if (*p) { cudaFree(*p); *p = nullptr; }
  }
};

namespace {

template <typename T>
void Download(std::vector<T>& dst, const T* src, int n) {
  safeCall(cudaMemcpy(dst.data(), src, (size_t)n * sizeof(T),
                      cudaMemcpyDeviceToHost));
}

// Everything after the upload: the reductions, then the download of their
// results.
void ReduceUploaded(GemmMatcherContext& ctx, int n1, int n2, int distanceType,
                    bool columns, GemmTop2& rows, GemmTop2& cols) {
  const bool angular_mode = (distanceType == kGemmDistanceAngular);
  const int TILE = 512;
  float *d1 = ctx.d1, *d2 = ctx.d2, *dSim = ctx.dSim;
  float *dBest = ctx.dBest, *dSecond = ctx.dSecond, *dColBest = ctx.dColBest;
  float *dColSecond = ctx.dColSecond;
  float *d1Sq = ctx.d1Sq, *d2Sq = ctx.d2Sq;
  int *dIdx = ctx.dIdx, *dIdx2 = ctx.dIdx2, *dColIdx = ctx.dColIdx;

  cublasHandle_t hb = ctx.blas;
  const float one = 1.f, zero = 0.f;

  InitTop2<<<(n1 + 255) / 256, 256>>>(n1, dBest, dSecond, dIdx, dIdx2);
  // Squared norms drive the distance-ranked reduction key (see RowTop2Update).
  // O(n*128) and computed once per call, negligible beside the GEMM.
  SqNorms<<<(n1 + 255) / 256, 256>>>(d1, n1, 128, d1Sq);
  SqNorms<<<(n2 + 255) / 256, 256>>>(d2, n2, 128, d2Sq);
  checkMsg("InitTop2/SqNorms launch failed");

  for (int off = 0; off < n2; off += TILE) {
    int tw = (n2 - off < TILE) ? (n2 - off) : TILE;
    // Row-major view: sim[q][t] = dot(D1[q], D2[off+t]).
    // cuBLAS is column-major: compute simCM (tw x n1) = D2tile * D1^T via
    //   C = A^T * B with A = D2tile (128 x tw CM == row-major tw x 128),
    //   B = D1 (128 x n1 CM == row-major n1 x 128)
    // then simCM(t, q) laid out col-major == row-major sim[q][t]. 
    CheckCublas(cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, tw, n1, 128, &one,
                            d2 + (size_t)off * 128, 128, d1, 128, &zero, dSim,
                            tw),
                "cublasSgemm");
    RowTop2Update<<<256, 256>>>(dSim, n1, tw, off,
                                angular_mode ? nullptr : d2Sq, dBest, dSecond,
                                dIdx, dIdx2);
    // Second-best per column comes from the same reduction pass as the best --
    // one extra shared array and a few instructions, no extra kernel launch and
    // no extra pass over dSim -- so the reverse ratio test is essentially free
    // once the mutual check is already being computed.
    if (columns)
      ColTop2<<<tw, 256>>>(dSim, n1, tw, angular_mode ? nullptr : d1Sq,
                           dColBest + off, dColSecond + off, dColIdx + off);
    checkMsg("RowTop2Update/ColTop2 launch failed");
  }
  // The blocking downloads below wait for the kernels and report their errors.
  rows.Resize(n1);
  cols.Resize(columns ? n2 : 0);
  Download(rows.best, dBest, n1);
  Download(rows.second, dSecond, n1);
  Download(rows.bestIdx, dIdx, n1);
  Download(rows.sqNorm, d1Sq, n1);
  if (columns) {
    Download(cols.best, dColBest, n2);
    Download(cols.second, dColSecond, n2);
    Download(cols.bestIdx, dColIdx, n2);
    Download(cols.sqNorm, d2Sq, n2);
  }

  // Nothing is freed here: the context owns the buffers and the handle for the
  // life of the matcher. A batch of pairs settles on the largest pair.
}

}  // namespace

GemmMatcher::GemmMatcher() : ctx_(new GemmMatcherContext) {}

GemmMatcher::~GemmMatcher() = default;

// The context on the calling thread's current device: one left on another
// device by an earlier call is released there first.
GemmMatcherContext& GemmMatcher::Context() {
  GemmMatcherContext& ctx = *ctx_;
  const int device = cusiftCurrentDevice();
  if (ctx.device != device) {
    if (ctx.device >= 0) {
      ScopedCudaDevice previous(ctx.device);
      ctx.ReleaseAll();
    }
    ctx.device = device;
  }
  return ctx;
}

void GemmMatcher::Reserve(int maxNumDescriptors) {
  if (maxNumDescriptors > 0) {
    Context().Reserve(maxNumDescriptors, maxNumDescriptors, 512);
  }
}

void GemmMatcher::Top2(const unsigned char* h_desc1, int n1,
                       const unsigned char* h_desc2, int n2, int distanceType,
                       bool columns, GemmTop2& rows, GemmTop2& cols) {
  rows.Clear();
  cols.Clear();
  // A side without descriptors has no candidates; a zero-sized grid would fail.
  if (n1 <= 0 || n2 <= 0) return;
  if (h_desc1 == nullptr || h_desc2 == nullptr) {
    throw std::invalid_argument("cusift: GemmMatcher::Top2 got a null "
                                "descriptor pointer for a non-empty side");
  }
  GemmMatcherContext& ctx = Context();
  ctx.Reserve(n1, n2, 512);
  // 128 B per descriptor instead of 512 B. The widening then happens on the
  // device, where it is memory-bound noise beside the GEMM.
  safeCall(cudaMemcpy(ctx.d1U8, h_desc1, (size_t)n1 * 128, cudaMemcpyHostToDevice));
  safeCall(cudaMemcpy(ctx.d2U8, h_desc2, (size_t)n2 * 128, cudaMemcpyHostToDevice));
  WidenU8<<<256, 256>>>(ctx.d1U8, n1, 128, ctx.d1);
  WidenU8<<<256, 256>>>(ctx.d2U8, n2, 128, ctx.d2);
  checkMsg("WidenU8 launch failed");
  ReduceUploaded(ctx, n1, n2, distanceType, columns, rows, cols);
}

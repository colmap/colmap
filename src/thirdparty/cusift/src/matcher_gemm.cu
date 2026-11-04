// matcher_gemm.cu — GEMM-class matcher (SiftMatchGPU speed class). MIT.
//
// sim = D1 (n1x128) * D2^T (128xn2) via cuBLAS SGEMM, tiled over n2 to bound
// memory. After each tile, two lightweight kernels update running
// best/second-best per query row (for matching) and best per target column
// (for the mutual cross-check). FP32 SGEMM at n=10k, k=128 is ~26 GFLOP:
// sub-millisecond on Ada — the tile reductions dominate, not the GEMM.
#include "matcher.h"

// Implemented in matcher_fused.cu. Shares this file's context buffers and
// epilogue so that only the PRODUCER differs between the two backends --
// which is what makes fused-vs-GEMM a meaningful bit-exactness oracle.
float MatchDescriptorsFusedRaw(const float* d1, int n1, const float* d2,
                               int n2, const float* d1Sq, const float* d2Sq,
                               bool useL2Key, float* rowBest,
                               float* rowSecond, int* rowIdx, int* rowIdx2,
                               float* colBest, float* colSecond,
                               int* colIdx, float* colPartBest,
                               float* colPartSecond, int* colPartIdx,
                               bool wantColumns);
int MatchDescriptorsFusedRowBlock();

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>

// Widens uint8 descriptors to float in place on the device. The uint8 buffer is
// the actual upload target (4x less PCIe traffic than float32) and is retained
// deliberately: row-major, 128 contiguous bytes per descriptor, so lda = 128.
//
// ON AN INTEGER GEMM FOLLOW-UP. Tempting, because the reference ranks on integer
// dot products anyway, so int32 accumulation is what the fp32 path is imitating
// rather than an approximation of it. But cuBLAS has NO unsigned 8-bit GEMM:
// CUDA_R_8I is SIGNED, and these descriptors reach 255. So it is not a
// mechanical swap. The options, none free:
//   1. Upload d-128 as int8 and correct exactly:
//        a.b = (a-128).(b-128) + 128*sum(a) + 128*sum(b) - 128^3
//      For ranking within one query row the row terms are constant and drop
//      out, leaving a per-column term -- the same shape as the colSqNorm key
//      this file already plumbs -- but the REPORTED distance still needs the
//      full correction, and both sums must be precomputed per descriptor set.
//   2. A hand-rolled __dp4a kernel (it does have an unsigned form). Exact, but
//      that means writing our own GEMM, which is precisely what this design
//      avoided by delegating to cuBLAS.
//   3. CUTLASS, which supports uint8 operands natively -- exact and fast, at the
//      cost of a heavyweight new dependency for one kernel.
// Decide it by measurement, not by appeal: if the SGEMM is not a meaningful
// share of per-pair time on the target GPU, option 1's epilogue complexity buys
// little. The bit-exactness gate (Jaccard 1.0000 against COLMAP's brute-force
// reference) makes that experiment cheap to judge.
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
// REDUCTION KEY. Ranking by the raw inner product only agrees with ranking by
// distance when every candidate has the same norm. COLMAP's descriptors sit
// NEAR its 512 convention but not exactly on it, and
//   ||q - c||^2 = ||q||^2 + ||c||^2 - 2 q.c
// so for a fixed query the distance-minimising candidate maximises
//   key = 2 q.c - ||c||^2
// which is what this kernel reduces on. Reducing on q.c instead silently
// prefers longer candidates and disagrees with an exact brute-force matcher on
// a thin band of near-ties (measured: ~1% of matches on real GoPro imagery).
// The caller recovers the distance as ||q||^2 - key.
//
// Note this is exact in fp32 for COLMAP-scale descriptors: with ||d||^2 ~ 2^18
// every intermediate stays below 2^24, so no mantissa bits are lost.
// CANONICAL TIE-BREAK: strictly better key, or an equal key at a LOWER index.
//
// Plain `v > b` makes the winner depend on the order candidates happen to be
// visited, which is only well-defined for a sequential ascending scan. The row
// reduction below does scan ascending, so it already resolves ties to the lowest
// index -- but by accident of decomposition, not by rule. The column reduction
// does NOT: it block-strides rows and merges in a shared tree, so it resolves to
// the lowest threadIdx.x, i.e. the lowest r % blockDim.x rather than the lowest
// r. That is invisible on data where exact key ties do not occur, which is why it
// has never shown up, but `hColIdx[j]` is compared for equality against `i` in
// the epilogue, so a tie there does change which pairs survive the cross-check.
//
// Making the rule explicit is a prerequisite for any parallel re-decomposition
// (e.g. a fused kernel's two-level reduction, where partials merge in no
// particular order): the answer must be a property of the data, not of the
// thread layout. Keys are integer-valued in [0, 262144] here, so exact ties are
// reachable and will eventually occur on some dataset.
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
      // selects distance ranking (COLMAP's faiss/index path). See matcher.h.
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
// Second-best is tracked (not just the best, as an earlier revision did) because
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
// reference. See the LICENSING note in matcher.h.
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
    // never used (the epilogue reads colSecond's value only), and max over
    // values is order-independent, so the second is canonical by construction.
    if (CanonicallyBetter(v, r, b, bi)) { s = b; b = v; bi = r; }
    else if (v > s) { s = v; }
  }
  __shared__ float sb[256]; __shared__ float ss[256]; __shared__ int si[256];
  sb[threadIdx.x] = b; ss[threadIdx.x] = s; si[threadIdx.x] = bi;
  __syncthreads();
  // Standard top-2 merge: the winner's best survives, and the second is the
  // larger of (loser's best, both seconds). With the canonical comparison the
  // result no longer depends on which thread happened to scan which rows --
  // previously this resolved ties to the lowest threadIdx.x, i.e. the lowest
  // r % blockDim.x rather than the lowest r.
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

// Tier 1.1, matcher half: the per-pair `cublasCreate` (ms-scale
// on its own) and 8 `cudaMalloc`/`cudaFree` (device-synchronizing) were most of
// the gap between the warm kernel time and what a pair loop actually costs.
// Everything is hoisted into one grow-only context, created on first use and
// reused for every later pair. Buffers only ever grow, so a batch settles after
// its largest pair. thread_local: one context (and one cuBLAS handle) per
// matching worker, since cuBLAS handles are not safe to share across threads.
namespace {

struct GemmMatcherContext {
  cublasHandle_t blas = nullptr;
  float *d1 = nullptr, *d2 = nullptr, *dSim = nullptr;
  float *dBest = nullptr, *dSecond = nullptr, *dColBest = nullptr;
  float *dColSecond = nullptr;
  float *d1Sq = nullptr, *d2Sq = nullptr;
  unsigned char *d1U8 = nullptr, *d2U8 = nullptr;
  // Fused-path per-block column partials, sized cap2 * ceil(cap1/BM).
  float *dColPartBest = nullptr, *dColPartSecond = nullptr;
  int *dColPartIdx = nullptr;
  int capPartBlocks = 0;
  int *dIdx = nullptr, *dIdx2 = nullptr, *dColIdx = nullptr;
  int cap1 = 0, cap2 = 0, capTile = 0;

  void Reserve(int n1, int n2, int tile) {
    if (!blas) cublasCreate(&blas);
    if (n1 > cap1) {
      Release(&d1); Release(&dBest); Release(&dSecond); Release(&dIdx);
      Release(&dIdx2); Release(&d1Sq); Release(&d1U8);
      cap1 = n1;
      cudaMalloc(&d1U8, (size_t)cap1 * 128 * sizeof(unsigned char));
      cudaMalloc(&d1, (size_t)cap1 * 128 * sizeof(float));
      cudaMalloc(&dBest, (size_t)cap1 * sizeof(float));
      cudaMalloc(&dSecond, (size_t)cap1 * sizeof(float));
      cudaMalloc(&dIdx, (size_t)cap1 * sizeof(int));
      cudaMalloc(&dIdx2, (size_t)cap1 * sizeof(int));
      cudaMalloc(&d1Sq, (size_t)cap1 * sizeof(float));
      Release(&dSim);   // dSim is cap1 x capTile -- both bounds matter
    }
    if (n2 > cap2) {
      Release(&d2); Release(&dColBest); Release(&dColSecond);
      Release(&dColIdx); Release(&d2Sq); Release(&d2U8);
      cap2 = n2;
      cudaMalloc(&d2U8, (size_t)cap2 * 128 * sizeof(unsigned char));
      cudaMalloc(&d2, (size_t)cap2 * 128 * sizeof(float));
      cudaMalloc(&d2Sq, (size_t)cap2 * sizeof(float));
      cudaMalloc(&dColBest, (size_t)cap2 * sizeof(float));
      cudaMalloc(&dColSecond, (size_t)cap2 * sizeof(float));
      cudaMalloc(&dColIdx, (size_t)cap2 * sizeof(int));
    }
    // Column partials depend on BOTH bounds, so resize when either grows.
    const int wantBlocks =
        (cap1 + MatchDescriptorsFusedRowBlock() - 1) /
        MatchDescriptorsFusedRowBlock();
    if (wantBlocks > capPartBlocks || !dColPartBest) {
      Release(&dColPartBest); Release(&dColPartSecond);
      Release(&dColPartIdx);
      capPartBlocks = wantBlocks;
    }
    if (!dColPartBest) {
      const size_t np = (size_t)cap2 * capPartBlocks;
      cudaMalloc(&dColPartBest, np * sizeof(float));
      cudaMalloc(&dColPartSecond, np * sizeof(float));
      cudaMalloc(&dColPartIdx, np * sizeof(int));
    }
    if (tile > capTile) { Release(&dSim); capTile = tile; }
    if (!dSim) cudaMalloc(&dSim, (size_t)cap1 * capTile * sizeof(float));
  }

  template <typename T>
  static void Release(T** p) {
    if (*p) { cudaFree(*p); *p = nullptr; }
  }
};

GemmMatcherContext& MatcherCtx() {
  static thread_local GemmMatcherContext ctx;
  return ctx;
}

}  // namespace

namespace {

// Everything after the upload. Both entry points funnel through here so the
// float and uint8 paths cannot drift in their matching semantics.
float MatchUploaded(int n1, int n2, float ratio, bool mutual,
                    std::vector<Match>& out, bool reverseRatio,
                    int distanceType, float maxDistance,
                    float sqDescriptorNorm, bool fused) {
  const bool angular_mode = (distanceType == kGemmDistanceAngular);
  const int TILE = 512;
  GemmMatcherContext& ctx = MatcherCtx();
  float *d1 = ctx.d1, *d2 = ctx.d2, *dSim = ctx.dSim;
  float *dBest = ctx.dBest, *dSecond = ctx.dSecond, *dColBest = ctx.dColBest;
  float *dColSecond = ctx.dColSecond;
  float *d1Sq = ctx.d1Sq, *d2Sq = ctx.d2Sq;
  int *dIdx = ctx.dIdx, *dIdx2 = ctx.dIdx2, *dColIdx = ctx.dColIdx;

  cublasHandle_t hb = ctx.blas;
  const float one = 1.f, zero = 0.f;

  cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);
  InitTop2<<<(n1 + 255) / 256, 256>>>(n1, dBest, dSecond, dIdx, dIdx2);
  // Squared norms drive the distance-ranked reduction key (see RowTop2Update).
  // O(n*128) and computed once per call, negligible beside the GEMM.
  SqNorms<<<(n1 + 255) / 256, 256>>>(d1, n1, 128, d1Sq);
  SqNorms<<<(n2 + 255) / 256, 256>>>(d2, n2, 128, d2Sq);

  if (fused) {
    // Same squared norms, same key selection, same output arrays -- only the
    // producer differs. The similarity matrix is never materialised.
    MatchDescriptorsFusedRaw(d1, n1, d2, n2, d1Sq, d2Sq, !angular_mode,
                             dBest, dSecond, dIdx, dIdx2, dColBest,
                             dColSecond, dColIdx, ctx.dColPartBest,
                             ctx.dColPartSecond, ctx.dColPartIdx, mutual);
  } else
  for (int off = 0; off < n2; off += TILE) {
    int tw = (n2 - off < TILE) ? (n2 - off) : TILE;
    // Row-major view: sim[q][t] = dot(D1[q], D2[off+t]).
    // cuBLAS is column-major: compute simCM (tw x n1) = D2tile * D1^T via
    //   C = A^T * B with A = D2tile (128 x tw CM == row-major tw x 128),
    //   B = D1 (128 x n1 CM == row-major n1 x 128)
    // then simCM(t, q) laid out col-major == row-major sim[q][t]. 
    cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, tw, n1, 128, &one,
                d2 + (size_t)off * 128, 128, d1, 128, &zero, dSim, tw);
    RowTop2Update<<<256, 256>>>(dSim, n1, tw, off,
                                angular_mode ? nullptr : d2Sq, dBest, dSecond,
                                dIdx, dIdx2);
    // Second-best per column comes from the same reduction pass as the best --
    // one extra shared array and a few instructions, no extra kernel launch and
    // no extra pass over dSim -- so the reverse ratio test is essentially free
    // once the mutual check is already being computed.
    if (mutual)
      ColTop2<<<tw, 256>>>(dSim, n1, tw, angular_mode ? nullptr : d1Sq,
                           dColBest + off, dColSecond + off, dColIdx + off);
  }
  cudaEventRecord(t1); cudaEventSynchronize(t1);
  float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0); cudaEventDestroy(t1);

  std::vector<float> hB(n1), hS(n1), h1Sq(n1), h2Sq(n2);
  std::vector<int> hIdx(n1), hIdx2(n1), hColIdx(mutual ? n2 : 0);
  std::vector<float> hColB(mutual && reverseRatio ? n2 : 0);
  std::vector<float> hColS(mutual && reverseRatio ? n2 : 0);
  cudaMemcpy(hB.data(), dBest, n1 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hS.data(), dSecond, n1 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hIdx.data(), dIdx, n1 * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(hIdx2.data(), dIdx2, n1 * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h1Sq.data(), d1Sq, n1 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h2Sq.data(), d2Sq, n2 * sizeof(float), cudaMemcpyDeviceToHost);
  if (mutual) {
    cudaMemcpy(hColIdx.data(), dColIdx, n2 * sizeof(int), cudaMemcpyDeviceToHost);
    if (reverseRatio) {
      cudaMemcpy(hColB.data(), dColBest, n2 * sizeof(float),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(hColS.data(), dColSecond, n2 * sizeof(float),
                 cudaMemcpyDeviceToHost);
    }
  }

  out.clear();
  const float r2 = ratio * ratio;
  const bool angular = angular_mode;
  const float invSqNorm = 1.0f / sqDescriptorNorm;

  // ANGULAR reproduces COLMAP's exact brute-force reference and SiftGPU
  // bit-for-bit in form: rank on the raw inner product, then convert with a
  // FIXED descriptor-norm constant (not per-row norms) and threshold on angles.
  //   angle = acos(min(dot / sqDescriptorNorm, 1))
  // L2 reproduces COLMAP's other convention (its faiss/index path): rank on
  // true distance and compare a ratio of distances.
  auto angle_of = [&](float dot) {
    return acosf(fminf(fmaxf(dot * invSqNorm, -1.0f), 1.0f));
  };

  for (int i = 0; i < n1; i++) {
    const int j = hIdx[i], j2 = hIdx2[i];
    if (j < 0) continue;

    float dB;  // reported squared distance, COLMAP scale
    if (angular) {
      const float aB = angle_of(hB[i]);
      // Threshold on the angle, matching the reference's ordering of tests.
      if (aB > maxDistance) continue;
      if (j2 >= 0) {
        const float aS = angle_of(hS[i]);
        if (aB >= ratio * aS) continue;
      }
      dB = h1Sq[i] + h2Sq[j] - 2.0f * hB[i];
    } else {
      dB = h1Sq[i] - hB[i];
      const float dS = (j2 < 0) ? FLT_MAX : (h1Sq[i] - hS[i]);
      if (dB > maxDistance * maxDistance * sqDescriptorNorm) continue;
      if (j2 >= 0 && dB >= r2 * dS) continue;
    }

    if (mutual) {
      if (hColIdx[j] != i) continue;
      // Reverse-direction test. COLMAP's reference applies the SAME one-way
      // test to the transposed matrix and intersects, so the reverse side gets
      // its own ratio test, not just mutual-best agreement (see ColTop2).
      if (reverseRatio) {
        if (angular) {
          const float cB = angle_of(hColB[j]);
          const float cS = angle_of(hColS[j]);
          if (cB > maxDistance) continue;
          if (cB >= ratio * cS) continue;
        } else {
          const float cB = h2Sq[j] - hColB[j];
          const float cS = h2Sq[j] - hColS[j];
          if (cB >= r2 * cS) continue;
        }
      }
    }
    out.push_back({i, j, dB});
  }

  // Nothing is freed here: the context owns the buffers and the handle for the
  // life of the thread (Tier 1.1). A batch of pairs settles on the largest pair.
  return ms;
}

}  // namespace

void ReserveGemmMatcher(int maxNumDescriptors) {
  if (maxNumDescriptors > 0) {
    MatcherCtx().Reserve(maxNumDescriptors, maxNumDescriptors, 512);
  }
}

float MatchDescriptorsGemm(const float* h_desc1, int n1, const float* h_desc2,
                           int n2, float ratio, bool mutual,
                           std::vector<Match>& out, bool reverseRatio,
                           int distanceType, float maxDistance,
                           float sqDescriptorNorm) {
  GemmMatcherContext& ctx = MatcherCtx();
  ctx.Reserve(n1, n2, 512);
  cudaMemcpy(ctx.d1, h_desc1, (size_t)n1 * 128 * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(ctx.d2, h_desc2, (size_t)n2 * 128 * sizeof(float),
             cudaMemcpyHostToDevice);
  return MatchUploaded(n1, n2, ratio, mutual, out, reverseRatio, distanceType,
                       maxDistance, sqDescriptorNorm, /*fused=*/false);
}

float MatchDescriptorsGemmU8(const unsigned char* h_desc1, int n1,
                             const unsigned char* h_desc2, int n2, float ratio,
                             bool mutual, std::vector<Match>& out,
                             bool reverseRatio, int distanceType,
                             float maxDistance, float sqDescriptorNorm,
                             bool fused) {
  GemmMatcherContext& ctx = MatcherCtx();
  ctx.Reserve(n1, n2, 512);
  // 128 B per descriptor instead of 512 B. The widening then happens on the
  // device, where it is memory-bound noise beside the GEMM.
  cudaMemcpy(ctx.d1U8, h_desc1, (size_t)n1 * 128, cudaMemcpyHostToDevice);
  cudaMemcpy(ctx.d2U8, h_desc2, (size_t)n2 * 128, cudaMemcpyHostToDevice);
  WidenU8<<<256, 256>>>(ctx.d1U8, n1, 128, ctx.d1);
  WidenU8<<<256, 256>>>(ctx.d2U8, n2, 128, ctx.d2);
  return MatchUploaded(n1, n2, ratio, mutual, out, reverseRatio, distanceType,
                       maxDistance, sqDescriptorNorm, fused);
}

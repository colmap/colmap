// matcher_fused.cu - fused single-pass bidirectional matcher. MIT.
//
// WHY THIS EXISTS. The GEMM matcher in matcher_gemm.cu is bit-exact but
// bandwidth-bound: cuBLAS writes the similarity tile to global memory,
// RowTop2Update reads all of it, and ColTop2 reads all of it again. That is
// 3 * n1 * n2 * 4 B = 1200 MB per pair at n=10k, or ~4.41 ms of the measured
// 11.0 ms on an RTX 4060 (272 GB/s). The GEMM arithmetic itself is only ~1.7 ms.
// Two independent confirmations of that figure: the arithmetic above, and the
// measured 4.49 ms gap between our 11.00 ms and CudaSift's fused one-directional
// kernel at 6.51 ms.
//
// So this keeps every similarity value in registers/shared and never writes the
// matrix. Descriptor traffic is O(n*128) and the only O(n * blocks) output is the
// per-column partials, ~75 MB of traffic at BM=32 instead of 1200 MB.
//
// SINGLE PASS, BOTH DIRECTIONS. This is the part that does not exist upstream.
// CudaSift's FindMaxCorr10 writes only sift1[], so a cross-check needs a second
// full pass -- measured at 14.19 ms, which is why the GEMM design won before.
// Here one block owns BM queries and sweeps ALL targets, so:
//   - the ROW top-2 is complete within the block and is written straight out;
//     no row partials and no row merge kernel are needed;
//   - each TARGET is visited in exactly one chunk, so the block's per-column
//     top-2 for that target is also complete on write -- it only has to be
//     merged across blocks, which a small second kernel does.
//
// Adapted from CudaSift (Marten Bjorkman, MIT; see ../LICENSE.CudaSift): the
// shared-memory descriptor tiling and the idea of accumulating dot products in
// registers. The bidirectional structure, the column partials, the key modes and
// the canonical tie-break are ours. NOTE this is genuinely adapted MIT code,
// unlike the kAngular path in matcher_gemm.cu which replicates SiftGPU
// BEHAVIOUR only with nothing copied -- do not conflate the two.
#include "matcher.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <climits>
#include <vector>

namespace {

constexpr int kDim = 128;
constexpr int kBM = 32;   // queries per block
constexpr int kBN = 32;   // targets per chunk
constexpr int kThreadsX = 32;
constexpr int kThreadsY = 8;
constexpr int kNRX = 2;   // queries per thread
constexpr int kMR = 4;    // targets per thread (REGISTER-TILE shape)
// Per-stripe strides for the two REDUCTIONS. These are deliberately derived from
// the tile dimensions rather than from kMR: an earlier version reused kMR as the
// stride, which covers kThreadsY*kMR items and therefore only spans the tile when
// kBN == kBM == 32 and kMR == 4 -- a coincidence of the default shape. At kMR = 2
// the column pass silently scanned 16 of 32 queries. Conflating the reduction
// shape with the register-tile shape is the bug; keep them independent so the
// tile is a free tuning parameter.
constexpr int kRowStripe = (kBN + kThreadsY - 1) / kThreadsY;  // targets/stripe
constexpr int kColStripe = (kBM + kThreadsY - 1) / kThreadsY;  // queries/stripe
// Exact division lets the reduction loops keep a COMPILE-TIME trip count, which
// is what allows nvcc to fully unroll them. Expressing the bound as
// min((ty+1)*stripe, kBN) instead makes the count runtime-variable and costs
// ~2.8 ms/pair at n=10000 -- the unroll, not the arithmetic, is the point.
// With exact division the clamp is provably redundant, so assert and drop it.
static_assert(kBN % kThreadsY == 0, "kBN must divide evenly for the unroll");
static_assert(kBM % kThreadsY == 0, "kBM must divide evenly for the unroll");
// The column pass maps tx to a TARGET, so it must be guarded when the tile is
// narrower than the block's x-extent -- but that is a compile-time question, and
// writing it as a bare `tx < kBN` leaves a runtime branch in the common
// kBN == kThreadsX case. Short-circuit on the constant so it vanishes there.
constexpr bool kTxGuardNeeded = kBN < kThreadsX;

// Same canonical rule as matcher_gemm.cu: strictly better key, or an equal key
// at a LOWER index. Required here rather than merely nice: this kernel's
// reductions and the cross-block merge visit candidates in an order that is a
// property of the thread layout, not of the data, so `v > b` alone would make
// the result depend on the decomposition and diverge from the GEMM path on ties.
__device__ __forceinline__ bool Better(float v, int i, float b, int bi) {
  return v > b || (v == b && (bi < 0 || i < bi));
}

// One block owns kBM queries and sweeps every target in kBN-wide chunks.
//
// rowSqNorm/colSqNorm follow matcher_gemm.cu's convention exactly: nullptr means
// rank on the raw inner product (what COLMAP's exact brute-force reference and
// SiftGPU do), non-null means rank on 2*dot - ||other||^2 (COLMAP's faiss/index
// convention). The two directions subtract DIFFERENT norms, so the shared tile
// holds the raw dot and each reduction applies its own key.
// One block owns kBM queries and sweeps every target in kBN-wide chunks.
//
// rowSqNorm/colSqNorm follow matcher_gemm.cu's convention exactly: nullptr means
// rank on the raw inner product (what COLMAP's exact brute-force reference and
// SiftGPU do), non-null means rank on 2*dot - ||other||^2 (COLMAP's faiss/index
// convention). The two directions subtract DIFFERENT norms, so the shared tile
// holds the raw dot and each reduction applies its own key.
//
// INNER-LOOP SHAPE, and why it is this shape. A first draft used one scalar
// query load plus four target loads per dimension for four outputs -- 1.25 shared
// floats per FMA -- and measured 21.9 ms against the GEMM path's 10.0 ms, i.e.
// the 4.41 ms of saved global traffic was handed back several times over by the
// compute loop. This follows CudaSift's proven shape instead: float4 vectorised
// shared loads and a register block of NRX x MR = 2 x 4 = 8 outputs per thread,
// giving 24 shared floats per 32 FMAs = 0.75 per FMA. The reductions also use all
// 256 threads rather than 32.
__global__ void FusedMatchKernel(
    const float* __restrict__ d1, int n1, const float* __restrict__ d2, int n2,
    const float* __restrict__ d1Sq, const float* __restrict__ d2Sq,
    bool useL2Key, float* __restrict__ rowBest, float* __restrict__ rowSecond,
    int* __restrict__ rowIdx, int* __restrict__ rowIdx2,
    float* __restrict__ colPartBest, float* __restrict__ colPartSecond,
    int* __restrict__ colPartIdx) {
  // float4 view: kDim/4 = 32 vectors per descriptor.
  __shared__ float4 qbuf[kBM][kDim / 4];
  __shared__ float4 tbuf[kBN][kDim / 4];
  __shared__ float stile[kBN][kBM + 1];   // +1 breaks column-walk bank conflicts

  const int tx = threadIdx.x;             // 0..31
  const int ty = threadIdx.y;             // 0..7
  const int idx = ty * kThreadsX + tx;    // 0..255
  const int q0 = blockIdx.x * kBM;

  // Cooperative query load, once per block. Tail-guarded: n1 need not be a
  // multiple of kBM and reading past it would pull in unrelated memory.
  for (int r = ty; r < kBM; r += kThreadsY) {
    const int q = min(q0 + r, n1 - 1);
    const bool live = (q0 + r) < n1;
    const float4* src = reinterpret_cast<const float4*>(d1 + (size_t)q * kDim);
    // Skewed (row-rotated) layout, and this is load-bearing rather than
    // cosmetic. The inner loop has all active threads reading the SAME d at the
    // same instant from 16 different rows; unrotated, those addresses are
    // 32 float4 = 512 B apart and land in one bank, a 16-way conflict that
    // serialises every query load. Measured cost of omitting it: 19.3 ms against
    // 6.8 ms for the same load+compute. The read side must apply the identical
    // rotation -- see v1 below.
    for (int d = tx; d < kDim / 4; d += kThreadsX) {
      qbuf[r][(d + r) % (kDim / 4)] =
          live ? src[d] : make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }

  // Register block: 128 active threads, each owning NRX queries x MR targets.
  const int ix = idx % (kBM / kNRX);   // 0..15 -> queries {ix, 16+ix}
  const int iy = idx / (kBM / kNRX);   // 0..7  -> targets 4*iy + {0..3}
  const bool active = idx < (kBM * kBN / kMR / kNRX);   // 32*32/4/2 = 128

  float rb[kNRX], rs[kNRX];
  int rbi[kNRX], rsi[kNRX];
#pragma unroll
  for (int i = 0; i < kNRX; ++i) {
    rb[i] = -FLT_MAX; rs[i] = -FLT_MAX; rbi[i] = -1; rsi[i] = -1;
  }

  for (int t0 = 0; t0 < n2; t0 += kBN) {
    for (int r = ty; r < kBN; r += kThreadsY) {
      const int t = min(t0 + r, n2 - 1);
      const bool live = (t0 + r) < n2;
      const float4* src = reinterpret_cast<const float4*>(d2 + (size_t)t * kDim);
      for (int d = tx; d < kDim / 4; d += kThreadsX) {
        tbuf[r][d] = live ? src[d] : make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
    __syncthreads();

    if (active) {
      float acc[kMR][kNRX];
#pragma unroll
      for (int dy = 0; dy < kMR; ++dy)
#pragma unroll
        for (int i = 0; i < kNRX; ++i) acc[dy][i] = 0.0f;

      for (int d = 0; d < kDim / 4; ++d) {
        float4 v1[kNRX];
#pragma unroll
        for (int i = 0; i < kNRX; ++i) {
          const int row = (kBM / kNRX) * i + ix;
          v1[i] = qbuf[row][(d + row) % (kDim / 4)];   // undo the store skew
        }
#pragma unroll
        for (int dy = 0; dy < kMR; ++dy) {
          const float4 v2 = tbuf[kMR * iy + dy][d];
#pragma unroll
          for (int i = 0; i < kNRX; ++i) {
            acc[dy][i] += v1[i].x * v2.x;
            acc[dy][i] += v1[i].y * v2.y;
            acc[dy][i] += v1[i].z * v2.z;
            acc[dy][i] += v1[i].w * v2.w;
          }
        }
      }
#pragma unroll
      for (int dy = 0; dy < kMR; ++dy)
#pragma unroll
        for (int i = 0; i < kNRX; ++i)
          stile[kMR * iy + dy][(kBM / kNRX) * i + ix] = acc[dy][i];
    }
    __syncthreads();

    // Row reduction across all 256 threads: 32 queries x 8 partials of 4 targets.
    // Each (tx = query, ty = stripe) thread folds 4 targets, then stripe 0 merges
    // the 8 stripes. Partials live in stile's unused padding column? No -- use
    // dedicated small shared arrays to keep stile intact for the column pass.
    __shared__ float prb[kThreadsY][kBM], prs[kThreadsY][kBM];
    __shared__ int pri[kThreadsY][kBM], prj[kThreadsY][kBM];
    {
      const int q = q0 + tx;
      float b = -FLT_MAX, sc = -FLT_MAX; int bi = -1, si = -1;
      if (q < n1) {
        for (int r = ty * kRowStripe; r < ty * kRowStripe + kRowStripe; ++r) {
          const int t = t0 + r;
          if (t >= n2) break;
          const float v = useL2Key ? (2.0f * stile[r][tx] - d2Sq[t])
                                   : stile[r][tx];
          if (Better(v, t, b, bi)) { sc = b; si = bi; b = v; bi = t; }
          else if (Better(v, t, sc, si)) { sc = v; si = t; }
        }
      }
      prb[ty][tx] = b; prs[ty][tx] = sc; pri[ty][tx] = bi; prj[ty][tx] = si;
    }
    __syncthreads();
    if (ty == 0) {
      const int q = q0 + tx;
      if (q < n1) {
#pragma unroll
        for (int k = 0; k < kThreadsY; ++k) {
          const float ob = prb[k][tx], os = prs[k][tx];
          const int oi = pri[k][tx], oj = prj[k][tx];
          if (oi < 0) continue;
          // Fold this stripe's top-2 into the running per-query top-2. Which of
          // rb[0]/rb[1] depends on nothing here: query q maps to register slot
          // determined by ix/i, so accumulate into a single logical pair via the
          // shared arrays instead -- see the write-back below.
          if (Better(ob, oi, rb[0], rbi[0])) {
            rs[0] = rb[0]; rsi[0] = rbi[0]; rb[0] = ob; rbi[0] = oi;
          } else if (Better(ob, oi, rs[0], rsi[0])) {
            rs[0] = ob; rsi[0] = oi;
          }
          if (oj >= 0 && Better(os, oj, rs[0], rsi[0])) { rs[0] = os; rsi[0] = oj; }
        }
      }
    }

    // Column reduction across all 256 threads: 32 targets x 8 partials of 4
    // queries. A target belongs to exactly one chunk, so the block's result for
    // it is COMPLETE here -- write straight out, no cross-chunk accumulation.
    __shared__ float pcb[kThreadsY][kBN], pcs[kThreadsY][kBN];
    __shared__ int pci[kThreadsY][kBN];
    // Guarded on tx < kBN: this pass maps tx to a TARGET, and kBN need not equal
    // the block's x-extent. Without the guard, shrinking kBN reads and writes
    // past stile/pcb -- caught by the fused-vs-GEMM oracle as 17/120 mismatches.
    if (!kTxGuardNeeded || tx < kBN) {
      const int t = t0 + tx;
      float b = -FLT_MAX, sc = -FLT_MAX; int bi = -1;
      if (t < n2) {
        for (int r = ty * kColStripe; r < ty * kColStripe + kColStripe; ++r) {
          const int q = q0 + r;
          if (q >= n1) break;
          const float v = useL2Key ? (2.0f * stile[tx][r] - d1Sq[q])
                                   : stile[tx][r];
          if (Better(v, q, b, bi)) { sc = b; b = v; bi = q; }
          else if (v > sc) { sc = v; }
        }
      }
      pcb[ty][tx] = b; pcs[ty][tx] = sc; pci[ty][tx] = bi;
    }
    __syncthreads();
    if (ty == 1 && (!kTxGuardNeeded || tx < kBN)) {
      const int t = t0 + tx;
      if (t < n2) {
        float b = -FLT_MAX, sc = -FLT_MAX; int bi = -1;
#pragma unroll
        for (int k = 0; k < kThreadsY; ++k) {
          const float ob = pcb[k][tx], os = pcs[k][tx];
          const int oi = pci[k][tx];
          if (oi < 0) continue;
          if (Better(ob, oi, b, bi)) { sc = fmaxf(b, fmaxf(sc, os)); b = ob; bi = oi; }
          else { sc = fmaxf(sc, fmaxf(ob, os)); }
        }
        const size_t o = (size_t)blockIdx.x * n2 + t;
        colPartBest[o] = b; colPartSecond[o] = sc; colPartIdx[o] = bi;
      }
    }
    __syncthreads();
  }

  if (ty == 0) {
    const int q = q0 + tx;
    if (q < n1) {
      rowBest[q] = rb[0]; rowSecond[q] = rs[0];
      rowIdx[q] = rbi[0]; rowIdx2[q] = rsi[0];
    }
  }
}

// Merge the per-block column partials. Ordinary top-2-of-top-2s: the winner's
// best survives and the runner-up is the largest of (loser's best, both
// runners-up). Correct including when the global best AND runner-up come from the
// same block, which is the one case a naive merge drops -- and it matters,
// because the runner-up feeds the reverse ratio test and therefore decides match
// acceptance, not just a reported value.
__global__ void ColMergeKernel(const float* __restrict__ partBest,
                               const float* __restrict__ partSecond,
                               const int* __restrict__ partIdx, int n2,
                               int numBlocks, float* __restrict__ colBest,
                               float* __restrict__ colSecond,
                               int* __restrict__ colIdx) {
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < n2;
       c += gridDim.x * blockDim.x) {
    float b = -FLT_MAX, s = -FLT_MAX;
    int bi = -1;
    for (int k = 0; k < numBlocks; ++k) {
      const size_t o = (size_t)k * n2 + c;
      const float ob = partBest[o], os = partSecond[o];
      const int oi = partIdx[o];
      if (oi < 0) continue;
      if (Better(ob, oi, b, bi)) {
        s = fmaxf(b, fmaxf(s, os));
        b = ob; bi = oi;
      } else {
        s = fmaxf(s, fmaxf(ob, os));
      }
    }
    colBest[c] = b; colSecond[c] = s; colIdx[c] = bi;
  }
}

}  // namespace

// Declared in matcher.h; see there for the parameter contract, which is
// identical to MatchDescriptorsGemmU8's so the two are interchangeable.
float MatchDescriptorsFusedRaw(const float* d1, int n1, const float* d2, int n2,
                               const float* d1Sq, const float* d2Sq,
                               bool useL2Key, float* rowBest, float* rowSecond,
                               int* rowIdx, int* rowIdx2, float* colBest,
                               float* colSecond, int* colIdx,
                               float* colPartBest, float* colPartSecond,
                               int* colPartIdx, bool wantColumns) {
  const int numBlocks = (n1 + kBM - 1) / kBM;
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);

  FusedMatchKernel<<<numBlocks, dim3(kThreadsX, kThreadsY)>>>(
      d1, n1, d2, n2, d1Sq, d2Sq, useL2Key, rowBest, rowSecond, rowIdx, rowIdx2,
      colPartBest, colPartSecond, colPartIdx);

  if (wantColumns) {
    ColMergeKernel<<<(n2 + 255) / 256, 256>>>(colPartBest, colPartSecond,
                                              colPartIdx, n2, numBlocks,
                                              colBest, colSecond, colIdx);
  }

  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return ms;
}

int MatchDescriptorsFusedRowBlock() { return kBM; }

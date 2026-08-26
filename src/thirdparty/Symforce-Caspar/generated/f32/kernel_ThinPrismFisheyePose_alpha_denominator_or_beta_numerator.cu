#include "kernel_ThinPrismFisheyePose_alpha_denominator_or_beta_numerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyePoseAlphaDenominatorOrBetaNumeratorKernel(
        float* ThinPrismFisheyePose_p_kp1,
        unsigned int ThinPrismFisheyePose_p_kp1_num_alloc,
        float* ThinPrismFisheyePose_w,
        unsigned int ThinPrismFisheyePose_w_num_alloc,
        float* const ThinPrismFisheyePose_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float ThinPrismFisheyePose_out_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyePose_p_kp1,
        0 * ThinPrismFisheyePose_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r3);
    ReadIdx4<1024, float, float, float4>(ThinPrismFisheyePose_w,
                                         0 * ThinPrismFisheyePose_w_num_alloc,
                                         global_thread_idx,
                                         r4,
                                         r5,
                                         r6,
                                         r7);
    r7 = fmaf(r3, r7, r2 * r6);
    ReadIdx2<1024, float, float, float2>(
        ThinPrismFisheyePose_p_kp1,
        4 * ThinPrismFisheyePose_p_kp1_num_alloc,
        global_thread_idx,
        r3,
        r6);
    ReadIdx2<1024, float, float, float2>(ThinPrismFisheyePose_w,
                                         4 * ThinPrismFisheyePose_w_num_alloc,
                                         global_thread_idx,
                                         r2,
                                         r8);
    r7 = fmaf(r1, r5, r7);
    r7 = fmaf(r0, r4, r7);
    r7 = fmaf(r6, r8, r7);
    r7 = fmaf(r3, r2, r7);
  };
  SumStore<float>(ThinPrismFisheyePose_out_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r7);
  SumFlushFinal<float>(
      ThinPrismFisheyePose_out_local, ThinPrismFisheyePose_out, 1);
}

void ThinPrismFisheyePoseAlphaDenominatorOrBetaNumerator(
    float* ThinPrismFisheyePose_p_kp1,
    unsigned int ThinPrismFisheyePose_p_kp1_num_alloc,
    float* ThinPrismFisheyePose_w,
    unsigned int ThinPrismFisheyePose_w_num_alloc,
    float* const ThinPrismFisheyePose_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyePoseAlphaDenominatorOrBetaNumeratorKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyePose_p_kp1,
      ThinPrismFisheyePose_p_kp1_num_alloc,
      ThinPrismFisheyePose_w,
      ThinPrismFisheyePose_w_num_alloc,
      ThinPrismFisheyePose_out,
      problem_size);
}

}  // namespace caspar
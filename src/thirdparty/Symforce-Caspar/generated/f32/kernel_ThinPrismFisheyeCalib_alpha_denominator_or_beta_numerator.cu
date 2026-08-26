#include "kernel_ThinPrismFisheyeCalib_alpha_denominator_or_beta_numerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeCalibAlphaDenominatorOrBetaNumeratorKernel(
        float* ThinPrismFisheyeCalib_p_kp1,
        unsigned int ThinPrismFisheyeCalib_p_kp1_num_alloc,
        float* ThinPrismFisheyeCalib_w,
        unsigned int ThinPrismFisheyeCalib_w_num_alloc,
        float* const ThinPrismFisheyeCalib_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float ThinPrismFisheyeCalib_out_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_p_kp1,
        4 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r3);
    ReadIdx4<1024, float, float, float4>(ThinPrismFisheyeCalib_w,
                                         4 * ThinPrismFisheyeCalib_w_num_alloc,
                                         global_thread_idx,
                                         r4,
                                         r5,
                                         r6,
                                         r7);
    r4 = fmaf(r0, r4, r1 * r5);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_p_kp1,
        0 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r5,
        r1,
        r8);
    ReadIdx4<1024, float, float, float4>(ThinPrismFisheyeCalib_w,
                                         0 * ThinPrismFisheyeCalib_w_num_alloc,
                                         global_thread_idx,
                                         r9,
                                         r10,
                                         r11,
                                         r12);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_p_kp1,
        8 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r13,
        r14,
        r15,
        r16);
    ReadIdx4<1024, float, float, float4>(ThinPrismFisheyeCalib_w,
                                         8 * ThinPrismFisheyeCalib_w_num_alloc,
                                         global_thread_idx,
                                         r17,
                                         r18,
                                         r19,
                                         r20);
    r4 = fmaf(r8, r12, r4);
    r4 = fmaf(r5, r10, r4);
    r4 = fmaf(r2, r6, r4);
    r4 = fmaf(r13, r17, r4);
    r4 = fmaf(r1, r11, r4);
    r4 = fmaf(r3, r7, r4);
    r4 = fmaf(r0, r9, r4);
    r4 = fmaf(r14, r18, r4);
    r4 = fmaf(r16, r20, r4);
    r4 = fmaf(r15, r19, r4);
  };
  SumStore<float>(ThinPrismFisheyeCalib_out_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r4);
  SumFlushFinal<float>(
      ThinPrismFisheyeCalib_out_local, ThinPrismFisheyeCalib_out, 1);
}

void ThinPrismFisheyeCalibAlphaDenominatorOrBetaNumerator(
    float* ThinPrismFisheyeCalib_p_kp1,
    unsigned int ThinPrismFisheyeCalib_p_kp1_num_alloc,
    float* ThinPrismFisheyeCalib_w,
    unsigned int ThinPrismFisheyeCalib_w_num_alloc,
    float* const ThinPrismFisheyeCalib_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeCalibAlphaDenominatorOrBetaNumeratorKernel<<<n_blocks,
                                                               1024>>>(
      ThinPrismFisheyeCalib_p_kp1,
      ThinPrismFisheyeCalib_p_kp1_num_alloc,
      ThinPrismFisheyeCalib_w,
      ThinPrismFisheyeCalib_w_num_alloc,
      ThinPrismFisheyeCalib_out,
      problem_size);
}

}  // namespace caspar
#include "kernel_ThinPrismFisheyeCalib_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeCalibAlphaNumeratorDenominatorKernel(
        float* ThinPrismFisheyeCalib_p_kp1,
        unsigned int ThinPrismFisheyeCalib_p_kp1_num_alloc,
        float* ThinPrismFisheyeCalib_r_k,
        unsigned int ThinPrismFisheyeCalib_r_k_num_alloc,
        float* ThinPrismFisheyeCalib_w,
        unsigned int ThinPrismFisheyeCalib_w_num_alloc,
        float* const ThinPrismFisheyeCalib_total_ag,
        float* const ThinPrismFisheyeCalib_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float ThinPrismFisheyeCalib_total_ag_local[1];

  __shared__ float ThinPrismFisheyeCalib_total_ac_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_p_kp1,
        8 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r3);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_r_k,
        8 * ThinPrismFisheyeCalib_r_k_num_alloc,
        global_thread_idx,
        r4,
        r5,
        r6,
        r7);
    r7 = fmaf(r3, r7, r2 * r6);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_p_kp1,
        0 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r6,
        r8,
        r9,
        r10);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_r_k,
        0 * ThinPrismFisheyeCalib_r_k_num_alloc,
        global_thread_idx,
        r11,
        r12,
        r13,
        r14);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_p_kp1,
        4 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r15,
        r16,
        r17,
        r18);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_r_k,
        4 * ThinPrismFisheyeCalib_r_k_num_alloc,
        global_thread_idx,
        r19,
        r20,
        r21,
        r22);
    r7 = fmaf(r9, r13, r7);
    r7 = fmaf(r18, r22, r7);
    r7 = fmaf(r16, r20, r7);
    r7 = fmaf(r10, r14, r7);
    r7 = fmaf(r8, r12, r7);
    r7 = fmaf(r17, r21, r7);
    r7 = fmaf(r0, r4, r7);
    r7 = fmaf(r15, r19, r7);
    r7 = fmaf(r6, r11, r7);
    r7 = fmaf(r1, r5, r7);
  };
  SumStore<float>(ThinPrismFisheyeCalib_total_ag_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r7);
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(ThinPrismFisheyeCalib_w,
                                         4 * ThinPrismFisheyeCalib_w_num_alloc,
                                         global_thread_idx,
                                         r7,
                                         r5,
                                         r11,
                                         r19);
    r7 = fmaf(r15, r7, r16 * r5);
    ReadIdx4<1024, float, float, float4>(ThinPrismFisheyeCalib_w,
                                         0 * ThinPrismFisheyeCalib_w_num_alloc,
                                         global_thread_idx,
                                         r15,
                                         r5,
                                         r16,
                                         r4);
    ReadIdx4<1024, float, float, float4>(ThinPrismFisheyeCalib_w,
                                         8 * ThinPrismFisheyeCalib_w_num_alloc,
                                         global_thread_idx,
                                         r21,
                                         r12,
                                         r14,
                                         r20);
    r7 = fmaf(r10, r4, r7);
    r7 = fmaf(r8, r5, r7);
    r7 = fmaf(r17, r11, r7);
    r7 = fmaf(r0, r21, r7);
    r7 = fmaf(r9, r16, r7);
    r7 = fmaf(r18, r19, r7);
    r7 = fmaf(r6, r15, r7);
    r7 = fmaf(r1, r12, r7);
    r7 = fmaf(r3, r20, r7);
    r7 = fmaf(r2, r14, r7);
  };
  SumStore<float>(ThinPrismFisheyeCalib_total_ac_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r7);
  SumFlushFinal<float>(
      ThinPrismFisheyeCalib_total_ag_local, ThinPrismFisheyeCalib_total_ag, 1);
  SumFlushFinal<float>(
      ThinPrismFisheyeCalib_total_ac_local, ThinPrismFisheyeCalib_total_ac, 1);
}

void ThinPrismFisheyeCalibAlphaNumeratorDenominator(
    float* ThinPrismFisheyeCalib_p_kp1,
    unsigned int ThinPrismFisheyeCalib_p_kp1_num_alloc,
    float* ThinPrismFisheyeCalib_r_k,
    unsigned int ThinPrismFisheyeCalib_r_k_num_alloc,
    float* ThinPrismFisheyeCalib_w,
    unsigned int ThinPrismFisheyeCalib_w_num_alloc,
    float* const ThinPrismFisheyeCalib_total_ag,
    float* const ThinPrismFisheyeCalib_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeCalibAlphaNumeratorDenominatorKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyeCalib_p_kp1,
      ThinPrismFisheyeCalib_p_kp1_num_alloc,
      ThinPrismFisheyeCalib_r_k,
      ThinPrismFisheyeCalib_r_k_num_alloc,
      ThinPrismFisheyeCalib_w,
      ThinPrismFisheyeCalib_w_num_alloc,
      ThinPrismFisheyeCalib_total_ag,
      ThinPrismFisheyeCalib_total_ac,
      problem_size);
}

}  // namespace caspar
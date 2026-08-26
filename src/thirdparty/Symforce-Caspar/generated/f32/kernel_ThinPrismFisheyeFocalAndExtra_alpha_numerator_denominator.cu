#include "kernel_ThinPrismFisheyeFocalAndExtra_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFocalAndExtraAlphaNumeratorDenominatorKernel(
        float* ThinPrismFisheyeFocalAndExtra_p_kp1,
        unsigned int ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        float* ThinPrismFisheyeFocalAndExtra_r_k,
        unsigned int ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        float* ThinPrismFisheyeFocalAndExtra_w,
        unsigned int ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        float* const ThinPrismFisheyeFocalAndExtra_total_ag,
        float* const ThinPrismFisheyeFocalAndExtra_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float ThinPrismFisheyeFocalAndExtra_total_ag_local[1];

  __shared__ float ThinPrismFisheyeFocalAndExtra_total_ac_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        8 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, float, float, float2>(
        ThinPrismFisheyeFocalAndExtra_r_k,
        8 * ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r2,
        r3);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        4 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r4,
        r5,
        r6,
        r7);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeFocalAndExtra_r_k,
        4 * ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r8,
        r9,
        r10,
        r11);
    r8 = fmaf(r4, r8, r1 * r3);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        0 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r3,
        r12,
        r13,
        r14);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeFocalAndExtra_r_k,
        0 * ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r15,
        r16,
        r17,
        r18);
    r8 = fmaf(r12, r16, r8);
    r8 = fmaf(r13, r17, r8);
    r8 = fmaf(r7, r11, r8);
    r8 = fmaf(r5, r9, r8);
    r8 = fmaf(r0, r2, r8);
    r8 = fmaf(r6, r10, r8);
    r8 = fmaf(r14, r18, r8);
    r8 = fmaf(r3, r15, r8);
  };
  SumStore<float>(ThinPrismFisheyeFocalAndExtra_total_ag_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r8);
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeFocalAndExtra_w,
        0 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r8,
        r15,
        r18,
        r10);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeFocalAndExtra_w,
        4 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r2,
        r9,
        r11,
        r17);
    r17 = fmaf(r7, r17, r3 * r8);
    ReadIdx2<1024, float, float, float2>(
        ThinPrismFisheyeFocalAndExtra_w,
        8 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r7,
        r8);
    r17 = fmaf(r4, r2, r17);
    r17 = fmaf(r13, r18, r17);
    r17 = fmaf(r14, r10, r17);
    r17 = fmaf(r0, r7, r17);
    r17 = fmaf(r1, r8, r17);
    r17 = fmaf(r5, r9, r17);
    r17 = fmaf(r12, r15, r17);
    r17 = fmaf(r6, r11, r17);
  };
  SumStore<float>(ThinPrismFisheyeFocalAndExtra_total_ac_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r17);
  SumFlushFinal<float>(ThinPrismFisheyeFocalAndExtra_total_ag_local,
                       ThinPrismFisheyeFocalAndExtra_total_ag,
                       1);
  SumFlushFinal<float>(ThinPrismFisheyeFocalAndExtra_total_ac_local,
                       ThinPrismFisheyeFocalAndExtra_total_ac,
                       1);
}

void ThinPrismFisheyeFocalAndExtraAlphaNumeratorDenominator(
    float* ThinPrismFisheyeFocalAndExtra_p_kp1,
    unsigned int ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
    float* ThinPrismFisheyeFocalAndExtra_r_k,
    unsigned int ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
    float* ThinPrismFisheyeFocalAndExtra_w,
    unsigned int ThinPrismFisheyeFocalAndExtra_w_num_alloc,
    float* const ThinPrismFisheyeFocalAndExtra_total_ag,
    float* const ThinPrismFisheyeFocalAndExtra_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFocalAndExtraAlphaNumeratorDenominatorKernel<<<n_blocks,
                                                                 1024>>>(
      ThinPrismFisheyeFocalAndExtra_p_kp1,
      ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
      ThinPrismFisheyeFocalAndExtra_r_k,
      ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
      ThinPrismFisheyeFocalAndExtra_w,
      ThinPrismFisheyeFocalAndExtra_w_num_alloc,
      ThinPrismFisheyeFocalAndExtra_total_ag,
      ThinPrismFisheyeFocalAndExtra_total_ac,
      problem_size);
}

}  // namespace caspar
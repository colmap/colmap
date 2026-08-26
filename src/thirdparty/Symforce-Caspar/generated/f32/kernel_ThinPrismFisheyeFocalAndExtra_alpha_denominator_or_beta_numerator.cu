#include "kernel_ThinPrismFisheyeFocalAndExtra_alpha_denominator_or_beta_numerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFocalAndExtraAlphaDenominatorOrBetaNumeratorKernel(
        float* ThinPrismFisheyeFocalAndExtra_p_kp1,
        unsigned int ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        float* ThinPrismFisheyeFocalAndExtra_w,
        unsigned int ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        float* const ThinPrismFisheyeFocalAndExtra_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float ThinPrismFisheyeFocalAndExtra_out_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        0 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r3);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeFocalAndExtra_w,
        0 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r4,
        r5,
        r6,
        r7);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        4 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r8,
        r9,
        r10,
        r11);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeFocalAndExtra_w,
        4 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r12,
        r13,
        r14,
        r15);
    r15 = fmaf(r11, r15, r0 * r4);
    ReadIdx2<1024, float, float, float2>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        8 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r11,
        r4);
    ReadIdx2<1024, float, float, float2>(
        ThinPrismFisheyeFocalAndExtra_w,
        8 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r0,
        r16);
    r15 = fmaf(r8, r12, r15);
    r15 = fmaf(r2, r6, r15);
    r15 = fmaf(r3, r7, r15);
    r15 = fmaf(r11, r0, r15);
    r15 = fmaf(r4, r16, r15);
    r15 = fmaf(r9, r13, r15);
    r15 = fmaf(r1, r5, r15);
    r15 = fmaf(r10, r14, r15);
  };
  SumStore<float>(ThinPrismFisheyeFocalAndExtra_out_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r15);
  SumFlushFinal<float>(ThinPrismFisheyeFocalAndExtra_out_local,
                       ThinPrismFisheyeFocalAndExtra_out,
                       1);
}

void ThinPrismFisheyeFocalAndExtraAlphaDenominatorOrBetaNumerator(
    float* ThinPrismFisheyeFocalAndExtra_p_kp1,
    unsigned int ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
    float* ThinPrismFisheyeFocalAndExtra_w,
    unsigned int ThinPrismFisheyeFocalAndExtra_w_num_alloc,
    float* const ThinPrismFisheyeFocalAndExtra_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFocalAndExtraAlphaDenominatorOrBetaNumeratorKernel<<<n_blocks,
                                                                       1024>>>(
      ThinPrismFisheyeFocalAndExtra_p_kp1,
      ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
      ThinPrismFisheyeFocalAndExtra_w,
      ThinPrismFisheyeFocalAndExtra_w_num_alloc,
      ThinPrismFisheyeFocalAndExtra_out,
      problem_size);
}

}  // namespace caspar
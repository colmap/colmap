#include "kernel_ThinPrismFisheyeFocalAndExtra_pred_decrease_times_two.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFocalAndExtraPredDecreaseTimesTwoKernel(
        float* ThinPrismFisheyeFocalAndExtra_step,
        unsigned int ThinPrismFisheyeFocalAndExtra_step_num_alloc,
        float* ThinPrismFisheyeFocalAndExtra_precond_diag,
        unsigned int ThinPrismFisheyeFocalAndExtra_precond_diag_num_alloc,
        const float* const diag,
        float* ThinPrismFisheyeFocalAndExtra_njtr,
        unsigned int ThinPrismFisheyeFocalAndExtra_njtr_num_alloc,
        float* const out_ThinPrismFisheyeFocalAndExtra_pred_dec,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ float out_ThinPrismFisheyeFocalAndExtra_pred_dec_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        ThinPrismFisheyeFocalAndExtra_step,
        8 * ThinPrismFisheyeFocalAndExtra_step_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, float, float, float2>(
        ThinPrismFisheyeFocalAndExtra_njtr,
        8 * ThinPrismFisheyeFocalAndExtra_njtr_num_alloc,
        global_thread_idx,
        r2,
        r3);
    ReadIdx2<1024, float, float, float2>(
        ThinPrismFisheyeFocalAndExtra_precond_diag,
        8 * ThinPrismFisheyeFocalAndExtra_precond_diag_num_alloc,
        global_thread_idx,
        r4,
        r5);
    r6 = r1 * r5;
  };
  LoadUnique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r6 = fmaf(r7, r6, r3);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeFocalAndExtra_step,
        0 * ThinPrismFisheyeFocalAndExtra_step_num_alloc,
        global_thread_idx,
        r3,
        r8,
        r9,
        r10);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeFocalAndExtra_njtr,
        0 * ThinPrismFisheyeFocalAndExtra_njtr_num_alloc,
        global_thread_idx,
        r11,
        r12,
        r13,
        r14);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeFocalAndExtra_precond_diag,
        0 * ThinPrismFisheyeFocalAndExtra_precond_diag_num_alloc,
        global_thread_idx,
        r15,
        r16,
        r17,
        r18);
    r19 = r9 * r17;
    r19 = fmaf(r7, r19, r13);
    r19 = fmaf(r9, r19, r1 * r6);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeFocalAndExtra_step,
        4 * ThinPrismFisheyeFocalAndExtra_step_num_alloc,
        global_thread_idx,
        r6,
        r13,
        r20,
        r21);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeFocalAndExtra_njtr,
        4 * ThinPrismFisheyeFocalAndExtra_njtr_num_alloc,
        global_thread_idx,
        r22,
        r23,
        r24,
        r25);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeFocalAndExtra_precond_diag,
        4 * ThinPrismFisheyeFocalAndExtra_precond_diag_num_alloc,
        global_thread_idx,
        r26,
        r27,
        r28,
        r29);
    r30 = r21 * r29;
    r30 = fmaf(r7, r30, r25);
    r25 = r3 * r15;
    r25 = fmaf(r7, r25, r11);
    r11 = r0 * r4;
    r11 = fmaf(r7, r11, r2);
    r2 = r20 * r28;
    r2 = fmaf(r7, r2, r24);
    r24 = r6 * r26;
    r24 = fmaf(r7, r24, r22);
    r22 = r13 * r27;
    r22 = fmaf(r7, r22, r23);
    r23 = r10 * r18;
    r23 = fmaf(r7, r23, r14);
    r14 = r8 * r16;
    r14 = fmaf(r7, r14, r12);
    r19 = fmaf(r21, r30, r19);
    r19 = fmaf(r3, r25, r19);
    r19 = fmaf(r0, r11, r19);
    r19 = fmaf(r20, r2, r19);
    r19 = fmaf(r6, r24, r19);
    r19 = fmaf(r13, r22, r19);
    r19 = fmaf(r10, r23, r19);
    r19 = fmaf(r8, r14, r19);
  };
  SumStore<float>(out_ThinPrismFisheyeFocalAndExtra_pred_dec_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r19);
  SumFlushFinal<float>(out_ThinPrismFisheyeFocalAndExtra_pred_dec_local,
                       out_ThinPrismFisheyeFocalAndExtra_pred_dec,
                       1);
}

void ThinPrismFisheyeFocalAndExtraPredDecreaseTimesTwo(
    float* ThinPrismFisheyeFocalAndExtra_step,
    unsigned int ThinPrismFisheyeFocalAndExtra_step_num_alloc,
    float* ThinPrismFisheyeFocalAndExtra_precond_diag,
    unsigned int ThinPrismFisheyeFocalAndExtra_precond_diag_num_alloc,
    const float* const diag,
    float* ThinPrismFisheyeFocalAndExtra_njtr,
    unsigned int ThinPrismFisheyeFocalAndExtra_njtr_num_alloc,
    float* const out_ThinPrismFisheyeFocalAndExtra_pred_dec,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFocalAndExtraPredDecreaseTimesTwoKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyeFocalAndExtra_step,
      ThinPrismFisheyeFocalAndExtra_step_num_alloc,
      ThinPrismFisheyeFocalAndExtra_precond_diag,
      ThinPrismFisheyeFocalAndExtra_precond_diag_num_alloc,
      diag,
      ThinPrismFisheyeFocalAndExtra_njtr,
      ThinPrismFisheyeFocalAndExtra_njtr_num_alloc,
      out_ThinPrismFisheyeFocalAndExtra_pred_dec,
      problem_size);
}

}  // namespace caspar